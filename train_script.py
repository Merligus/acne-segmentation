import re
import os
import cv2
import numpy as np
import torch
import torch.nn as nn  # Keep for loss functions
import torch.nn.functional as F  # Needed for SegFormer upsampling
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

# Import metrics directly from smp.metrics
from segmentation_models_pytorch.metrics import get_stats, iou_score, f1_score
from segmentation_models_pytorch.losses import DiceLoss, SoftBCEWithLogitsLoss
import albumentations as albu
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time  # To add timestamps
import argparse  # To handle command-line arguments

# --- Argument Parser
parser = argparse.ArgumentParser(
    description="Train UNet or SegFormer for Acne Segmentation."
)
parser.add_argument(
    "--architecture",
    type=str,
    default="Unet",
    choices=["Unet", "SegFormer"],
    help="Model architecture to use: Unet or SegFormer. Default: Unet",
)
parser.add_argument(
    "--unet_encoder",
    type=str,
    default="efficientnet-b4",
    help="Encoder for Unet architecture (ignored if architecture=SegFormer). Default: efficientnet-b4",
)
parser.add_argument(
    "--segformer_model",
    type=str,
    default="nvidia/segformer-b0-finetuned-ade-512-512",
    help="Pretrained SegFormer model name from Hugging Face Hub (ignored if architecture=Unet). Default: nvidia/segformer-b0-finetuned-ade-512-512",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="",
    help="Optional path to a checkpoint file to resume training.",
)
# Add other arguments if you want to control them via CLI
parser.add_argument(
    "--epochs", type=int, default=200, help="Number of training epochs."
)
parser.add_argument(
    "--lr",
    type=float,
    default=None,
    help="Learning rate (default chosen based on architecture).",
)
parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
parser.add_argument(
    "--img_size",
    type=int,
    nargs=2,
    default=[256, 256],
    help="Target image size (height width).",
)


args = parser.parse_args()

# --- Conditional Imports ---
if args.architecture == "SegFormer":
    try:
        from transformers import SegformerForSemanticSegmentation, AutoImageProcessor
    except ImportError:
        print(
            "Error: 'transformers' library not found. Please install it (`pip install transformers`) to use SegFormer."
        )
        exit()

# --- Configuration ---
MODEL_ARCHITECTURE = args.architecture
CHECKPOINT_DIR = "./checkpoint/"
CHECKPOINT_PATH = args.checkpoint  # Get from argparse
DATA_DIR = "./data/"
IMAGES_DIR = os.path.join(DATA_DIR, "JPEGImages")
MASKS_DIR = os.path.join(DATA_DIR, "mask")
TRAIN_FILE = os.path.join(DATA_DIR, "with_mask_train.txt")
TEST_FILE = os.path.join(DATA_DIR, "with_mask_test.txt")

# --- Model-Specific Configuration ---
if MODEL_ARCHITECTURE == "Unet":
    ENCODER = args.unet_encoder
    ENCODER_WEIGHTS = "imagenet"
    MODEL_NAME_FOR_SAVE = f"unet_{ENCODER}"  # For saving files
    # Default LR for Unet/CNNs
    DEFAULT_LR = 0.001
elif MODEL_ARCHITECTURE == "SegFormer":
    MODEL_NAME = args.segformer_model
    MODEL_NAME_FOR_SAVE = (
        f"segformer_{MODEL_NAME.replace('/', '_')}"  # For saving files
    )
    # Default LR for Transformers
    DEFAULT_LR = 6e-5
    # Define labels needed for SegFormer head initialization
    id2label = {0: "background", 1: "acne"}
    label2id = {v: k for k, v in id2label.items()}
else:
    raise ValueError(f"Unsupported architecture: {MODEL_ARCHITECTURE}")

ACTIVATION = None  # Common for both when using logits loss
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Hyperparameters
IMG_SIZE = tuple(args.img_size)
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = args.lr if args.lr is not None else DEFAULT_LR  # Use specified LR or default


# --- Dataset Class ---
class AcneDataset(BaseDataset):
    """
    Reads images and masks based on filenames listed in a text file.
    Handles varying image sizes and ensures binary masks (0/1).
    Applies preprocessing based on the chosen model architecture.
    """

    def __init__(
        self,
        images_dir,
        masks_dir,
        file_list_path,
        architecture,  # Pass the architecture choice
        unet_encoder=None,  # Needed for SMP preprocessing
        unet_encoder_weights=None,  # Needed for SMP preprocessing
        target_img_size=(256, 256),
        augmentation=None,
    ):
        self.image_paths = []
        self.mask_paths = []
        self.architecture = architecture
        self.target_img_size = target_img_size
        skipped_files = 0

        print(f"Reading file list from: {file_list_path}")
        try:
            ext = ".jpg"
            with open(file_list_path, "r") as f:
                for line_num, line in enumerate(f):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    filename = parts[0]

                    img_path = os.path.join(images_dir, filename)
                    base_name = os.path.splitext(filename)[0]
                    mask_filename = base_name + ext
                    mask_path = os.path.join(masks_dir, mask_filename)

                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)
                    else:
                        skipped_files += 1
        except FileNotFoundError:
            print(f"Error: File not found at {file_list_path}")
            raise
        if not self.image_paths:
            raise RuntimeError(
                f"No valid image/mask pairs found from {file_list_path}."
            )
        if skipped_files > 0:
            print(f"Warning: Skipped {skipped_files} files (missing image or mask).")

        self.augmentation = augmentation

        # --- Conditional Preprocessing Setup ---
        if self.architecture == "Unet":
            if unet_encoder is None or unet_encoder_weights is None:
                raise ValueError(
                    "Unet encoder and weights must be provided for Unet architecture."
                )
            self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
                unet_encoder, unet_encoder_weights
            )
        elif self.architecture == "SegFormer":
            # Use standard ImageNet normalization, applied manually after augmentation
            self.normalize_transform = albu.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )
        else:
            # Fallback or error if needed
            self.normalize_transform = albu.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )
            print(
                f"Warning: Unknown architecture '{self.architecture}' in Dataset. Using default ImageNet normalization."
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        # Read image
        try:
            image = cv2.imread(self.image_paths[i])
            if image is None:
                raise IOError(f"Could not read image: {self.image_paths[i]}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Read mask - read as grayscale
            mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise IOError(f"Could not read mask: {self.mask_paths[i]}")

            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

            # Apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]

            # --- Apply Conditional Preprocessing ---
            if self.architecture == "Unet":
                image = self.preprocessing_fn(image)  # Apply SMP normalization
                image = (
                    torch.from_numpy(image).permute(2, 0, 1).float()
                )  # HWC -> CHW tensor
            elif self.architecture == "SegFormer":
                image = self.normalize_transform(image=image)[
                    "image"
                ]  # Apply manual normalization
                image = (
                    torch.from_numpy(image).permute(2, 0, 1).float()
                )  # HWC -> CHW tensor
            else:  # Default/Fallback
                image = self.normalize_transform(image=image)["image"]
                image = torch.from_numpy(image).permute(2, 0, 1).float()

            # Mask to Tensor (common step)
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=0)  # 1, H, W
            mask = torch.from_numpy(mask).float()

        except Exception as e:
            # print(f"Error loading item {i}: Image: {self.image_paths[i]}, Mask: {self.mask_paths[i]}, Error: {e}")
            dummy_image = torch.zeros(
                (3, self.target_img_size[0], self.target_img_size[1]),
                dtype=torch.float32,
            )
            dummy_mask = torch.zeros(
                (1, self.target_img_size[0], self.target_img_size[1]),
                dtype=torch.float32,
            )
            return dummy_image, dummy_mask

        return image, mask


def custom_collate_fn(batch):
    """
    Custom collate function to handle batching of images and masks,
    ensuring correct types and handling potential None items from dataset errors.
    """
    # Filter out None items that might be returned by __getitem__ on error
    batch = [item for item in batch if item is not None]
    if not batch:
        # Return None if the batch is empty after filtering
        # This needs to be handled in the training/validation loop
        return None

    # Separate images and masks
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch]

    # Stack images and masks into batch tensors
    # Ensure they are float32 before stacking
    try:
        images_batch = torch.stack([img.float() for img in images])
        masks_batch = torch.stack([mask.float() for mask in masks])
    except RuntimeError as e:
        print(f"Error during torch.stack in custom_collate_fn: {e}")
        # Print shapes for debugging if stacking fails (e.g., due to inconsistent sizes)
        for i, img in enumerate(images):
            print(f"Image {i} shape: {img.shape}")
        for i, mask in enumerate(masks):
            print(f"Mask {i} shape: {mask.shape}")
        # Return None or raise error
        return None  # Propagate error as None batch

    return images_batch, masks_batch


# --- Augmentations ---
def get_training_augmentation(img_size):
    max_size = max(img_size)  # Get the target size for the longest side
    train_transform = [
        # Resize longest side to max_size, maintaining aspect ratio
        albu.LongestMaxSize(max_size=max_size, interpolation=cv2.INTER_LINEAR),
        # Pad to target square size (img_size[0] x img_size[1])
        # Apply padding BEFORE other spatial transforms that might distort it significantly
        albu.PadIfNeeded(
            min_height=img_size[0],
            min_width=img_size[1],
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,  # Explicit mask_value=0
        ),
        # Other spatial augmentations (applied to padded image)
        albu.HorizontalFlip(p=0.5),
        # ShiftScaleRotate might introduce padding; ensure it uses constant border mode
        albu.ShiftScaleRotate(
            scale_limit=0.1,
            rotate_limit=15,
            shift_limit=0.1,
            p=0.7,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
        ),
        # Random crops can be useful after padding if images are large, but ensure size consistency
        # albu.RandomCrop(height=img_size[0], width=img_size[1], always_apply=True), # Optional: If needed after padding
        # Non-spatial augmentations
        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.3),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=1
                ),
                albu.RandomGamma(p=1),
            ],
            p=0.7,  # Apply one of these color transforms 70% of the time
        ),
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.5,  # Apply one of these blur/sharpen transforms 50% of the time
        ),
        albu.OneOf(
            [
                albu.HueSaturationValue(p=1),
            ],
            p=0.3,  # Apply Hue/Saturation changes 30% of the time
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(img_size):
    """Add paddings to make image shape divisible by 32"""
    max_size = max(img_size)
    test_transform = [
        albu.LongestMaxSize(max_size=max_size, interpolation=cv2.INTER_LINEAR),
        albu.PadIfNeeded(
            min_height=img_size[0],
            min_width=img_size[1],
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
        ),
    ]
    return albu.Compose(test_transform)


# --- Model Definition (Conditional) ---
print(f"--- Defining Model ---")
print(f"Architecture: {MODEL_ARCHITECTURE}")
if MODEL_ARCHITECTURE == "Unet":
    print(f"Unet Encoder: {ENCODER}")
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,  # Input channels (RGB)
        classes=1,  # Output channels (binary: acne vs background)
        activation=ACTIVATION,  # Keep None for BCEWithLogitsLoss
    )
elif MODEL_ARCHITECTURE == "SegFormer":
    print(f"SegFormer Model: {MODEL_NAME}")
    try:
        model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_NAME,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
    except Exception as e:
        print(f"Error initializing SegFormer: {e}")
        exit()

model.to(DEVICE)  # Move model to device AFTER definition

# --- Load Checkpoint (Conditional Check) ---
max_iou_score = 0.0
if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
    print(f"Loading weights from checkpoint: {CHECKPOINT_PATH}")
    try:
        # Load the checkpoint dictionary
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        state_dict = checkpoint
        # Try to load the model state_dict
        # Check if the checkpoint is the state_dict itself or a dictionary containing it
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print("  -> Loaded 'model_state_dict' from checkpoint dictionary.")
        elif (
            isinstance(checkpoint, dict) and "state_dict" in checkpoint
        ):  # Some frameworks use 'state_dict'
            state_dict = checkpoint["state_dict"]
            print("  -> Loaded 'state_dict' from checkpoint dictionary.")

        # Use strict=False for SegFormer due to head mismatch, strict=True might work for resuming Unet
        load_strict = MODEL_ARCHITECTURE == "Unet"
        # Load the state dictionary into the model
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict, strict=load_strict
        )

        if missing_keys:
            print(f"  Warning: Missing keys in state_dict: {missing_keys}")
        if unexpected_keys:
            print(f"  Warning: Unexpected keys in state_dict: {unexpected_keys}")
        print(f"  -> Weights loaded successfully into model (strict={load_strict}).")

        # Try to parse max_iou_score from filename
        pattern = r"iou(\d+\.\d+)\.pth$"
        base_filename = os.path.basename(CHECKPOINT_PATH)
        match = re.search(pattern, base_filename)
        if match:
            try:
                max_iou_score = float(match.group(1))
            except:
                pass  # Keep 0.0 if parsing fails
            print(f"  -> Resuming with Max IoU Score: {max_iou_score:.4f}")

    except Exception as e:
        print(
            f"  Error loading checkpoint: {e}. Starting with initialized/pre-trained weights."
        )
else:
    print(f"No valid checkpoint path provided or found ({CHECKPOINT_PATH}).")
    print("Starting with initialized/pre-trained weights.")


# --- Datasets & Dataloaders ---
# Pass architecture info to Dataset
train_dataset = AcneDataset(
    IMAGES_DIR,
    MASKS_DIR,
    TRAIN_FILE,
    architecture=MODEL_ARCHITECTURE,
    unet_encoder=ENCODER if MODEL_ARCHITECTURE == "Unet" else None,
    unet_encoder_weights=ENCODER_WEIGHTS if MODEL_ARCHITECTURE == "Unet" else None,
    target_img_size=IMG_SIZE,
    augmentation=get_training_augmentation(IMG_SIZE),
)
valid_dataset = AcneDataset(
    IMAGES_DIR,
    MASKS_DIR,
    TEST_FILE,
    architecture=MODEL_ARCHITECTURE,
    unet_encoder=ENCODER if MODEL_ARCHITECTURE == "Unet" else None,
    unet_encoder_weights=ENCODER_WEIGHTS if MODEL_ARCHITECTURE == "Unet" else None,
    target_img_size=IMG_SIZE,
    augmentation=get_validation_augmentation(IMG_SIZE),
)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(valid_dataset)}")

if len(train_dataset) == 0 or len(valid_dataset) == 0:
    print("Error: One or both datasets are empty. Please check file paths and content.")
    exit()


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=max(1, os.cpu_count() // 2),
    pin_memory=True,
    drop_last=True,
    collate_fn=custom_collate_fn,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=max(1, os.cpu_count() // 2),
    pin_memory=True,
    drop_last=False,
    collate_fn=custom_collate_fn,
)


# --- Loss Function (Conditional Upsampling needed) ---
dice_loss = DiceLoss(mode="binary", from_logits=True).to(DEVICE)
bce_loss = SoftBCEWithLogitsLoss().to(DEVICE)
LOSS_WEIGHT_DICE = 0.6
LOSS_WEIGHT_BCE = 0.4


def combined_loss(y_pred_logits, y_true, architecture):
    # Ensure y_true is on the correct device AND is float32
    y_true_dev = y_true.to(device=y_pred_logits.device, dtype=torch.float32)

    # --- Conditional Upsampling for SegFormer ---
    if architecture == "SegFormer":
        target_size = y_true_dev.shape[-2:]  # Get (H, W) from mask
        y_pred_logits = F.interpolate(
            y_pred_logits, size=target_size, mode="bilinear", align_corners=False
        )
        # Handle potential 2-channel output from SegFormer for binary case
        if y_pred_logits.shape[1] == 2:
            y_pred_logits = y_pred_logits[
                :, 1:2, :, :
            ]  # Select class 1, keep channel dim
        elif y_pred_logits.shape[1] != 1:
            # Log error or warning, default to channel 1 if > 1
            # print(f"Warning: Unexpected SegFormer logit channels: {y_pred_logits.shape[1]}. Using channel 1.")
            y_pred_logits = y_pred_logits[
                :, 1:2, :, :
            ]  # Assuming channel 1 is foreground

    # Ensure y_pred is float32
    y_pred_float = y_pred_logits.float()

    loss1 = dice_loss(y_pred_float, y_true_dev)
    loss2 = bce_loss(y_pred_float, y_true_dev)
    return LOSS_WEIGHT_DICE * loss1 + LOSS_WEIGHT_BCE * loss2


# --- Optimizer ---
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

# --- Learning Rate Scheduler ---
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=LR / 100
)

# --- Training Loop (Conditional Forward Pass & Upsampling for Metrics) ---
history = {"train_loss": [], "valid_loss": [], "iou_score": [], "f1_score": []}
start_time = time.time()

print(f"\n--- Starting Training ---")
print(f"Architecture: {MODEL_ARCHITECTURE}")
if MODEL_ARCHITECTURE == "Unet":
    print(f"Unet Encoder: {ENCODER}")
if MODEL_ARCHITECTURE == "SegFormer":
    print(f"SegFormer Model: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"Image Size: {IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Initial Learning Rate: {LR}")
print(f"Resuming with Max IoU: {max_iou_score:.4f}")
print(f"-------------------------")

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    print(f"\nEpoch: {epoch+1}/{EPOCHS}")

    # --- Training Phase ---
    model.train()
    train_loss = 0.0
    pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False)

    for batch_idx, batch_data in enumerate(pbar_train):
        if batch_data is None:
            print(
                f"Warning: Skipping training batch {batch_idx} due to data loading error."
            )
            continue
        images, masks_true = batch_data
        images = images.to(DEVICE).float()  # Ensure float input
        masks_true = masks_true.to(DEVICE).float()

        optimizer.zero_grad()

        # --- Conditional Forward Pass ---
        if MODEL_ARCHITECTURE == "Unet":
            masks_pred_logits = model(images)  # Direct logits output
        elif MODEL_ARCHITECTURE == "SegFormer":
            outputs = model(pixel_values=images)
            masks_pred_logits = outputs.logits  # Logits are in outputs.logits

        loss = combined_loss(masks_pred_logits, masks_true, MODEL_ARCHITECTURE)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar_train.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = train_loss / len(pbar_train) if len(pbar_train) > 0 else 0
    history["train_loss"].append(avg_train_loss)

    # --- Validation Phase ---
    model.eval()
    valid_loss = 0.0
    # Initialize lists to store stats per batch
    tp_total, fp_total, fn_total, tn_total = [], [], [], []
    pbar_valid = tqdm(valid_loader, desc=f"Epoch {epoch+1} Validation", leave=False)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(pbar_valid):
            if batch_data is None:
                print(
                    f"Warning: Skipping validation batch {batch_idx} due to data loading error."
                )
                continue
            images, masks_true = batch_data
            images = images.to(DEVICE).float()
            masks_true = masks_true.to(DEVICE).float()

            # --- Conditional Forward Pass & Upsampling ---
            if MODEL_ARCHITECTURE == "Unet":
                masks_pred_logits = model(images)
                # No upsampling needed for Unet if output matches target size
                # If Unet output size differs, upsampling might be needed here too
                upsampled_logits = masks_pred_logits  # Assume same size for now
            elif MODEL_ARCHITECTURE == "SegFormer":
                outputs = model(pixel_values=images)
                masks_pred_logits = outputs.logits
                # Upsample SegFormer logits for metrics/loss
                target_size = masks_true.shape[-2:]
                upsampled_logits = F.interpolate(
                    masks_pred_logits,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
                # Handle SegFormer's potential 2-channel output
                if upsampled_logits.shape[1] == 2:
                    upsampled_logits = upsampled_logits[:, 1:2, :, :]
                elif upsampled_logits.shape[1] != 1:
                    print(
                        f"Warning: Skipping metrics val batch {batch_idx}, unexpected SegFormer channels: {upsampled_logits.shape[1]}"
                    )
                    continue  # Skip metrics calculation for this batch

            # Calculate loss using original (potentially non-upsampled for Unet) logits
            loss = combined_loss(masks_pred_logits, masks_true, MODEL_ARCHITECTURE)
            valid_loss += loss.item()

            # Calculate metrics using upsampled logits
            masks_pred_prob = torch.sigmoid(
                upsampled_logits.float()
            )  # Ensure float for sigmoid
            # Threshold probabilities -> binary masks (0 or 1)
            masks_pred_binary = (masks_pred_prob > 0.5).long()

            # Ensure masks_true is integer type for metrics
            masks_true_int = masks_true.long()

            # Calculate tp, fp, fn, tn for the batch
            tp, fp, fn, tn = get_stats(masks_pred_binary, masks_true_int, mode="binary")

            # Append batch stats to lists
            tp_total.append(tp)
            fp_total.append(fp)
            fn_total.append(fn)
            tn_total.append(tn)

    # Aggregate stats
    if tp_total:  # Check if any valid batches were processed
        tp_agg = torch.cat(tp_total).sum()
        fp_agg = torch.cat(fp_total).sum()
        fn_agg = torch.cat(fn_total).sum()
        tn_agg = torch.cat(tn_total).sum()
        # Calculate metrics
        avg_iou = iou_score(tp_agg, fp_agg, fn_agg, tn_agg, reduction="micro")
        avg_f1 = f1_score(tp_agg, fp_agg, fn_agg, tn_agg, reduction="micro")
    else:
        avg_iou = torch.tensor(0.0)
        avg_f1 = torch.tensor(0.0)
        print(
            "Warning: No validation metrics calculated (loader empty or batch errors)."
        )

    avg_valid_loss = valid_loss / len(pbar_valid) if len(pbar_valid) > 0 else 0

    history["valid_loss"].append(avg_valid_loss)
    history["iou_score"].append(avg_iou.cpu().item())
    history["f1_score"].append(avg_f1.cpu().item())

    # Update learning rate scheduler
    scheduler.step()  # CosineAnnealingLR steps each epoch

    # Print epoch summary
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1} Summary:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Valid Loss: {avg_valid_loss:.4f}")
    print(f"  Valid IoU:  {avg_iou:.4f}")
    print(f"  Valid F1:   {avg_f1:.4f}")
    print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")  # Corrected LR access
    print(f"  Time:       {epoch_time:.2f}s")

    # Save best model based on IoU score
    current_iou = avg_iou.item()
    if current_iou > max_iou_score:
        max_iou_score = current_iou
        # Use MODEL_NAME_FOR_SAVE for consistent naming
        model_save_path = f"best_model_{MODEL_NAME_FOR_SAVE}_epoch{epoch+1}_iou{max_iou_score:.4f}.pth"
        SAVE_PATH = os.path.join(CHECKPOINT_DIR, model_save_path)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  >> Model Saved: {SAVE_PATH} (IoU: {max_iou_score:.4f})")
    else:
        print(f"  (IoU did not improve from {max_iou_score:.4f})")

# --- Final Summary & Plotting ---
total_training_time = time.time() - start_time
print(f"\n--- Training Finished ---")
print(
    f"Total Time: {total_training_time // 3600:.0f}h {(total_training_time % 3600) // 60:.0f}m {total_training_time % 60:.2f}s"
)
print(f"Best Validation IoU for {MODEL_ARCHITECTURE}: {max_iou_score:.4f}")

# --- Plotting History ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["valid_loss"], label="Valid Loss")
plt.title("Loss History")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(history["iou_score"], label="IoU Score")
plt.plot(history["f1_score"], label="F1 Score (Dice)")
plt.title("Metrics History (Validation)")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
history_plot_path = f"training_history_{MODEL_NAME_FOR_SAVE}.png"
plt.savefig(history_plot_path)
print(f"Training history plot saved to: {history_plot_path}")
# plt.show()
