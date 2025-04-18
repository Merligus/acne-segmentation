import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

# Import metrics directly from smp.metrics
from segmentation_models_pytorch.metrics import (
    get_stats,
    iou_score,
    f1_score,
    accuracy,
    precision,
    recall,
)
from segmentation_models_pytorch.losses import DiceLoss, SoftBCEWithLogitsLoss
import albumentations as albu
from albumentations.pytorch import ToTensorV2  # Use ToTensorV2 for PyTorch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time  # To add timestamps

# --- Configuration ---
CHECKPOINT_DIR = "./checkpoint/"
DATA_DIR = "./data/"
IMAGES_DIR = os.path.join(DATA_DIR, "JPEGImages")
MASKS_DIR = os.path.join(DATA_DIR, "mask")
TRAIN_FILE = os.path.join(DATA_DIR, "with_mask_train.txt")
TEST_FILE = os.path.join(DATA_DIR, "with_mask_test.txt")

ENCODER = "mobilenet_v2"  # Choose a lightweight encoder
ENCODER_WEIGHTS = "imagenet"
ACTIVATION = None  # Use None for logits output, required by BCEWithLogitsLoss/DiceLoss
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Hyperparameters
IMG_SIZE = (256, 256)  # Resize images/masks to this size
BATCH_SIZE = 8
EPOCHS = 25
LR = 0.001


# --- Dataset Class ---
class AcneDataset(BaseDataset):
    """
    Reads images and masks based on filenames listed in a text file.
    Handles varying image sizes and ensures binary masks (0/1).
    """

    def __init__(
        self,
        images_dir,
        masks_dir,
        file_list_path,
        target_img_size=(256, 256),
        augmentation=None,
    ):
        self.image_paths = []
        self.mask_paths = []
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
                f"No valid image/mask pairs found after checking {file_list_path}. Please check paths and filenames."
            )

        if skipped_files > 0:
            print(
                f"Warning: Skipped {skipped_files} files due to missing image or mask."
            )

        self.augmentation = augmentation
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
            ENCODER, ENCODER_WEIGHTS
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

            # Apply SMP preprocessing
            image = self.preprocessing_fn(image)
            image = torch.from_numpy(image).permute(2, 0, 1).float()

            # Add channel dimension to mask for SMP: (H, W) -> (1, H, W)
            # Ensure mask is float tensor for loss calculation
            if mask.ndim == 2:  # H, W
                mask = np.expand_dims(mask, axis=0)  # 1, H, W
            mask = torch.from_numpy(mask).float()
            # Make sure mask is float (Albumentations ToTensorV2 usually handles this)
            # If using custom preprocessing, ensure mask becomes torch.float32

        except Exception as e:
            # print(
            #     f"Error loading item {i}: Image: {self.image_paths[i]}, Mask: {self.mask_paths[i]}"
            # )
            # print(f"Error details: {e}")
            # Return None or raise error, or return a dummy sample
            # Returning dummy sample to avoid crashing the training loop entirely
            # Adjust dummy data size as needed
            dummy_image = torch.zeros(
                (3, IMG_SIZE[0], IMG_SIZE[1]), dtype=torch.float32
            )
            dummy_mask = torch.zeros((1, IMG_SIZE[0], IMG_SIZE[1]), dtype=torch.float32)
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
        ),
    ]
    return albu.Compose(test_transform)


# --- Model Definition ---
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=3,  # Input channels (RGB)
    classes=1,  # Output channels (binary: acne vs background)
    activation=ACTIVATION,  # Keep None for BCEWithLogitsLoss
)
model.to(DEVICE)

# --- Datasets & Dataloaders ---
train_dataset = AcneDataset(
    IMAGES_DIR,
    MASKS_DIR,
    TRAIN_FILE,
    augmentation=get_training_augmentation(IMG_SIZE),
)

valid_dataset = AcneDataset(
    IMAGES_DIR,
    MASKS_DIR,
    TEST_FILE,
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
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    collate_fn=custom_collate_fn,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
    collate_fn=custom_collate_fn,
)


# --- Loss Function ---
dice_loss = DiceLoss(mode="binary", from_logits=True).to(DEVICE)
bce_loss = SoftBCEWithLogitsLoss().to(DEVICE)

LOSS_WEIGHT_DICE = 0.6
LOSS_WEIGHT_BCE = 0.4


def combined_loss(y_pred, y_true):
    # Ensure y_true is on the correct device AND is float32
    # This explicitly casts the target mask to float32 on the correct device
    y_true_dev = y_true.to(device=y_pred.device, dtype=torch.float32)

    # Ensure y_pred is float32 (it should be logits from the model, usually float32)
    # This explicit cast adds robustness
    y_pred_float = y_pred.float()

    # Calculate individual losses with explicitly typed inputs
    loss1 = dice_loss(y_pred_float, y_true_dev)
    loss2 = bce_loss(y_pred_float, y_true_dev)

    return LOSS_WEIGHT_DICE * loss1 + LOSS_WEIGHT_BCE * loss2


# --- Optimizer ---
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

# --- Learning Rate Scheduler ---
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=LR / 100
)

# --- Training Loop ---
max_iou_score = 0.0
history = {"train_loss": [], "valid_loss": [], "iou_score": [], "f1_score": []}
start_time = time.time()

print(f"\n--- Starting Training ---")
print(f"Device: {DEVICE}")
print(f"Encoder: {ENCODER}")
print(f"Image Size: {IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LR}")
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

        images, masks_true = batch_data  # Unpack the batch
        images = images.to(DEVICE)
        masks_true = masks_true.to(DEVICE)

        optimizer.zero_grad()
        masks_pred_logits = model(images)
        loss = combined_loss(masks_pred_logits, masks_true)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar_train.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)
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

            images, masks_true = batch_data  # Unpack the batch
            images = images.to(DEVICE)
            masks_true = masks_true.to(DEVICE)  # Ground truth masks

            masks_pred_logits = model(images)
            loss = combined_loss(masks_pred_logits, masks_true)
            valid_loss += loss.item()

            # Calculate metrics: Need probabilities and then binary predictions
            # Apply sigmoid to logits -> probabilities
            masks_pred_prob = torch.sigmoid(masks_pred_logits)
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

    # Calculate overall metrics from accumulated stats
    # Summing tensors in the lists
    tp_agg = torch.cat(tp_total).sum()
    fp_agg = torch.cat(fp_total).sum()
    fn_agg = torch.cat(fn_total).sum()
    tn_agg = torch.cat(tn_total).sum()

    # Calculate metrics using aggregated stats
    avg_iou = iou_score(tp_agg, fp_agg, fn_agg, tn_agg, reduction="micro")
    avg_f1 = f1_score(tp_agg, fp_agg, fn_agg, tn_agg, reduction="micro")
    avg_acc = accuracy(tp_agg, fp_agg, fn_agg, tn_agg, reduction="micro")
    avg_prec = precision(tp_agg, fp_agg, fn_agg, tn_agg, reduction="micro")
    avg_rec = recall(tp_agg, fp_agg, fn_agg, tn_agg, reduction="micro")

    avg_valid_loss = valid_loss / len(valid_loader) if len(valid_loader) > 0 else 0

    history["valid_loss"].append(avg_valid_loss)
    history["iou_score"].append(avg_iou)
    history["f1_score"].append(avg_f1)

    # Update learning rate scheduler
    scheduler.step()  # CosineAnnealingLR steps each epoch

    # Print epoch summary
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1} Summary:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Valid Loss: {avg_valid_loss:.4f}")
    print(f"  Valid IoU:  {avg_iou:.4f}")
    print(f"  Valid F1:   {avg_f1:.4f}")
    print(f"  Valid Acc:  {avg_acc:.4f}")
    print(f"  Valid Prec: {avg_prec:.4f}")
    print(f"  Valid Rec:  {avg_rec:.4f}")
    print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")
    print(f"  Time:       {epoch_time:.2f}s")

    # Save best model based on IoU score
    if avg_iou > max_iou_score:
        max_iou_score = avg_iou
        model_save_path = (
            f"best_model_unet_{ENCODER}_epoch{epoch+1}_iou{max_iou_score:.4f}.pth"
        )
        SAVE_PATH = os.path.join(CHECKPOINT_DIR, model_save_path)
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  >> Model Saved: {SAVE_PATH} (IoU: {max_iou_score:.4f})")
    else:
        print(f"  (IoU did not improve from {max_iou_score:.4f})")

total_training_time = time.time() - start_time
print(f"\n--- Training Finished ---")
print(
    f"Total Time: {total_training_time // 3600:.0f}h {(total_training_time % 3600) // 60:.0f}m {total_training_time % 60:.2f}s"
)
print(f"Best Validation IoU: {max_iou_score:.4f}")

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
history_plot_path = f"training_history_unet_{ENCODER}.png"
plt.savefig(history_plot_path)
print(f"Training history plot saved to: {history_plot_path}")
# plt.show()
