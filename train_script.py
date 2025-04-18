import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, SoftBCEWithLogitsLoss
import albumentations as albu
from albumentations.pytorch import ToTensorV2  # Use ToTensorV2 for PyTorch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR = "./data/"
IMAGES_DIR = os.path.join(DATA_DIR, "JPEGImages")
MASKS_DIR = os.path.join(DATA_DIR, "masks")
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
    """Reads images and masks based on filenames listed in a text file."""

    def __init__(
        self,
        images_dir,
        masks_dir,
        file_list_path,  # Path to with_mask_train.txt or with_mask_test.txt
        augmentation=None,
        preprocessing=None,
    ):
        self.image_paths = []
        self.mask_paths = []
        # mask extension
        ext = ".jpg"
        with open(file_list_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:  # Skip empty lines
                    continue
                filename = parts[0]  # We only need the filename for segmentation

                # Construct full paths
                img_path = os.path.join(images_dir, filename)

                # --- Determine mask filename ---
                # Try matching base name with common mask extensions (.png, .jpg, etc.)
                base_name = os.path.splitext(filename)[0]
                mask_filename = base_name + ext
                mask_path_try = os.path.join(masks_dir, mask_filename)
                if os.path.exists(mask_path_try):
                    mask_path = mask_path_try

                # Only add if both image and mask exist
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
                else:
                    print(
                        f"Warning: Skipping {filename}. Image or mask not found."
                    )  # Optional warning

        if not self.image_paths:
            raise FileNotFoundError(
                f"No valid image/mask pairs found. Check paths and filenames in {file_list_path}"
            )

        self.augmentation = augmentation
        self.preprocessing = preprocessing

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

            # Convert mask to binary format (0 background, 1 foreground)
            # Assuming acne pixels are non-zero (e.g., 255)
            # Adjust the threshold if your masks use different values
            _, mask = cv2.threshold(
                mask, 127, 1, cv2.THRESH_BINARY
            )  # Use 127 as threshold for common 0/255 masks

            # Apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]

            # Apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]

            # Add channel dimension to mask for SMP: (H, W) -> (1, H, W)
            # Ensure mask is float tensor for loss calculation
            if mask.ndim == 2:  # H, W
                mask = np.expand_dims(mask, axis=0)  # 1, H, W
            # Make sure mask is float (Albumentations ToTensorV2 usually handles this)
            # If using custom preprocessing, ensure mask becomes torch.float32

        except Exception as e:
            print(
                f"Error loading item {i}: Image: {self.image_paths[i]}, Mask: {self.mask_paths[i]}"
            )
            print(f"Error details: {e}")
            # Return None or raise error, or return a dummy sample
            # Returning dummy sample to avoid crashing the training loop entirely
            # Adjust dummy data size as needed
            dummy_image = torch.zeros(
                (3, IMG_SIZE[0], IMG_SIZE[1]), dtype=torch.float32
            )
            dummy_mask = torch.zeros((1, IMG_SIZE[0], IMG_SIZE[1]), dtype=torch.float32)
            return dummy_image, dummy_mask

        return image, mask


# --- Augmentations ---
def get_training_augmentation(img_size):
    train_transform = [
        albu.Resize(height=img_size[0], width=img_size[1], always_apply=True),
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=0.1, rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=0
        ),
        albu.GaussNoise(p=0.1),
        albu.Perspective(p=0.2),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.5,
        ),
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.5,
        ),
        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.5,
        ),
        albu.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),  # ImageNet Normalization
        ToTensorV2(),  # Converts image and mask to PyTorch tensors
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(img_size):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(height=img_size[0], width=img_size[1], always_apply=True),
        # PadIfNeeded necessary if using architectures like UNet that require specific input sizes
        # albu.PadIfNeeded(min_height=img_size[0], min_width=img_size[1], always_apply=True, border_mode=0),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return albu.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),  # Apply SMP preprocessing
        ToTensorV2(),  # Converts image and mask to PyTorch tensors
    ]
    return albu.Compose(_transform)


# --- Model Definition ---
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=3,  # Input channels (RGB)
    classes=1,  # Output channels (binary: acne vs background)
    activation=ACTIVATION,  # Keep None for BCEWithLogitsLoss
)

# --- Preprocessing Function (from SMP) ---
# Note: Albumentations Normalize transform is used above, which is standard ImageNet normalization.
# SMP's preprocessing often does the same. If you want to use SMP's specific function:
# preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
# Instead of Albumentations Normalize, you would use this preprocessing_fn.
# For simplicity here, we stick with Albumentations Normalize + ToTensorV2.

# --- Datasets & Dataloaders ---
train_dataset = AcneDataset(
    IMAGES_DIR,
    MASKS_DIR,
    TRAIN_FILE,
    augmentation=get_training_augmentation(IMG_SIZE),
    # preprocessing=get_preprocessing(preprocessing_fn), # Uncomment if using SMP preprocess
)

valid_dataset = AcneDataset(
    IMAGES_DIR,
    MASKS_DIR,
    TEST_FILE,
    augmentation=get_validation_augmentation(IMG_SIZE),
    # preprocessing=get_preprocessing(preprocessing_fn), # Uncomment if using SMP preprocess
)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(valid_dataset)}")

if len(train_dataset) == 0 or len(valid_dataset) == 0:
    print("Error: One or both datasets are empty. Please check file paths and content.")
    exit()


train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)


# --- Loss Function ---
# Combine Dice Loss and BCE Loss for better stability and gradient flow
dice_loss = DiceLoss(mode="binary", from_logits=True)
bce_loss = SoftBCEWithLogitsLoss()  # SMP's version, handles logits


def combined_loss(y_pred, y_true):
    return 0.5 * dice_loss(y_pred, y_true) + 0.5 * bce_loss(y_pred, y_true)


# --- Metrics ---
# Use SMP's metrics
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.Fscore(threshold=0.5),  # Dice score is F1-score
    smp.utils.metrics.Accuracy(threshold=0.5),
    smp.utils.metrics.Precision(threshold=0.5),
    smp.utils.metrics.Recall(threshold=0.5),
]

# --- Optimizer ---
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# --- Learning Rate Scheduler ---
# Example: Reduce LR on plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

# --- Training & Validation Epochs (Using SMP's helper classes) ---
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=combined_loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=combined_loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

# --- Training Loop ---
max_score = 0
history = {"train_loss": [], "valid_loss": [], "iou_score": [], "f1_score": []}

print(f"\nStarting validation on {DEVICE}...")

valid_logs = valid_epoch.run(valid_loader)
current_score = valid_logs["iou_score"]
max_score = current_score
print(f"Actual IoU: {max_score:.4f}")

print(f"\nStarting training on {DEVICE}...")

for i in range(0, EPOCHS):
    print(f"\nEpoch: {i+1}/{EPOCHS}")

    # Perform training & validation
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    # Store logs
    history["train_loss"].append(
        train_logs["combined_loss"]
    )  # Use the name of your combined loss function if different
    history["valid_loss"].append(valid_logs["combined_loss"])
    history["iou_score"].append(valid_logs["iou_score"])
    history["f1_score"].append(valid_logs["fscore"])  # smp uses 'fscore' for F1

    # Update learning rate
    scheduler.step(valid_logs["combined_loss"])  # Reduce LR based on validation loss

    # Save best model based on IoU score
    current_score = valid_logs["iou_score"]
    if max_score < current_score:
        max_score = current_score
        torch.save(model.state_dict(), f"best_model_unet_{ENCODER}.pth")
        print(f"Model saved! New best IoU: {max_score:.4f}")

print(f"\nTraining finished. Best IoU: {max_score:.4f}")

# --- Plotting History (Optional) ---
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
plt.title("Metrics History")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"training_history_unet_{ENCODER}.png")
print("Training history plot saved.")
# plt.show() # Uncomment to display plot directly
