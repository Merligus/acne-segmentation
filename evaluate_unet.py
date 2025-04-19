import os
import cv2
import numpy as np
import torch
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
import argparse  # To handle command-line arguments

# --- Configuration (Can be overridden by command-line arguments) ---
DEFAULT_MODEL_PATH = "./checkpoint/best_model_unet_mobilenet_v2_epoch118_iou0.4578.pth"  # Default model path
DEFAULT_INPUT_TXT = "./data/with_mask_test.txt"  # Default input text file
DEFAULT_INPUT_IMG_DIR = (
    "./data/JPEGImages/"  # Base directory containing images listed in the txt file
)
DEFAULT_OUTPUT_DIR = "./results/"  # Directory to save highlighted images
DEFAULT_IMG_SIZE = (256, 256)  # Target size model was trained on
DEFAULT_ENCODER = "mobilenet_v2"
DEFAULT_THRESHOLD = 0.5  # Threshold to binarize the mask prediction
DARKEN_FACTOR = 0.3  # Factor to darken the background (0.0 = black, 1.0 = original)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Argument Parser ---
parser = argparse.ArgumentParser(
    description="Evaluate segmentation model on images listed in a text file."
)
parser.add_argument(
    "--model",
    type=str,
    default=DEFAULT_MODEL_PATH,
    help=f"Path to the trained model checkpoint (.pth file). Default: {DEFAULT_MODEL_PATH}",
)
parser.add_argument(
    "--input_txt",
    type=str,
    default=DEFAULT_INPUT_TXT,
    help=f"Path to the text file listing input image filenames. Default: {DEFAULT_INPUT_TXT}",
)
parser.add_argument(
    "--img_dir",
    type=str,
    default=DEFAULT_INPUT_IMG_DIR,
    help=f"Base directory containing the images listed in input_txt. Default: {DEFAULT_INPUT_IMG_DIR}",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=DEFAULT_OUTPUT_DIR,
    help=f"Directory to save the highlighted output images. Default: {DEFAULT_OUTPUT_DIR}",
)
parser.add_argument(
    "--img_size",
    type=int,
    nargs=2,
    default=list(DEFAULT_IMG_SIZE),
    help=f"Target image size (height width) model was trained on. Default: {DEFAULT_IMG_SIZE}",
)
parser.add_argument(
    "--encoder",
    type=str,
    default=DEFAULT_ENCODER,
    help=f"Encoder architecture used for the model. Default: {DEFAULT_ENCODER}",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=DEFAULT_THRESHOLD,
    help=f"Probability threshold for binarizing the segmentation mask. Default: {DEFAULT_THRESHOLD}",
)
parser.add_argument(
    "--darken",
    type=float,
    default=DARKEN_FACTOR,
    help=f"Factor to darken background pixels (0.0=black, 1.0=original). Default: {DARKEN_FACTOR}",
)

args = parser.parse_args()

# Use parsed arguments
MODEL_PATH = args.model
INPUT_TXT_FILE = args.input_txt
INPUT_IMAGE_DIR = args.img_dir
OUTPUT_DIR = args.output_dir
IMG_SIZE = tuple(args.img_size)
ENCODER = args.encoder
THRESHOLD = args.threshold
DARKEN_FACTOR = args.darken


# --- Preprocessing / Augmentation Function ---
def get_preprocessing_transform(img_size):
    """Resize longest side, pad, normalize for inference."""
    max_size = max(img_size)
    test_transform = [
        albu.LongestMaxSize(
            max_size=max_size, interpolation=cv2.INTER_LINEAR, always_apply=True
        ),
        albu.PadIfNeeded(
            min_height=img_size[0],
            min_width=img_size[1],
            always_apply=True,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,  # Pad image with 0 (black)
        ),
        # Normalization is done separately using SMP's function
    ]
    return albu.Compose(test_transform)


# --- Load Model ---
print(f"Loading model: {MODEL_PATH}")
print(f"Using encoder: {ENCODER}")
try:
    # Instantiate the model architecture (must match the saved weights)
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,  # Don't need imagenet weights, loading from checkpoint
        in_channels=3,
        classes=1,  # Binary output
    )

    # Load the state dictionary
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    # Handle potential extra keys if saved directly from TrainEpoch or DataParallel
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # Remove DataParallel wrapper prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully.")

except FileNotFoundError:
    print(f"Error: Model checkpoint not found at {MODEL_PATH}")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Prepare Preprocessing ---
preprocessing_fn = smp.encoders.get_preprocessing_fn(
    ENCODER, "imagenet"
)  # Use imagenet stats for normalization consistency
transform = get_preprocessing_transform(IMG_SIZE)

# --- Create Output Directory ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# --- Read Input Filenames from Text File ---
print(f"Reading image filenames from: {INPUT_TXT_FILE}")
filenames_to_process = []
try:
    with open(INPUT_TXT_FILE, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:  # Ensure line is not empty
                filenames_to_process.append(parts[0])  # Get the first column (filename)
except FileNotFoundError:
    print(f"Error: Input text file not found at {INPUT_TXT_FILE}")
    exit()

if not filenames_to_process:
    print("Error: No filenames found in the input text file.")
    exit()

print(f"Found {len(filenames_to_process)} filenames to process.")

# --- Process Images ---
processed_count = 0
error_count = 0
with torch.no_grad():  # Disable gradient calculation for inference
    for filename in tqdm(filenames_to_process, desc="Processing Images"):
        input_image_path = os.path.join(INPUT_IMAGE_DIR, filename)

        if not os.path.exists(input_image_path):
            print(f"Warning: Image file not found, skipping: {input_image_path}")
            error_count += 1
            continue

        try:
            # --- Load and Preprocess Original Image ---
            image_orig = cv2.imread(input_image_path)
            if image_orig is None:
                print(
                    f"Warning: Could not read image file, skipping: {input_image_path}"
                )
                error_count += 1
                continue
            image_orig_rgb = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
            original_h, original_w = image_orig.shape[:2]

            # --- Apply Transforms for Model Input ---
            # 1. Apply Resize/Pad transform first
            transformed = transform(image=image_orig_rgb)
            image_padded = transformed["image"]  # Image is now resized+padded (H, W, C)

            # 2. Apply SMP normalization and convert to tensor
            image_normalized = preprocessing_fn(image_padded)
            input_tensor = (
                torch.from_numpy(image_normalized)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(DEVICE)
                .float()
            )  # Add batch dim, move to device

            # --- Perform Inference ---
            pred_logits = model(input_tensor)  # Output is (1, 1, H, W) logits
            pred_prob = torch.sigmoid(
                pred_logits
            )  # Convert logits to probabilities (0-1)

            # --- Postprocess Mask ---
            # Remove batch dimension, move to CPU, convert to numpy
            pred_mask_resized = pred_prob.squeeze().cpu().numpy()  # (H, W)

            # --- Unpad/Resize Mask to Original Image Size ---
            # Need to reverse the LongestMaxSize and PadIfNeeded steps approximately

            # 1. Figure out the dimensions *before* padding was applied
            # This requires knowing how LongestMaxSize resized it
            scale = max(IMG_SIZE) / max(original_h, original_w)
            resized_h = int(original_h * scale)
            resized_w = int(original_w * scale)

            # 2. Calculate padding amounts (assuming padding was added symmetrically, which PadIfNeeded tries to do)
            pad_h = IMG_SIZE[0] - resized_h
            pad_w = IMG_SIZE[1] - resized_w
            top_pad = pad_h // 2
            left_pad = pad_w // 2

            # 3. Crop out the padding from the predicted mask
            # Ensure indices are non-negative
            crop_h_start = max(0, top_pad)
            crop_w_start = max(0, left_pad)
            crop_h_end = min(IMG_SIZE[0], crop_h_start + resized_h)
            crop_w_end = min(IMG_SIZE[1], crop_w_start + resized_w)

            pred_mask_unpadded = pred_mask_resized[
                crop_h_start:crop_h_end, crop_w_start:crop_w_end
            ]

            # 4. Resize the unpadded mask back to the original image dimensions
            pred_mask_orig_size = cv2.resize(
                pred_mask_unpadded,
                (original_w, original_h),
                interpolation=cv2.INTER_LINEAR,
            )

            # 5. Binarize the final mask
            binary_mask = (pred_mask_orig_size > THRESHOLD).astype(
                np.uint8
            )  # (H_orig, W_orig) binary mask {0, 1}

            # --- Create Highlighted Image ---
            # Create a darkened version of the original image
            darkened_image = (image_orig * DARKEN_FACTOR).astype(np.uint8)
            # Create a 3-channel version of the binary mask
            binary_mask_3ch = (
                cv2.cvtColor(binary_mask * 255, cv2.COLOR_GRAY2BGR) // 255
            )  # Gives mask with {0, 1} in 3 channels

            # Combine: Use original pixels where mask is 1, darkened pixels where mask is 0
            highlighted_image = np.where(
                binary_mask_3ch == 1, image_orig, darkened_image
            )

            # --- Save Highlighted Image ---
            output_filename = (
                os.path.splitext(filename)[0] + "_highlighted.png"
            )  # Save as PNG
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            cv2.imwrite(output_path, highlighted_image)
            processed_count += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            error_count += 1


print(f"\n--- Inference Complete ---")
print(f"Processed: {processed_count} images")
print(f"Skipped/Errors: {error_count} images")
print(f"Highlighted outputs saved to: {OUTPUT_DIR}")
