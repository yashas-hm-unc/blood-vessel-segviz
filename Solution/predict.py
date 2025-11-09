import os

import numpy as np
import torch
from skimage import io 
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TestVesselDataset
from unet_model import UNet

# --- Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_WORKERS = 2
PIN_MEMORY = True
MODEL_PATH = "unet_checkpoint.pth.tar"

# --- Data Paths ---
# Corrected BASE_PATH to be relative to the script's location
BASE_PATH = os.path.join(os.path.dirname(__file__), 'Data/preprocessed')
TEST_IMG_DIR = os.path.join(BASE_PATH, 'test/image')
OUTPUT_PRED_DIR = os.path.join(BASE_PATH, 'test/predictions')  # New directory for predictions


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Loads model checkpoint.
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])


def predict_fn(loader, model, output_dir):
    """
    Makes predictions on the test set and saves the segmented masks.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for sample in tqdm(loader, desc="Predicting"):
            images = sample['image'].to(device=DEVICE)
            filenames = sample['filename']  # Assuming dataset returns filenames

            predictions = torch.sigmoid(model(images))
            predictions = (predictions > 0.5).float()  # Threshold to get binary mask

            for i, pred_mask in enumerate(predictions):
                filename = filenames[i]
                # Convert tensor to numpy array, remove channel dimension, and scale to 0-255
                pred_mask_np = pred_mask.squeeze().cpu().numpy() * 255

                # Save the predicted mask
                output_path = os.path.join(output_dir, filename)
                io.imsave(str(output_path), pred_mask_np.astype(np.uint8), check_contrast=False)
    model.train()  # Set model back to training mode


def main():
    print(f"Using device: {DEVICE}")

    # Initialize the model
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)

    # Load the trained model checkpoint
    load_checkpoint(MODEL_PATH, model)

    test_ds = TestVesselDataset(image_dir=TEST_IMG_DIR)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    # Make predictions
    predict_fn(test_loader, model, OUTPUT_PRED_DIR)
    print(f"Predictions saved to: {os.path.abspath(OUTPUT_PRED_DIR)}")


if __name__ == "__main__":
    if not os.path.isdir(TEST_IMG_DIR):
        print("\nError: Preprocessed test data directories not found!")
        print("Please ensure you have run the data_preprocessing.py script")
        print(f"and that the data is located in '{os.path.abspath(BASE_PATH)}'")
    else:
        main()
