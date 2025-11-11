import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VesselDataset
# Import our custom modules
from unet_model import UNet

# --- Hyperparameters ---
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 15
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
MODEL_PATH = "unet_checkpoint.pth.tar"

# --- Data Paths ---
BASE_PATH = 'Data/preprocessed'
TRAIN_IMG_DIR = os.path.join(BASE_PATH, 'train/image')
TRAIN_MASK_DIR = os.path.join(BASE_PATH, 'train/mask')
VAL_IMG_DIR = os.path.join(BASE_PATH, 'validation/image')
VAL_MASK_DIR = os.path.join(BASE_PATH, 'validation/mask')


def train_fn(loader, model, optimizer, loss_fn, scaler):
    """
    Trains the model for one epoch.
    """
    loop = tqdm(loader, desc="Training")

    for batch_idx, sample in enumerate(loop):
        data = sample['image'].to(device=DEVICE)
        targets = sample['mask'].to(device=DEVICE).float().unsqueeze(1)

        # Forward pass
        if DEVICE == "cuda":
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)
        else:
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        if DEVICE == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())


def check_accuracy(loader, model, device="cpu"):
    """
    Checks accuracy on the validation set.
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for sample in tqdm(loader, desc="Validation"):
            x = sample['image'].to(device)
            y = sample['mask'].to(device).float().unsqueeze(1)
            y = 1 - y  # Invert the validation targets to match training logic

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
            )

    accuracy = num_correct / num_pixels * 100
    dice = dice_score / len(loader)
    print(f"Validation Accuracy: {accuracy:.2f}%")
    print(f"Dice Score: {dice:.4f}")

    model.train()
    return accuracy, dice


def main():
    print(f"Using device: {DEVICE}")

    # Initialize the model
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)

    # Loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Create DataLoaders ---
    train_ds = VesselDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )

    val_ds = VesselDataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, validation=True)
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    # For mixed precision training
    scaler = None
    if DEVICE == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    best_dice_score = -1.0

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Check accuracy
        current_accuracy, dice_score = check_accuracy(val_loader, model, device=DEVICE)

        # Save model if it's the best one so far
        if dice_score > best_dice_score:
            best_dice_score = dice_score
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, MODEL_PATH)
            print(f"-> Model saved with Dice score: {dice_score:.4f}")


if __name__ == "__main__":
    # Add a check for the data directories
    if not os.path.isdir(TRAIN_IMG_DIR) or not os.path.isdir(VAL_IMG_DIR):
        print("\nError: Preprocessed data directories not found!")
        print("Please ensure you have run the data_preprocessing.py script")
        print(f"and that the data is located in '{os.path.abspath(BASE_PATH)}'")
    else:
        main()
