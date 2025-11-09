import os
import numpy as np
from skimage import io
import torch
from torch.utils.data import Dataset

class VesselDataset(Dataset):
    """
    A custom PyTorch Dataset for loading vessel image patches and their masks.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (string): Directory with all the image patches.
            mask_dir (string): Directory with all the mask patches.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Get a sorted list of image filenames
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Construct file paths
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        mask_path = os.path.join(self.mask_dir, image_filename) # Mask has the same filename

        # Load image and mask
        image = io.imread(image_path)
        mask = io.imread(mask_path)

        # If the image is loaded with a channel dimension, convert to grayscale
        if image.ndim == 3:
            image = image[:, :, 0] # Convert from (H, W, C) to (H, W)
        if mask.ndim == 3:
            mask = mask[:, :, 0] # Also handle mask if it has channels


        # --- Preprocessing ---
        # 1. Convert to float tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        # 2. Normalize image to [0, 1]
        image = image / 255.0
        
        # 3. Normalize mask to have values 0 and 1
        mask = (mask > 0).long()

        # 4. Add a channel dimension (C, H, W) as expected by PyTorch models
        image = image.unsqueeze(0) # From (H, W) to (1, H, W)

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == '__main__':
    # --- A quick test to see if the dataset works correctly ---
    
    # Use the corrected path for the preprocessed data
    base_path = 'Data/preprocessed' # Relative to the Solution directory
    train_image_dir = os.path.join(base_path, 'train/image')
    train_mask_dir = os.path.join(base_path, 'train/mask')

    print(f"Looking for training images in: {os.path.abspath(train_image_dir)}")
    print(f"Looking for training masks in: {os.path.abspath(train_mask_dir)}")

    # Check if the directories exist
    if not os.path.isdir(train_image_dir) or not os.path.isdir(train_mask_dir):
        print("\nError: Preprocessed data directories not found!")
        print("Please ensure you have run the data_preprocessing.py script")
        print("and that the data is located in Solution/Data/preprocessed/")
    else:
        # Create an instance of the dataset
        vessel_dataset = VesselDataset(image_dir=train_image_dir, mask_dir=train_mask_dir)

        print(f"\nDataset created successfully!")
        print(f"Number of training samples: {len(vessel_dataset)}")

        # Get a sample from the dataset
        if len(vessel_dataset) > 0:
            sample = vessel_dataset[0]
            image, mask = sample['image'], sample['mask']

            print(f"Sample image shape: {image.shape}")   # Should be [1, 512, 512]
            print(f"Sample image dtype: {image.dtype}") # Should be torch.float32
            print(f"Sample mask shape: {mask.shape}")     # Should be [512, 512]
            print(f"Sample mask dtype: {mask.dtype}")   # Should be torch.int64
            print(f"Unique values in sample mask: {torch.unique(mask)}") # Should be [0, 1]
        else:
            print("Dataset is empty. No patches were found.")
