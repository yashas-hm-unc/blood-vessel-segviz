import os
import numpy as np
from skimage import io, color, util
from tqdm import tqdm

def create_patches(image, mask, patch_size, stride, output_path_image, output_path_mask, name):
    """
    Creates and saves overlapping patches from an image and its mask.
    Filters out patches that are all/mostly black, or have no/few vessel pixels.
    """
    # Create output directories if they don't exist
    os.makedirs(output_path_image, exist_ok=True)
    os.makedirs(output_path_mask, exist_ok=True)

    if image.ndim == 3:
        image = color.rgb2gray(image)

    image = util.img_as_float(image)
    mask = mask.astype(float)

    patch_count = 0
    total_patches = 0
    for y in range(0, image.shape[0] - patch_size + 1, stride):
        for x in range(0, image.shape[1] - patch_size + 1, stride):
            total_patches += 1
            # Extract patch from image and mask
            patch_image = image[y:y + patch_size, x:x + patch_size]
            patch_mask = mask[y:y + patch_size, x:x + patch_size]
           
            black_pixel_ratio = np.sum(patch_mask) / (patch_size * patch_size)
            is_mostly_black = black_pixel_ratio < 0.05
           

            if not is_mostly_black:
                # Save patch
                patch_filename = f"{os.path.splitext(name)[0]}_patch_{patch_count}.tif"
                # Convert boolean mask back to uint8 for saving
                io.imsave(os.path.join(output_path_image, patch_filename), patch_image, check_contrast=False)
                io.imsave(os.path.join(output_path_mask, patch_filename), (patch_mask * 255).astype(np.uint8), check_contrast=False)
                patch_count += 1
                
    
    print(f"Created {patch_count} valid patches out of {total_patches}")
    

def create_patches_test(image, patch_size, stride, output_path_image, name):
    """
    Creates and saves overlapping patches from an image.
    """
    os.makedirs(output_path_image, exist_ok=True)

    if image.ndim == 3:
        image = color.rgb2gray(image)

    image = util.img_as_float(image)

    patch_count = 0
    total_patches = 0
    for y in range(0, image.shape[0] - patch_size + 1, stride):
        for x in range(0, image.shape[1] - patch_size + 1, stride):
            total_patches += 1
            patch_image = image[y:y + patch_size, x:x + patch_size]
            
            patch_filename = f"{os.path.splitext(name)[0]}_patch_{patch_count}.tif"
            io.imsave(os.path.join(output_path_image, patch_filename), patch_image, check_contrast=False)
            patch_count += 1
                
    print(f"Created {patch_count} patches for test image {name} out of {total_patches}")


def main():

    PATCH_SIZE = 512  # Size of the patches (512x512 pixels)
    STRIDE = 256      # Overlap between patches (512-256 = 256 pixels overlap)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    base_data_path = os.path.join(project_root, 'Problem/Data')
    output_base_path = os.path.join(project_root, 'Solution/Data/preprocessed')

    # Paths for training data
    image_path = os.path.join(base_data_path, 'train_data/image/N 134 copy.tif')
    mask_path = os.path.join(base_data_path, 'train_data/ground truth/N 134 groundtruth.tif')
    output_image_path = os.path.join(output_base_path, 'train/image')
    output_mask_path = os.path.join(output_base_path, 'train/mask')

    # Process Training Data
    print("Processing training data...")
    image = io.imread(image_path)
    mask = io.imread(mask_path)
    mask = mask>0
    # here image is a little different than validation
    mask = np.logical_not(mask)

    create_patches(image, mask, PATCH_SIZE, STRIDE, output_image_path, output_mask_path, 'train')
        

    # Paths for validation data
    image_path = os.path.join(base_data_path, 'validation_data/N 129 crop.tif')
    mask_path = os.path.join(base_data_path, 'validation_data/N 129 groundtruth.tif')
    output_image_path = os.path.join(output_base_path, 'validation/image')
    output_mask_path = os.path.join(output_base_path, 'validation/mask')

    # Process Validation Data
    print("\nProcessing validation data...")
    image = io.imread(image_path)
    mask = io.imread(mask_path)
    mask = mask>0

    create_patches(image, mask, PATCH_SIZE, STRIDE, output_image_path, output_mask_path, 'validation')

    # Paths for test data
    test_image_dir = os.path.join(base_data_path, 'test_data')
    output_test_image_path = os.path.join(output_base_path, 'test/image')

    # Process Test Data
    print("\nProcessing test data...")
    test_image_filenames = sorted([f for f in os.listdir(test_image_dir) if f.endswith('.tif')])
    for filename in tqdm(test_image_filenames, desc="Creating test patches"):
        image_path = os.path.join(test_image_dir, filename)
        image = io.imread(image_path)
        create_patches_test(image, PATCH_SIZE, STRIDE, output_test_image_path, filename)

if __name__ == "__main__":
    main()
