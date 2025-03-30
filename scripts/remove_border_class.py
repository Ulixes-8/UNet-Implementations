import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

# Set paths
dataset_dir = Path('/home/ulixes/segmentation_cv/unet/nnUNet/nnUNet_raw/Dataset001_PetSegmentation')
masks_dir = dataset_dir / 'labelsTr'
dataset_json_path = dataset_dir / 'dataset.json'

# Load and update dataset.json
with open(dataset_json_path, 'r') as f:
    dataset_json = json.load(f)

# Update labels in dataset.json - remove the border class
dataset_json['labels'] = {
    "background": 0,
    "cat": 1,
    "dog": 2
}

# Save updated dataset.json
with open(dataset_json_path, 'w') as f:
    json.dump(dataset_json, f, indent=4)

print(f"Updated dataset.json: removed border class (255)")

# Process all mask files
mask_files = list(masks_dir.glob('*.png'))
print(f"Found {len(mask_files)} mask files to process")

for mask_file in tqdm(mask_files, desc="Updating masks"):
    # Load mask
    mask = np.array(Image.open(mask_file))
    
    # Create a new mask with remapped values
    new_mask = mask.copy()
    
    # Remap 255 to 0 (merge with background)
    new_mask[mask == 255] = 0
    
    # Save the updated mask
    Image.fromarray(new_mask.astype(np.uint8)).save(mask_file)

print("Mask update complete!")
print(f"Processed {len(mask_files)} masks, merging border class (255) with background (0)")