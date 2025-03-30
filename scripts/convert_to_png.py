import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Set paths
dataset_dir = Path('/home/ulixes/segmentation_cv/unet/nnUNet/nnUNet_raw/Dataset001_PetSegmentation')
images_dir = dataset_dir / 'imagesTr'
dataset_json_path = dataset_dir / 'dataset.json'

# Load and update dataset.json
with open(dataset_json_path, 'r') as f:
    dataset_json = json.load(f)

# Update file_ending in dataset.json
dataset_json['file_ending'] = '.png'

# Save updated dataset.json
with open(dataset_json_path, 'w') as f:
    json.dump(dataset_json, f, indent=4)

print(f"Updated dataset.json: file_ending changed to .png")

# Get all JPG files
jpg_files = list(images_dir.glob('*.jpg'))
print(f"Found {len(jpg_files)} JPG files to convert")

# Convert JPG to PNG
for jpg_file in tqdm(jpg_files, desc="Converting images"):
    # Load image
    img = Image.open(jpg_file)
    
    # Create PNG filename (same name but with .png extension)
    png_file = jpg_file.with_suffix('.png')
    
    # Save as PNG
    img.save(png_file)
    
    # Remove the original JPG file
    jpg_file.unlink()

print("Conversion complete!")
print(f"Converted {len(jpg_files)} images from JPG to PNG")