#!/usr/bin/env python
"""
Script to convert the Oxford-IIIT Pet Dataset to nnU-Net format.
This script only copies files; it does not modify the original data.
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Pet Dataset to nnU-Net format")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to processed data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to nnUNet_raw output directory")
    parser.add_argument("--use_augmented", action="store_true", help="Use augmented data if available")
    return parser.parse_args()

def create_dataset_json(output_path, num_training_cases):
    """Create dataset.json file required by nnU-Net."""
    dataset_json = {
        "channel_names": {
            "0": "RGB"  # Single RGB channel
        },
        "labels": {
            "background": 0,
            "cat": 1,
            "dog": 2
        },
        "numTraining": num_training_cases,
        "file_ending": ".jpg"
    }
    
    with open(output_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)

def main():
    args = parse_args()
    
    # Create dataset directory in nnUNet_raw
    dataset_name = "Dataset001_PetSegmentation"
    output_dataset_dir = Path(args.output_dir) / dataset_name
    output_imagesTr_dir = output_dataset_dir / "imagesTr"
    output_labelsTr_dir = output_dataset_dir / "labelsTr"
    
    # Create directories
    output_imagesTr_dir.mkdir(parents=True, exist_ok=True)
    output_labelsTr_dir.mkdir(parents=True, exist_ok=True)
    
    # Source directories
    data_dir = Path(args.data_dir)
    
    # Choose between augmented and regular data
    if args.use_augmented and (data_dir / "Train" / "augmented" / "images").exists():
        print("Using augmented data...")
        train_imgs_dir = data_dir / "Train" / "augmented" / "images"
        train_masks_dir = data_dir / "Train" / "augmented" / "masks"
    else:
        print("Using regular data...")
        train_imgs_dir = data_dir / "Train" / "resized"
        train_masks_dir = data_dir / "Train" / "resized_label"
    
    # Check if directories exist
    if not train_imgs_dir.exists() or not train_masks_dir.exists():
        print(f"Error: Source directories not found. Please check the path: {args.data_dir}")
        return
    
    # Get all image files
    image_files = sorted(list(train_imgs_dir.glob("*.jpg")))
    if len(image_files) == 0:
        print(f"Error: No image files found in {train_imgs_dir}")
        return
    
    print(f"Found {len(image_files)} training images")
    
    # Process each image
    for idx, img_path in enumerate(tqdm(image_files, desc="Converting images")):
        # Get corresponding mask path
        mask_path = train_masks_dir / f"{img_path.stem}.png"
        
        if not mask_path.exists():
            print(f"Warning: No mask found for {img_path.name}, skipping")
            continue
        
        # Define nnU-Net file naming scheme
        # Format: case_identifier_XXXX.jpg - we'll use pet_### as case identifiers
        case_id = f"pet_{idx:03d}"
        
        # For RGB images, nnU-Net allows a single file (not split by channel)
        output_img_path = output_imagesTr_dir / f"{case_id}_0000.jpg"
        output_mask_path = output_labelsTr_dir / f"{case_id}.png"
        
        # Copy files to nnU-Net directory
        shutil.copy(img_path, output_img_path)
        shutil.copy(mask_path, output_mask_path)
    
    # Create dataset.json file
    create_dataset_json(output_dataset_dir / "dataset.json", len(image_files))
    
    print(f"Conversion complete. Dataset saved to {output_dataset_dir}")
    print(f"Next step: Run nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity")

if __name__ == "__main__":
    main()