#!/usr/bin/env python
"""
Script: convert_to_nnunet.py

This script converts the Oxford-IIIT Pet Dataset to nnU-Net format, combining
both original and augmented training data. It creates a properly structured
dataset that nnU-Net can use for training and architecture search.

The script handles:
1. Combining original (resized) and augmented training images
2. Creating properly named image/label pairs according to nnU-Net conventions
3. Generating the dataset.json file with appropriate metadata

Example Usage:
    python convert_to_nnunet.py --data_dir data/processed --output_dir $nnUNet_raw/Dataset001_PetSegmentation
"""

import os
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Optional path to log file
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("convert_to_nnunet")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if requested)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Pet Dataset to nnU-Net format, combining original and augmented data"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to processed data directory"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to nnUNet_raw output directory for the dataset"
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Dataset001_PetSegmentation",
        help="Dataset name (should follow the pattern DatasetXXX_Name)"
    )
    
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to log file (optional)"
    )
    
    parser.add_argument(
        "--include_augmented",
        action="store_true",
        help="Include augmented data in the training set"
    )
    
    return parser.parse_args()


def verify_dataset_structure(data_dir: Path, logger: logging.Logger) -> bool:
    """
    Verify that the expected dataset structure exists.
    
    Args:
        data_dir: Path to the processed data directory
        logger: Logger for output
        
    Returns:
        True if dataset structure is valid, False otherwise
    """
    required_dirs = [
        "Train/resized",
        "Train/resized_label",
        "Train/augmented/images",
        "Train/augmented/masks",
        "Val/resized",
        "Val/processed_labels",
        "Test/resized",
        "Test/processed_labels"
    ]
    
    valid = True
    for dir_path in required_dirs:
        full_path = data_dir / dir_path
        if not full_path.exists():
            logger.error(f"Required directory not found: {full_path}")
            valid = False
    
    return valid


def create_nnunet_directory_structure(output_dir: Path, logger: logging.Logger) -> Tuple[Path, Path]:
    """
    Create the nnU-Net directory structure.
    
    Args:
        output_dir: Path to the output directory for the dataset
        logger: Logger for output
        
    Returns:
        Tuple of (imagesTr_dir, labelsTr_dir)
    """
    # Create main directories
    imagesTr_dir = output_dir / "imagesTr"
    labelsTr_dir = output_dir / "labelsTr"
    
    imagesTr_dir.mkdir(parents=True, exist_ok=True)
    labelsTr_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created nnU-Net directory structure in {output_dir}")
    
    return imagesTr_dir, labelsTr_dir


def get_class_from_mask(mask_path: Path) -> int:
    """
    Determine if a mask contains a cat (1) or dog (2).
    
    Args:
        mask_path: Path to the mask file
        
    Returns:
        1 for cat, 2 for dog, 0 if neither could be determined
    """
    try:
        # Read mask
        mask = np.array(Image.open(mask_path))
        
        # Check for presence of class 1 (cat) or class 2 (dog)
        unique_values = np.unique(mask)
        if 1 in unique_values:
            return 1  # Cat
        elif 2 in unique_values:
            return 2  # Dog
        else:
            # As a fallback, check the filename
            return get_class_from_filename(mask_path.stem)
    except Exception:
        # If mask reading fails, check the filename
        return get_class_from_filename(mask_path.stem)


def get_class_from_filename(filename: str) -> int:
    """
    Determine if a file represents a cat or dog based on filename.
    
    Args:
        filename: Filename without extension
        
    Returns:
        1 for cat, 2 for dog, 0 if undetermined
    """
    # List of cat breeds
    cat_breeds = [
        'abyssinian', 'bengal', 'birman', 'bombay', 
        'british', 'egyptian', 'maine', 
        'persian', 'ragdoll', 'russian', 'siamese', 'sphynx'
    ]
    
    # List of dog breeds
    dog_breeds = [
        'american_bulldog', 'american_pit', 'pit_bull', 
        'basset', 'beagle', 'boxer', 'chihuahua', 
        'cocker_spaniel', 'english_setter', 
        'german_shorthaired', 'great_pyrenees', 'havanese', 
        'japanese_chin', 'keeshond', 'leonberger', 
        'miniature_pinscher', 'newfoundland', 'pomeranian', 
        'pug', 'saint_bernard', 'samoyed', 'scottish', 
        'shiba', 'staffordshire', 
        'wheaten', 'yorkshire'
    ]
    
    filename_lower = filename.lower()
    
    # Check if filename contains a known cat breed
    if any(breed in filename_lower for breed in cat_breeds):
        return 1  # Cat
    # Check if filename contains a known dog breed
    elif any(breed in filename_lower for breed in dog_breeds):
        return 2  # Dog
    
    return 0  # Unknown


def copy_and_rename_files(
    image_files: List[Path],
    mask_files: Dict[str, Path],
    imagesTr_dir: Path,
    labelsTr_dir: Path,
    logger: logging.Logger
) -> List[Dict[str, Union[str, int]]]:
    """
    Copy and rename files according to nnU-Net naming convention.
    
    Args:
        image_files: List of image file paths
        mask_files: Dictionary mapping image stem to mask path
        imagesTr_dir: Output directory for images
        labelsTr_dir: Output directory for labels
        logger: Logger for output
        
    Returns:
        List of dictionaries with case information (name, class)
    """
    case_info = []
    
    for idx, img_path in enumerate(tqdm(image_files, desc="Processing files")):
        # Get corresponding mask path
        mask_path = mask_files.get(img_path.stem)
        if not mask_path:
            logger.warning(f"No mask found for {img_path.name}, skipping")
            continue
        
        # Determine class
        class_id = get_class_from_mask(mask_path)
        if class_id == 0:
            logger.warning(f"Could not determine class for {img_path.name}, skipping")
            continue
        
        # Generate nnU-Net case identifier
        case_id = f"pet_{idx:04d}"
        
        # Define output paths
        output_img_path = imagesTr_dir / f"{case_id}_0000.jpg"
        output_mask_path = labelsTr_dir / f"{case_id}.png"
        
        # Copy files
        shutil.copy(img_path, output_img_path)
        shutil.copy(mask_path, output_mask_path)
        
        # Store case information
        case_info.append({
            "case_id": case_id,
            "original_image": str(img_path),
            "original_mask": str(mask_path),
            "class": class_id  # 1 for cat, 2 for dog
        })
    
    return case_info


def create_dataset_json(
    output_dir: Path,
    case_info: List[Dict[str, Union[str, int]]],
    logger: logging.Logger
) -> None:
    """
    Create dataset.json file required by nnU-Net.
    
    Args:
        output_dir: Output directory for the dataset
        case_info: List of dictionaries with case information
        logger: Logger for output
    """
    # Count class distribution
    cats = sum(1 for case in case_info if case["class"] == 1)
    dogs = sum(1 for case in case_info if case["class"] == 2)
    
    # Create dataset.json
    dataset_json = {
        "name": "PetSegmentation",
        "description": "Oxford-IIIT Pet Dataset for semantic segmentation",
        "reference": "Parkhi et al., Cats and Dogs, IEEE Conference on Computer Vision and Pattern Recognition, 2012",
        "licence": "Creative Commons Attribution 4.0 International License",
        "release": "1.0",
        "channel_names": {
            "0": "R",
            "1": "G",
            "2": "B"
        },
        "labels": {
            "background": 0,
            "cat": 1,
            "dog": 2,
            "border": 255
        },
        "numTraining": len(case_info),
        "numTest": 0,  # nnU-Net doesn't need this for training/architecture search
        "file_ending": ".jpg"
    }
    
    # Save dataset.json
    with open(output_dir / "dataset.json", 'w') as f:
        json.dump(dataset_json, f, indent=4)
    
    # Also save a summary file with more detailed information
    summary = {
        "total_cases": len(case_info),
        "cats": cats,
        "dogs": dogs,
        "class_distribution": f"{cats/len(case_info)*100:.1f}% cats, {dogs/len(case_info)*100:.1f}% dogs",
        "cases": case_info
    }
    
    with open(output_dir / "dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"Created dataset.json and dataset_summary.json in {output_dir}")
    logger.info(f"Dataset summary: {len(case_info)} total cases ({cats} cats, {dogs} dogs)")


def create_split_file(
    output_dir: Path,
    case_info: List[Dict[str, Union[str, int]]],
    logger: logging.Logger
) -> None:
    """
    Create a splits_final.json file for custom cross-validation splits.
    
    Args:
        output_dir: Output directory for the dataset
        case_info: List of dictionaries with case information
        logger: Logger for output
    """
    # Get all case IDs
    all_cases = [case["case_id"] for case in case_info]
    
    # Splitting strategy: ensure balanced distribution of cats and dogs across folds
    cat_cases = [case["case_id"] for case in case_info if case["class"] == 1]
    dog_cases = [case["case_id"] for case in case_info if case["class"] == 2]
    
    # Shuffle the cases
    np.random.seed(42)
    np.random.shuffle(cat_cases)
    np.random.shuffle(dog_cases)
    
    # Create 5 folds with balanced cat/dog distribution
    splits = []
    for fold in range(5):
        # Calculate indices for this fold
        cat_start = (fold * len(cat_cases)) // 5
        cat_end = ((fold + 1) * len(cat_cases)) // 5
        
        dog_start = (fold * len(dog_cases)) // 5
        dog_end = ((fold + 1) * len(dog_cases)) // 5
        
        # Extract validation set for this fold
        val_cats = cat_cases[cat_start:cat_end]
        val_dogs = dog_cases[dog_start:dog_end]
        val_cases = val_cats + val_dogs
        
        # Training set is everything except validation
        train_cases = [case for case in all_cases if case not in val_cases]
        
        splits.append({
            "train": train_cases,
            "val": val_cases
        })
    
    # Create splits_final.json
    with open(output_dir / "splits_final.json", 'w') as f:
        json.dump(splits, f, indent=4)
    
    logger.info(f"Created splits_final.json in {output_dir} with 5 balanced folds")


def main() -> None:
    """Main function to convert dataset to nnU-Net format."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    # Set paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Verify dataset structure
    if not verify_dataset_structure(data_dir, logger):
        logger.error("Invalid dataset structure. Please check the data directory.")
        return
    
    # Create nnU-Net directory structure
    imagesTr_dir, labelsTr_dir = create_nnunet_directory_structure(output_dir, logger)
    
    # Get paths for original training images and masks
    train_original_imgs = list((data_dir / "Train" / "resized").glob("*.jpg"))
    train_original_masks = {mask_path.stem: mask_path for mask_path in (data_dir / "Train" / "resized_label").glob("*.png")}
    
    # Get paths for augmented training images and masks
    train_augmented_imgs = []
    train_augmented_masks = {}
    
    if args.include_augmented:
        train_augmented_imgs = list((data_dir / "Train" / "augmented" / "images").glob("*.jpg"))
        train_augmented_masks = {mask_path.stem: mask_path for mask_path in (data_dir / "Train" / "augmented" / "masks").glob("*.png")}
    
    # Combine paths
    all_image_paths = train_original_imgs + train_augmented_imgs
    all_mask_paths = {**train_original_masks, **train_augmented_masks}
    
    logger.info(f"Found {len(train_original_imgs)} original training images")
    logger.info(f"Found {len(train_augmented_imgs)} augmented training images")
    logger.info(f"Processing {len(all_image_paths)} total images")
    
    # Copy and rename files
    case_info = copy_and_rename_files(
        all_image_paths, all_mask_paths,
        imagesTr_dir, labelsTr_dir,
        logger
    )
    
    # Create dataset.json
    create_dataset_json(output_dir, case_info, logger)
    
    # Create splits_final.json
    create_split_file(output_dir, case_info, logger)
    
    logger.info(f"Dataset conversion complete. Data saved to {output_dir}")
    logger.info(f"Next step: Run 'nnUNetv2_plan_and_preprocess -d {args.dataset_name} --verify_dataset_integrity'")


if __name__ == "__main__":
    main()