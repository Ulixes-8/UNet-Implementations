#!/usr/bin/env python
"""
Script: create_clip_resized_images.py

This script creates 224x224 resized images for CLIP from original training images:
1. Reads the original color images from the Train/color/ directory
2. Resizes them to 224x224 with aspect ratio preservation and padding
3. Saves them to Train/resized_clip/ directory

Example Usage:
    python scripts/create_clip_resized_images.py --processed-dir data/processed
"""

import argparse
import logging
import sys
from pathlib import Path
import shutil
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from tqdm import tqdm

# Add the project root to path to import our modules
project_root = Path("/home/ulixes/segmentation_cv/unet")



def create_directory(directory_path: Union[str, Path], overwrite: bool = False) -> Path:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path: Path to the directory to create
        overwrite: Whether to remove the directory if it exists
        
    Returns:
        Path to the created directory
    """
    directory = Path(directory_path)
    
    if directory.exists() and overwrite:
        shutil.rmtree(directory)
    
    directory.mkdir(parents=True, exist_ok=True)
    
    return directory


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Optional path to log file
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("resize_clip")
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
        description="Create 224x224 resized images for CLIP from original training images"
    )
    
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(project_root / "data" / "processed"),
        help="Path to the processed dataset directory"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (optional)"
    )
    
    return parser.parse_args()


def resize_image_with_padding(
    image: np.ndarray, 
    target_size: int
) -> np.ndarray:
    """
    Resize image preserving aspect ratio and pad to square.
    
    Args:
        image: Input image
        target_size: Target size for both dimensions after padding
        
    Returns:
        Resized and padded image
    """
    height, width = image.shape[:2]
    
    # Calculate target dimensions preserving aspect ratio
    if height > width:
        # Portrait orientation
        scale = target_size / height
        new_height = target_size
        new_width = int(width * scale)
    else:
        # Landscape or square orientation
        scale = target_size / width
        new_width = target_size
        new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height))
    
    # Create a square canvas with black background
    channels = image.shape[2] if len(image.shape) == 3 else 1
    if channels > 1:
        padded = np.zeros((target_size, target_size, channels), dtype=np.uint8)
    else:
        padded = np.zeros((target_size, target_size), dtype=np.uint8)
    
    # Calculate padding offsets to center the image
    pad_y = (target_size - new_height) // 2
    pad_x = (target_size - new_width) // 2
    
    # Place the resized image on the padded canvas
    if channels > 1:
        padded[pad_y:pad_y+new_height, pad_x:pad_x+new_width, :] = resized
    else:
        padded[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized
    
    return padded


def process_dataset_split_for_clip(
    split_name: str,
    processed_dir: Path,
    target_size: int,
    logger: logging.Logger
) -> None:
    """
    Process a dataset split (Train, Val, or Test) for CLIP by resizing to 224x224.
    
    Args:
        split_name: Name of the split (Train, Val, Test)
        processed_dir: Path to processed dataset directory
        target_size: Target size for CLIP (224x224)
        logger: Logger for output
    """
    # Define paths
    split_dir = processed_dir / split_name
    color_dir = split_dir / "color"
    clip_dir = split_dir / "resized_clip"
    
    # Create output directory
    create_directory(clip_dir)
    
    # Get all images from color directory
    color_images = list(color_dir.glob("*.jpg"))
    
    # For training set, also process augmented images
    augmented_images = []
    if split_name == "Train":
        augmented_dir = split_dir / "augmented" / "images"
        if augmented_dir.exists():
            augmented_images = list(augmented_dir.glob("*.jpg"))
            logger.info(f"Found {len(augmented_images)} augmented training images")
    
    # Combine image lists for training set
    all_images = color_images + augmented_images
    logger.info(f"Found {len(all_images)} total {split_name.lower()} images to resize for CLIP")
    
    # Process each image
    for img_path in tqdm(all_images, desc=f"Resizing {split_name} for CLIP"):
        try:
            # Read image with OpenCV (BGR format)
            image = cv2.imread(str(img_path))
            if image is None:
                logger.error(f"Failed to read image: {img_path}")
                continue
            
            # Resize image for CLIP
            resized_img = resize_image_with_padding(image, target_size)
            
            # Save resized image
            output_path = clip_dir / img_path.name
            cv2.imwrite(str(output_path), resized_img)
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")


def main() -> None:
    print(f"Project root directory: {project_root}")

    """Main function to create CLIP-resized images."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    # Set paths
    processed_dir = Path(args.processed_dir)
    
    # Log basic info
    logger.info(f"Processed dataset directory: {processed_dir}")
    logger.info(f"Target size for CLIP images: 224x224")
    
    # Check if processed directory exists
    if not processed_dir.exists():
        logger.error(f"Processed directory does not exist: {processed_dir}")
        sys.exit(1)
    
    # Process all dataset splits for CLIP
    for split in ["Train", "Val", "Test"]:
        split_dir = processed_dir / split
        if not split_dir.exists():
            logger.warning(f"{split} directory does not exist: {split_dir}")
            continue
            
        logger.info(f"Processing {split} split for CLIP...")
        process_dataset_split_for_clip(split, processed_dir, 224, logger)
    
    logger.info("CLIP-resized images created successfully for all dataset splits!")


if __name__ == "__main__":
    main()