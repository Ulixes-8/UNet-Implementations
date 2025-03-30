#!/usr/bin/env python
"""
Script: preprocess_training_masks.py

This script resizes the training mask files to match the dimensions of the resized training images
while preserving the exact class values:
- 0: background
- 1: cat
- 2: dog
- 255: border/don't care

Example Usage:
    python scripts/preprocess_training_masks.py --size 512
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add the project root to path to import our modules
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utils.helpers import create_directory


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Optional path to log file
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("preprocess_masks")
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
        description="Resize mask files to match the dimensions of resized training images"
    )
    
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(project_root / "data" / "processed"),
        help="Path to the processed dataset directory"
    )
    
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Target size for mask images (should match resized image size)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (optional)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--visualize",
        type=int,
        default=0,
        help="Number of examples to visualize (0 for none)"
    )
    
    return parser.parse_args()


def resize_mask_with_padding(
    mask: np.ndarray, 
    target_size: int,
    logger: Optional[logging.Logger] = None,
    debug: bool = False
) -> np.ndarray:
    """
    Resize mask preserving aspect ratio and pad to square.
    Uses nearest neighbor interpolation to preserve mask values.
    
    Args:
        mask: Input mask
        target_size: Target size for both dimensions after padding
        logger: Optional logger for debugging
        debug: Whether to print debug information
        
    Returns:
        Resized and padded mask
    """
    if debug and logger:
        logger.debug(f"Original mask shape: {mask.shape}, values: {np.unique(mask)}")
    
    height, width = mask.shape if len(mask.shape) == 2 else mask.shape[:2]
    
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
    
    # Resize mask using nearest neighbor interpolation to preserve label values
    resized = cv2.resize(
        mask, (new_width, new_height),
        interpolation=cv2.INTER_NEAREST
    )
    
    if debug and logger:
        logger.debug(f"Resized mask shape: {resized.shape}, values: {np.unique(resized)}")
    
    # Create a square canvas with zeros (background class)
    padded = np.zeros((target_size, target_size), dtype=mask.dtype)
    
    # Calculate padding offsets to center the mask
    pad_y = (target_size - new_height) // 2
    pad_x = (target_size - new_width) // 2
    
    # Place the resized mask on the padded canvas
    padded[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized
    
    if debug and logger:
        logger.debug(f"Padded mask shape: {padded.shape}, values: {np.unique(padded)}")
    
    return padded


def create_visualization(
    orig_mask: np.ndarray,
    resized_mask: np.ndarray,
    output_path: Path,
    index: int
) -> None:
    """
    Create a visualization comparing original and resized masks.
    
    Args:
        orig_mask: Original mask
        resized_mask: Resized mask
        output_path: Directory to save visualization
        index: Index for filename
    """
    # Create a color map for visualization
    def colorize_mask(mask):
        colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        # Class 1 (cats) - Red
        colored[mask == 1] = [0, 0, 255]  # BGR
        # Class 2 (dogs) - Green
        colored[mask == 2] = [0, 255, 0]  # BGR
        # Border/Don't care - White
        colored[mask == 255] = [255, 255, 255]  # BGR
        return colored
    
    # Resize original mask for display without changing values
    display_orig = cv2.resize(
        orig_mask, (512, 512),
        interpolation=cv2.INTER_NEAREST
    )
    
    # Colorize both masks
    colored_orig = colorize_mask(display_orig)
    colored_resized = colorize_mask(resized_mask)
    
    # Create a side-by-side comparison
    h, w = resized_mask.shape
    comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
    comparison[:, :w] = colored_orig
    comparison[:, w:] = colored_resized
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original (resized for display)", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison, "Processed", (w + 10, 30), font, 0.7, (255, 255, 255), 2)
    
    # Add color legend at the bottom
    legend_y = h - 30
    cv2.putText(comparison, "Red: Cats (1)", (10, legend_y), font, 0.6, (0, 0, 255), 2)
    cv2.putText(comparison, "Green: Dogs (2)", (150, legend_y), font, 0.6, (0, 255, 0), 2)
    cv2.putText(comparison, "White: Border (255)", (300, legend_y), font, 0.6, (255, 255, 255), 2)
    
    # Save the comparison
    cv2.imwrite(str(output_path / f"mask_comparison_{index}.jpg"), comparison)


def load_mask_properly(mask_path: Path, logger: logging.Logger) -> np.ndarray:
    """
    Load a mask with the correct class values preserved.
    
    Args:
        mask_path: Path to the mask file
        logger: Logger for output
        
    Returns:
        Mask as numpy array with correct class values
    """
    # Use PIL to load the mask which preserves the original values better
    try:
        with Image.open(mask_path) as pil_img:
            mask = np.array(pil_img)
            
        # Check if mask has class values we expect
        unique_values = np.unique(mask)
        logger.info(f"Mask loaded with PIL has values: {unique_values}")
        
        # Ensure mask is single channel
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
            
        return mask
    except Exception as e:
        logger.error(f"Error loading mask with PIL: {e}")
        
        # Fallback to OpenCV
        try:
            # Try different OpenCV reading modes to find the one that preserves values
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            logger.info(f"Mask loaded with CV2 IMREAD_UNCHANGED has values: {np.unique(mask)}")
            
            if 1 not in np.unique(mask) and 2 not in np.unique(mask):
                # Try loading as grayscale
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                logger.info(f"Mask loaded with CV2 IMREAD_GRAYSCALE has values: {np.unique(mask)}")
                
                # Special handling if we find 128 (which might be cat=1 shifted to 128)
                if 128 in np.unique(mask):
                    logger.info("Found value 128 in mask, converting to class 1 (cat)")
                    cat_mask = (mask == 128).astype(np.uint8) * 1
                    # Keep other values
                    dog_mask = (mask == 2).astype(np.uint8) * 2
                    border_mask = (mask == 255).astype(np.uint8) * 255
                    
                    # Combine masks
                    mask = cat_mask + dog_mask + border_mask
                    logger.info(f"After conversion, mask has values: {np.unique(mask)}")
            
            # Ensure mask is single channel
            if len(mask.shape) > 2:
                mask = mask[:, :, 0]
                
            return mask
        except Exception as e2:
            logger.error(f"Error loading mask with OpenCV: {e2}")
            raise ValueError(f"Failed to load mask: {mask_path}")


def process_masks(
    processed_dir: Path,
    target_size: int,
    logger: logging.Logger,
    debug: bool = False,
    visualize: int = 0
) -> None:
    """
    Process all masks in the training set.
    
    Args:
        processed_dir: Path to processed dataset directory
        target_size: Target size for masks
        logger: Logger for output
        debug: Whether to print debug information
        visualize: Number of examples to visualize
    """
    # Define paths
    train_dir = processed_dir / "Train"
    orig_mask_dir = train_dir / "label"
    resized_mask_dir = train_dir / "resized_label"
    vis_dir = resized_mask_dir.parent / "mask_visualizations"
    
    # Create output directories
    create_directory(resized_mask_dir)
    if visualize > 0:
        create_directory(vis_dir)
    
    # Get all mask files
    mask_files = list(orig_mask_dir.glob("*.png"))
    logger.info(f"Found {len(mask_files)} mask files in {orig_mask_dir}")
    
    # Test loading a few masks to check values
    for i, test_mask_path in enumerate(mask_files[:5]):
        logger.info(f"Testing mask loading for: {test_mask_path.name}")
        
        # Load with PIL
        try:
            with Image.open(test_mask_path) as pil_img:
                pil_mask = np.array(pil_img)
            logger.info(f"PIL load - unique values: {np.unique(pil_mask)}, dtype: {pil_mask.dtype}")
        except Exception as e:
            logger.warning(f"Failed to load with PIL: {e}")
        
        # Load with OpenCV different methods
        try:
            cv_mask1 = cv2.imread(str(test_mask_path), cv2.IMREAD_UNCHANGED)
            logger.info(f"OpenCV IMREAD_UNCHANGED - unique values: {np.unique(cv_mask1)}, dtype: {cv_mask1.dtype}")
            
            cv_mask2 = cv2.imread(str(test_mask_path), cv2.IMREAD_GRAYSCALE)
            logger.info(f"OpenCV IMREAD_GRAYSCALE - unique values: {np.unique(cv_mask2)}, dtype: {cv_mask2.dtype}")
            
            cv_mask3 = cv2.imread(str(test_mask_path), cv2.IMREAD_ANYDEPTH)
            logger.info(f"OpenCV IMREAD_ANYDEPTH - unique values: {np.unique(cv_mask3)}, dtype: {cv_mask3.dtype}")
        except Exception as e:
            logger.warning(f"Failed to load with OpenCV: {e}")
    
    # Process each mask
    successful = 0
    failed = 0
    visualization_count = 0
    
    for mask_path in tqdm(mask_files, desc="Resizing masks"):
        output_path = resized_mask_dir / mask_path.name
        
        try:
            # Load mask properly
            mask = load_mask_properly(mask_path, logger)
            
            # Check mask values
            unique_values = np.unique(mask)
            logger.info(f"Mask {mask_path.name} has values: {unique_values}")
            
            # Resize mask with padding
            resized_mask = resize_mask_with_padding(mask, target_size, logger, debug)
            
            # Verify values after resizing
            resized_values = np.unique(resized_mask)
            logger.info(f"After resizing, mask has values: {resized_values}")
            
            # Create visualization if requested
            if visualize > 0 and visualization_count < visualize:
                create_visualization(mask, resized_mask, vis_dir, visualization_count + 1)
                visualization_count += 1
            
            # Save resized mask - Use PIL to save to ensure values are preserved exactly
            Image.fromarray(resized_mask).save(output_path)
            
            # Double-check the saved file
            check_mask = np.array(Image.open(output_path))
            if debug:
                logger.debug(f"Saved mask has values: {np.unique(check_mask)}")
            
            successful += 1
            
        except Exception as e:
            logger.error(f"Error processing mask {mask_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            failed += 1
    
    logger.info(f"Mask processing complete: {successful} successful, {failed} failed")
    
    # Verify that masks have the correct dimensions and values
    verification_problems = 0
    masks_with_cat_class = 0
    masks_with_dog_class = 0
    
    for mask_path in resized_mask_dir.glob("*.png"):
        try:
            # Use PIL for verification to ensure we get exact saved values
            mask = np.array(Image.open(mask_path))
            
            # Check dimensions
            if mask.shape[0] != target_size or mask.shape[1] != target_size:
                logger.warning(f"Incorrect dimensions for {mask_path}: {mask.shape}")
                verification_problems += 1
                continue
                
            # Check that we have valid class values
            mask_values = np.unique(mask)
            if 1 in mask_values:
                masks_with_cat_class += 1
            if 2 in mask_values:
                masks_with_dog_class += 1
                
            if 1 not in mask_values and 2 not in mask_values:
                logger.warning(f"No valid class values in {mask_path}: {mask_values}")
                verification_problems += 1
                
        except Exception as e:
            logger.warning(f"Could not verify {mask_path}: {e}")
            verification_problems += 1
    
    logger.info(f"Verification results:")
    logger.info(f"- Masks with cat class (1): {masks_with_cat_class}")
    logger.info(f"- Masks with dog class (2): {masks_with_dog_class}")
    
    if verification_problems > 0:
        logger.warning(f"Found {verification_problems} problems during verification")
    else:
        logger.info("All masks verified successfully!")


def main() -> None:
    """Main function to preprocess the mask files."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Set paths
    processed_dir = Path(args.processed_dir)
    
    # Log basic info
    logger.info(f"Processed dataset directory: {processed_dir}")
    logger.info(f"Target size for masks: {args.size}x{args.size}")
    
    # Process masks
    process_masks(processed_dir, args.size, logger, args.debug, args.visualize)
    
    logger.info("Mask preprocessing complete!")


if __name__ == "__main__":
    main()