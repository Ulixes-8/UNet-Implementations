#!/usr/bin/env python
"""
Script: preprocess_test_val.py

This script processes the validation and test mask files without resizing them, 
converting them from 3D to 2D format and standardizing class values:
- 0: background
- 1: cat
- 2: dog
- 255: border/don't care

This script is necessary because the original masks are 3D with values [0, 128, 255],
while the processed training masks are 2D with values [0, 1, 2, 255].

IMPORTANT: Unlike the training masks, validation and test masks maintain their original
dimensions and are NOT resized.

Example Usage:
    python scripts/preprocess_test_val.py
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
    logger = logging.getLogger("preprocess_test_val_masks")
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
        description="Resize and process validation and test mask files to match the dimensions of resized images"
    )
    
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(project_root / "data" / "processed"),
        help="Path to the processed dataset directory"
    )
    
    # No size parameter needed since we're not resizing
    
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


# No resizing function needed as we're preserving original dimensions


def create_visualization(
    orig_mask: np.ndarray,
    processed_mask: np.ndarray,
    output_path: Path,
    index: int,
    is_cat: bool
) -> None:
    """
    Create a visualization comparing original and processed masks.
    
    Args:
        orig_mask: Original mask
        processed_mask: Processed mask
        output_path: Directory to save visualization
        index: Index for filename
        is_cat: Whether this is a cat image
    """
    # Create a color map for visualization
    def colorize_mask(mask, is_cat=None):
        colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        if is_cat is not None and 128 in np.unique(mask):
            # For original masks with 128 value
            if is_cat:
                colored[mask == 128] = [0, 0, 255]  # Red in BGR (cat)
            else:
                colored[mask == 128] = [0, 255, 0]  # Green in BGR (dog)
        else:
            # For processed masks with class values
            colored[mask == 1] = [0, 0, 255]    # Red in BGR (cat)
            colored[mask == 2] = [0, 255, 0]    # Green in BGR (dog)
            
        # Common values
        colored[mask == 255] = [255, 255, 255]  # White (border)
        
        return colored
    
    # Colorize both masks
    colored_orig = colorize_mask(orig_mask, is_cat)
    colored_processed = colorize_mask(processed_mask)
    
    # Create a side-by-side comparison
    h, w = processed_mask.shape
    comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
    comparison[:, :w] = colored_orig
    comparison[:, w:] = colored_processed
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, h / 500)  # Scale font based on image height
    cv2.putText(comparison, "Original", (10, 30), font, font_scale, (255, 255, 255), 2)
    cv2.putText(comparison, "Processed", (w + 10, 30), font, font_scale, (255, 255, 255), 2)
    
    # Add class labels
    animal_type = "Cat" if is_cat else "Dog"
    cv2.putText(comparison, animal_type, (10, 60), font, font_scale, (255, 255, 255), 2)
    
    # Add original values
    orig_values = np.unique(orig_mask)
    processed_values = np.unique(processed_mask)
    cv2.putText(comparison, f"Values: {orig_values}", (10, h - 30), font, 0.5, (255, 255, 255), 1)
    cv2.putText(comparison, f"Values: {processed_values}", (w + 10, h - 30), font, 0.5, (255, 255, 255), 1)
    
    # Save the comparison
    cv2.imwrite(str(output_path / f"mask_comparison_{index}.jpg"), comparison)


def is_cat_image(filename: str) -> bool:
    """
    Determine if an image is a cat based on filename.
    
    Args:
        filename: Image filename
        
    Returns:
        True if it's a cat image, False otherwise (dog)
    """
    # List of cat breeds to match in the filename
    cat_breeds = [
        'abyssinian', 'bengal', 'birman', 'bombay', 
        'british', 'egyptian', 'maine', 
        'persian', 'ragdoll', 'russian', 'siamese', 'sphynx'
    ]
    
    filename_lower = filename.lower()
    return any(breed in filename_lower for breed in cat_breeds)


def process_mask(
    mask_path: Path,
    output_path: Path,
    logger: logging.Logger,
    debug: bool = False
) -> np.ndarray:
    """
    Process a single mask file:
    1. Load the mask
    2. Convert from 3D to 2D if needed
    3. Determine class (cat=1 or dog=2) based on filename
    4. Remap pixel values to proper class values
    5. Keep original dimensions (NO resizing)
    
    Args:
        mask_path: Path to the original mask file
        output_path: Path where the processed mask will be saved
        logger: Logger for output
        debug: Whether to print debug information
        
    Returns:
        The processed mask
    """
    try:
        # Load the original mask
        with Image.open(mask_path) as pil_img:
            mask = np.array(pil_img)
        
        # Store original shape for debugging
        original_shape = mask.shape
        
        # Check if mask is 3D, convert to 2D
        if len(mask.shape) > 2:
            if debug:
                logger.debug(f"Converting 3D mask to 2D: {mask_path}, shape: {mask.shape}")
            mask = mask[:, :, 0]  # Take first channel
        
        # Get unique values
        unique_values = np.unique(mask)
        if debug:
            logger.debug(f"Mask {mask_path.name}, shape: {original_shape}, values: {unique_values}")
        
        # Determine if this is a cat or dog based on filename
        is_cat = is_cat_image(mask_path.stem)
        animal_type = "cat" if is_cat else "dog"
        
        # Create a new mask with proper class values
        processed_mask = np.zeros_like(mask)
        
        # IMPORTANT: Enhanced class value detection
        # We need to identify which pixel values represent the foreground object
        
        # If we have the expected 128 value
        if 128 in unique_values:
            if is_cat:
                processed_mask[mask == 128] = 1  # Cat
            else:
                processed_mask[mask == 128] = 2  # Dog
        # Try other possible foreground values if 128 not found
        elif len(unique_values) > 1:
            # Exclude 0 (background) and 255 (border)
            foreground_values = [v for v in unique_values if v not in (0, 255)]
            
            if foreground_values:
                # Use the first non-background, non-border value as foreground
                foreground_value = foreground_values[0]
                logger.info(f"Using value {foreground_value} as foreground for {mask_path.name}")
                
                if is_cat:
                    processed_mask[mask == foreground_value] = 1  # Cat
                else:
                    processed_mask[mask == foreground_value] = 2  # Dog
            else:
                # No foreground values found, check if there's any non-zero pixel
                # This is a fallback for masks that might be using a different encoding
                non_zero_mask = (mask > 0) & (mask < 255)
                if np.any(non_zero_mask):
                    if is_cat:
                        processed_mask[non_zero_mask] = 1  # Cat
                    else:
                        processed_mask[non_zero_mask] = 2  # Dog
                    logger.info(f"Using non-zero pixels as foreground for {mask_path.name}")
                else:
                    # Last resort: use histogram analysis to find the most likely foreground
                    # Masks often have large background (0) areas, smaller foreground areas,
                    # and very small border (255) areas
                    values, counts = np.unique(mask, return_counts=True)
                    # Sort by frequency (ascending)
                    sorted_indices = np.argsort(counts)
                    
                    # The middle frequency value is often the foreground
                    # (not the most common which is background, not the least common which might be noise)
                    if len(sorted_indices) >= 3:
                        middle_value = values[sorted_indices[-2]]  # Second most common
                        if middle_value not in (0, 255):
                            if is_cat:
                                processed_mask[mask == middle_value] = 1  # Cat
                            else:
                                processed_mask[mask == middle_value] = 2  # Dog
                            logger.info(f"Using histogram analysis value {middle_value} for {mask_path.name}")
        
        # Border/Don't care remains 255
        processed_mask[mask == 255] = 255
        
        # Check if we've created a valid mask with foreground objects
        if 1 not in processed_mask and 2 not in processed_mask:
            # As a last resort, create a simple foreground based on non-border, non-background pixels
            if is_cat:
                # Create a basic cat mask where anything not 0 or 255 becomes 1
                processed_mask[(mask != 0) & (mask != 255)] = 1
            else:
                # Create a basic dog mask where anything not 0 or 255 becomes 2
                processed_mask[(mask != 0) & (mask != 255)] = 2
            
            logger.warning(f"Created fallback mask for {mask_path.name} - no foreground detected in original")
        
        # Save the processed mask - no resizing
        Image.fromarray(processed_mask.astype(np.uint8)).save(output_path)
        
        # Final verification
        final_unique = np.unique(processed_mask)
        if debug or 1 not in final_unique and 2 not in final_unique:
            logger.debug(f"Processed {mask_path.name} as {animal_type}, values: {final_unique}")
        
        return processed_mask
        
    except Exception as e:
        logger.error(f"Error processing mask {mask_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def process_split_masks(
    split_name: str,
    processed_dir: Path,
    logger: logging.Logger,
    debug: bool = False,
    visualize: int = 0
) -> None:
    """
    Process all masks in a specific split (Val or Test).
    
    Args:
        split_name: Name of the split ('Val' or 'Test')
        processed_dir: Path to processed dataset directory
        logger: Logger for output
        debug: Whether to print debug information
        visualize: Number of examples to visualize
    """
    # Define paths
    split_dir = processed_dir / split_name
    orig_mask_dir = split_dir / "label"
    resized_mask_dir = split_dir / "processed_labels"
    vis_dir = resized_mask_dir.parent / "mask_visualizations"
    
    # Create output directories
    create_directory(resized_mask_dir)
    if visualize > 0:
        create_directory(vis_dir)
    
    # Get all mask files
    mask_files = list(orig_mask_dir.glob("*.png"))
    logger.info(f"Found {len(mask_files)} mask files in {orig_mask_dir}")
    
    # Test loading a few masks to check values
    for i, test_mask_path in enumerate(mask_files[:3]):
        logger.info(f"Testing mask loading for: {test_mask_path.name}")
        
        # Load with PIL
        try:
            with Image.open(test_mask_path) as pil_img:
                pil_mask = np.array(pil_img)
            logger.info(f"PIL load - shape: {pil_mask.shape}, unique values: {np.unique(pil_mask)}")
        except Exception as e:
            logger.warning(f"Failed to load with PIL: {e}")
    
    # Process each mask
    successful = 0
    failed = 0
    visualization_count = 0
    
    for mask_path in tqdm(mask_files, desc=f"Processing {split_name} masks"):
        output_path = resized_mask_dir / mask_path.name
        
        # Determine if this is a cat or dog
        is_cat = is_cat_image(mask_path.stem)
        
        # Load original mask for visualization if needed
        if visualize > 0 and visualization_count < visualize:
            with Image.open(mask_path) as pil_img:
                orig_mask = np.array(pil_img)
                if len(orig_mask.shape) > 2:
                    orig_mask = orig_mask[:, :, 0]  # Take first channel for visualization
        
        # Process the mask
        processed_mask = process_mask(
            mask_path, output_path, logger, debug
        )
        
        if processed_mask is not None:
            successful += 1
            
            # Create visualization if requested
            if visualize > 0 and visualization_count < visualize:
                create_visualization(
                    orig_mask, processed_mask, vis_dir, 
                    visualization_count + 1, is_cat
                )
                visualization_count += 1
        else:
            failed += 1
    
    logger.info(f"{split_name} mask processing complete: {successful} successful, {failed} failed")
    
    # Verify that masks have the correct values
    logger.info(f"Verifying processed masks...")
    verification_problems = 0
    masks_with_cat_class = 0
    masks_with_dog_class = 0
    
    for mask_path in resized_mask_dir.glob("*.png"):
        try:
            # Use PIL for verification to ensure we get exact saved values
            mask = np.array(Image.open(mask_path))
                
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
    
    logger.info(f"Verification results for {split_name}:")
    logger.info(f"- Masks with cat class (1): {masks_with_cat_class}")
    logger.info(f"- Masks with dog class (2): {masks_with_dog_class}")
    
    if verification_problems > 0:
        logger.warning(f"Found {verification_problems} problems during verification")
    else:
        logger.info(f"All {split_name} masks verified successfully!")


def main() -> None:
    """Main function to preprocess test and validation mask files."""
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
    logger.info("IMPORTANT: Keeping original mask dimensions (no resizing)")
    
    # Process validation masks
    logger.info("Processing validation masks...")
    process_split_masks(
        'Val', processed_dir, logger, args.debug, args.visualize
    )
    
    # Process test masks
    logger.info("Processing test masks...")
    process_split_masks(
        'Test', processed_dir, logger, args.debug, args.visualize
    )
    
    logger.info("Test and validation mask preprocessing complete!")


if __name__ == "__main__":
    main()