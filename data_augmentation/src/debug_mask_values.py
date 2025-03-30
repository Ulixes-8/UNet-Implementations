#!/usr/bin/env python
"""
Script: debug_mask_values.py

This script analyzes the actual pixel values in the validation and test masks
to understand the encoding used for cat and dog classes.
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from collections import Counter

# Add the project root to path to import our modules
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Define paths
PROCESSED_DIR = Path(project_root / "data" / "processed") 
TEST_MASK_DIR = PROCESSED_DIR / "Test" / "label"
VAL_MASK_DIR = PROCESSED_DIR / "Val" / "label"

# List of cat breeds to identify cat images
CAT_BREEDS = [
    'abyssinian', 'bengal', 'birman', 'bombay', 
    'british', 'egyptian', 'maine', 
    'persian', 'ragdoll', 'russian', 'siamese', 'sphynx'
]

def is_cat_image(filename: str) -> bool:
    """Determine if an image is a cat based on filename."""
    filename_lower = filename.lower()
    return any(breed in filename_lower for breed in CAT_BREEDS)

def analyze_mask_detailed(mask_path, is_cat):
    """Analyze a mask in detail, printing various info about pixel values"""
    print(f"\nAnalyzing {'cat' if is_cat else 'dog'} mask: {mask_path.name}")
    
    # Load with PIL and check shape
    pil_mask = np.array(Image.open(mask_path))
    print(f"PIL load - shape: {pil_mask.shape}, dtype: {pil_mask.dtype}")
    
    # For 3D masks, check individual channel values
    if len(pil_mask.shape) == 3:
        print("3D mask detected, analyzing channels:")
        for i in range(pil_mask.shape[2]):
            channel = pil_mask[:, :, i]
            print(f"  Channel {i}: unique values = {np.unique(channel)}")
        
        # Count pixel value combinations
        # Reshape to a list of pixels (each pixel is an RGB tuple)
        pixels = pil_mask.reshape(-1, pil_mask.shape[2])
        
        # Find the most common RGB values
        pixel_counter = Counter(map(tuple, pixels))
        print("\nMost common pixel values (RGB tuples):")
        for pixel, count in pixel_counter.most_common(5):
            print(f"  {pixel}: {count} pixels")
    
    # Try OpenCV loads
    cv_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    print(f"OpenCV IMREAD_UNCHANGED - shape: {cv_mask.shape}, unique values: {np.unique(cv_mask)}")
    
    cv_mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    print(f"OpenCV IMREAD_GRAYSCALE - unique values: {np.unique(cv_mask_gray)}")

def main():
    """Analyze masks to understand their encoding"""
    print("Analyzing Test masks...")
    
    # Analyze a few cat and dog masks from test set
    cat_masks = []
    dog_masks = []
    
    # Get all mask files
    for mask_file in TEST_MASK_DIR.glob("*.png"):
        if is_cat_image(mask_file.stem):
            cat_masks.append(mask_file)
        else:
            dog_masks.append(mask_file)
    
    # Analyze a few of each
    for mask in cat_masks[:3]:
        analyze_mask_detailed(mask, True)
    
    for mask in dog_masks[:3]:
        analyze_mask_detailed(mask, False)
    
    print("\n------ Summary ------")
    print(f"Found {len(cat_masks)} cat masks and {len(dog_masks)} dog masks in Test set")

if __name__ == "__main__":
    main()