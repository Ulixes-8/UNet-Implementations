"""
Module: helpers.py

This module provides utility functions for working with image datasets,
file I/O operations, and other common tasks.
"""

import os
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


def seed_everything(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and other libraries.
    
    Args:
        seed: The seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_image(
    image_path: Union[str, Path], 
    grayscale: bool = False,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Load an image from disk.
    
    Args:
        image_path: Path to the image file
        grayscale: Whether to load as grayscale
        target_size: Optional (width, height) to resize to
        
    Returns:
        Loaded image as a NumPy array (BGR format if not grayscale)
    
    Raises:
        ValueError: If the image cannot be loaded
    """
    # Convert to string for OpenCV
    image_path_str = str(image_path)
    
    # Read the image
    if grayscale:
        image = cv2.imread(image_path_str, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path_str, cv2.IMREAD_COLOR)
    
    # Check if image was loaded successfully
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Resize if needed
    if target_size is not None:
        image = cv2.resize(image, target_size)
    
    return image


def load_mask(
    mask_path: Union[str, Path], 
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Load a segmentation mask from disk.
    
    Args:
        mask_path: Path to the mask file
        target_size: Optional (width, height) to resize to
        
    Returns:
        Loaded mask as a NumPy array with original values
    
    Raises:
        ValueError: If the mask cannot be loaded
    """
    # Use PIL to preserve label values during loading
    try:
        mask = np.array(Image.open(str(mask_path)))
    except Exception as e:
        raise ValueError(f"Failed to load mask {mask_path}: {e}")
    
    # Resize if needed (using nearest neighbor to preserve label values)
    if target_size is not None:
        mask = cv2.resize(
            mask, target_size, 
            interpolation=cv2.INTER_NEAREST
        )
    
    return mask


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


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Get the file extension from a file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension (e.g., '.jpg', '.png')
    """
    return Path(file_path).suffix.lower()


def split_dataset(
    file_list: List[Path], 
    val_ratio: float = 0.2, 
    seed: int = 42
) -> Tuple[List[Path], List[Path]]:
    """
    Split a list of files into training and validation sets.
    
    Args:
        file_list: List of file paths to split
        val_ratio: Proportion of files to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_files, val_files)
    """
    # Set seed for reproducibility
    random.seed(seed)
    
    # Shuffle files
    shuffled_files = file_list.copy()
    random.shuffle(shuffled_files)
    
    # Calculate split index
    val_count = max(1, int(len(file_list) * val_ratio))
    
    # Split the dataset
    val_files = shuffled_files[:val_count]
    train_files = shuffled_files[val_count:]
    
    return train_files, val_files


def copy_files(
    files: List[Path], 
    dest_dir: Union[str, Path], 
    preserve_structure: bool = False,
    src_root: Optional[Path] = None
) -> None:
    """
    Copy a list of files to a destination directory.
    
    Args:
        files: List of file paths to copy
        dest_dir: Destination directory
        preserve_structure: Whether to preserve the directory structure
        src_root: Source root directory (used with preserve_structure)
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in files:
        if preserve_structure and src_root:
            # Calculate relative path from src_root
            rel_path = file_path.relative_to(src_root)
            dest_file = dest_dir / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Just copy to flat destination
            dest_file = dest_dir / file_path.name
        
        shutil.copy2(file_path, dest_file)


def visualize_segmentation_mask(
    mask: np.ndarray, 
    num_classes: int
) -> np.ndarray:
    """
    Create a colored visualization of a segmentation mask.
    
    Args:
        mask: Segmentation mask with class indices
        num_classes: Number of classes (including background)
        
    Returns:
        Colored visualization as an RGB image
    """
    # Define a color map for visualization
    # Using distinct colors for better visibility
    color_map = np.array([
        [0, 0, 0],        # Class 0: Background (black)
        [255, 0, 0],      # Class 1: Red
        [0, 255, 0],      # Class 2: Green
        [0, 0, 255],      # Class 3: Blue
        [255, 255, 0],    # Class 4: Yellow
        [255, 0, 255],    # Class 5: Magenta
        [0, 255, 255],    # Class 6: Cyan
        [128, 0, 0],      # Class 7: Maroon
        [0, 128, 0],      # Class 8: Dark Green
        [0, 0, 128],      # Class 9: Navy Blue
    ])
    
    # Ensure we have enough colors for all classes
    if num_classes > len(color_map):
        # Add random colors for additional classes
        additional_colors = np.random.randint(
            0, 255, size=(num_classes - len(color_map), 3)
        )
        color_map = np.vstack([color_map, additional_colors])
    
    # Create a colored image
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    # Assign colors to each class
    for class_idx in range(num_classes):
        colored_mask[mask == class_idx] = color_map[class_idx]
    
    return colored_mask