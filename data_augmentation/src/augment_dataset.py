#!/usr/bin/env python
"""
Script: augment_dataset.py

This script applies data augmentation to the Oxford-IIIT Pet Dataset
for semantic segmentation, implementing class-specific augmentation
to address the class imbalance between cats and dogs.

Key features:
1. Applies more aggressive augmentation to cats (minority class)
2. Uses conservative augmentation for dogs (majority class)
3. Generates 3 augmented versions of each cat image and 1 per dog image
4. Ensures proper handling of mask labels during augmentation
5. Maintains directory structure and clear naming conventions

Example Usage:
    python scripts/augment_dataset.py --config config/augmentation_config.yaml
"""

import argparse
import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import albumentations as A
from PIL import Image
from tqdm import tqdm

# Add the project root to path to import our modules
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

#from the utils directory, import the create_directory and seed_everything functions
from utils.helpers import create_directory, seed_everything


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Optional path to log file
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("augmentation")
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
        description="Augment the Oxford-IIIT Pet Dataset for semantic segmentation"
    )
    
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(project_root / "data" / "processed"),
        help="Path to processed dataset directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to output augmented data (defaults to processed-dir/Train/augmented)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root / "config" / "augmentation_config.yaml"),
        help="Path to augmentation configuration file"
    )
    
    parser.add_argument(
        "--cat-augmentations",
        type=int,
        default=5,
        help="Number of augmentations to generate per cat image"
    )
    
    parser.add_argument(
        "--dog-augmentations",
        type=int,
        default=2,
        help="Number of augmentations to generate per dog image"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (optional)"
    )
    
    parser.add_argument(
        "--visualize",
        type=int,
        default=0,
        help="Number of augmentation examples to visualize (0 for none)"
    )
    
    parser.add_argument(
        "--debug-mask",
        action="store_true",
        help="Enable debugging of mask loading/saving"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load augmentation configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the configuration file is invalid
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)


def create_augmentation_pipelines(config: Dict[str, Any]) -> Tuple[A.Compose, A.Compose]:
    """
    Create cat and dog augmentation pipelines from configuration.
    
    Args:
        config: Dictionary containing augmentation parameters
        
    Returns:
        Tuple of (cat_transform, dog_transform)
    """
    # Create cat augmentation pipeline (more aggressive)
    cat_transform = A.Compose([
        # Spatial Transforms - More aggressive for cats
        A.HorizontalFlip(p=config['cat']['horizontal_flip_prob']),
        A.ShiftScaleRotate(
            scale_limit=config['cat']['scale_limit'],
            rotate_limit=config['cat']['rotate_limit'],
            shift_limit=config['cat']['shift_limit'],
            p=config['cat']['shift_scale_rotate_prob'],
            border_mode=cv2.BORDER_CONSTANT,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST  # Critical for mask value preservation
        ),
        
        # RandomResizedCrop - previously removed due to potential mask misalignment, now testing again
        A.RandomResizedCrop(
            # size=config['cat']['random_resized_crop']['size'],
            size=(512, 512),
            scale=tuple(config['cat']['random_resized_crop']['scale']),
            ratio=tuple(config['cat']['random_resized_crop']['ratio']),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,  # Critical for mask value preservation
            p=config['cat']['random_resized_crop']['prob']
        ),
        
        # CoarseDropout for occlusion tests (g)
        A.CoarseDropout(
            max_holes=config['cat']['coarse_dropout']['max_holes'],
            max_height=config['cat']['coarse_dropout']['max_height'],
            max_width=config['cat']['coarse_dropout']['max_width'],
            min_height=config['cat']['coarse_dropout']['min_height'],
            min_width=config['cat']['coarse_dropout']['min_width'],
            fill_value=config['cat']['coarse_dropout']['fill_value'],
            mask_fill_value=0,  # For masks, keeping background value
            p=config['cat']['coarse_dropout']['prob']
        ),
        
        # Elastic Transforms - Only for cats
        A.OneOf([
            A.ElasticTransform(
                alpha=config['cat']['elastic']['alpha'],
                sigma=config['cat']['elastic']['sigma'],
                alpha_affine=config['cat']['elastic']['alpha_affine'],
                p=config['cat']['elastic']['prob'],
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST  # Critical for mask value preservation
            ),
            A.GridDistortion(
                num_steps=config['cat']['grid_distortion']['num_steps'],
                distort_limit=config['cat']['grid_distortion']['distort_limit'],
                p=config['cat']['grid_distortion']['prob'],
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST  # Critical for mask value preservation
            ),
            A.OpticalDistortion(
                distort_limit=config['cat']['optical_distortion']['distort_limit'],
                shift_limit=config['cat']['optical_distortion']['shift_limit'],
                p=config['cat']['optical_distortion']['prob'],
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST  # Critical for mask value preservation
            )
        ], p=config['cat']['elastic_transform_prob']),
        
        # Additional Spatial Transform - Perspective
        A.Perspective(
            scale=tuple(config['cat']['perspective']['scale']),
            keep_size=True,
            pad_mode=cv2.BORDER_CONSTANT,
            mask_pad_val=0,
            fit_output=False,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,  # Critical for mask value preservation
            p=config['cat']['perspective']['prob']
        ),
        
        # Pixel-level Transforms - More variety for cats
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=config['cat']['brightness_contrast']['brightness_limit'],
                contrast_limit=config['cat']['brightness_contrast']['contrast_limit'],
                p=config['cat']['brightness_contrast']['prob']
            ),
            A.HueSaturationValue(
                hue_shift_limit=config['cat']['hsv']['hue_shift_limit'],
                sat_shift_limit=config['cat']['hsv']['sat_shift_limit'],
                val_shift_limit=config['cat']['hsv']['val_shift_limit'],
                p=config['cat']['hsv']['prob']
            ),
            A.RGBShift(
                r_shift_limit=config['cat']['rgb_shift']['r_shift_limit'],
                g_shift_limit=config['cat']['rgb_shift']['g_shift_limit'],
                b_shift_limit=config['cat']['rgb_shift']['b_shift_limit'],
                p=config['cat']['rgb_shift']['prob']
            )
        ], p=config['cat']['color_transform_prob']),
        
        # Additional Color Transformations
        A.OneOf([
            A.CLAHE(
                clip_limit=config['cat']['clahe_equalize']['clahe_clip_limit'], 
                tile_grid_size=tuple(config['cat']['clahe_equalize']['clahe_tile_grid_size']), 
                p=config['cat']['clahe_equalize']['clahe_prob']
            ),
            A.Equalize(
                mode='cv', 
                by_channels=True, 
                p=config['cat']['clahe_equalize']['equalize_prob']
            ),
            A.ToGray(
                p=config['cat']['clahe_equalize']['to_gray_prob']
            )
        ], p=config['cat']['clahe_equalize']['prob']),
        
        # Noise and blur - More for cats
        A.OneOf([
            A.GaussNoise(
                var_limit=config['cat']['gauss_noise']['var_limit'],
                p=config['cat']['gauss_noise']['prob']
            ),
            A.GaussianBlur(
                blur_limit=config['cat']['gaussian_blur']['blur_limit'],
                p=config['cat']['gaussian_blur']['prob']
            ),
            A.MotionBlur(
                blur_limit=config['cat']['motion_blur']['blur_limit'],
                p=config['cat']['motion_blur']['prob']
            ),
        ], p=config['cat']['noise_transform_prob']),
        
        # Salt and Pepper Noise
        A.SaltAndPepper(
            always_apply=False,
            p=config['cat']['salt_pepper']['prob'],
            salt_p=config['cat']['salt_pepper']['salt_p'],
            pepper_p=config['cat']['salt_pepper']['pepper_p']
        ),
        
        # ISO Noise (camera sensor noise)
        A.ISONoise(
            color_shift=tuple(config['cat']['iso_noise']['color_shift']),
            intensity=tuple(config['cat']['iso_noise']['intensity']),
            p=config['cat']['iso_noise']['prob']
        ),
        
        # Lighting variations
        A.OneOf([
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                p=config['cat']['shadow']['prob']
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 1),
                angle_lower=0,
                angle_upper=1,
                p=config['cat']['sunflare']['prob']
            ),
            # Random Fog for cats
            A.RandomFog(
                fog_coef_lower=config['cat']['fog']['fog_coef_lower'], 
                fog_coef_upper=config['cat']['fog']['fog_coef_upper'], 
                alpha_coef=config['cat']['fog']['alpha_coef'], 
                p=config['cat']['fog']['prob']
            ),
        ], p=config['cat']['lighting_transform_prob']),
    ])
    
    # Create dog augmentation pipeline
    dog_transform = A.Compose([
        # Spatial Transforms for dogs
        A.HorizontalFlip(p=config['dog']['horizontal_flip_prob']),
        A.ShiftScaleRotate(
            scale_limit=config['dog']['scale_limit'],
            rotate_limit=config['dog']['rotate_limit'],
            shift_limit=config['dog']['shift_limit'],
            p=config['dog']['shift_scale_rotate_prob'],
            border_mode=cv2.BORDER_CONSTANT,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST  # Critical for mask value preservation
        ),
        
        # RandomResizedCrop - previously removed due to potential mask misalignment, now testing again
        A.RandomResizedCrop(
            size=(512, 512),
            # size=config['dog']['random_resized_crop']['size'],
            scale=tuple(config['dog']['random_resized_crop']['scale']),
            ratio=tuple(config['dog']['random_resized_crop']['ratio']),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,  # Critical for mask value preservation
            p=config['dog']['random_resized_crop']['prob']
        ),
        
        # CoarseDropout for occlusion tests (g)
        A.CoarseDropout(
            max_holes=config['dog']['coarse_dropout']['max_holes'],
            max_height=config['dog']['coarse_dropout']['max_height'],
            max_width=config['dog']['coarse_dropout']['max_width'],
            min_height=config['dog']['coarse_dropout']['min_height'],
            min_width=config['dog']['coarse_dropout']['min_width'],
            fill_value=config['dog']['coarse_dropout']['fill_value'],
            mask_fill_value=0,  # For masks, keeping background value
            p=config['dog']['coarse_dropout']['prob']
        ),
        
        # Elastic transforms for dogs
        A.OneOf([
            A.ElasticTransform(
                alpha=config['dog']['elastic']['alpha'],
                sigma=config['dog']['elastic']['sigma'],
                alpha_affine=config['dog']['elastic']['alpha_affine'],
                p=config['dog']['elastic']['prob'],
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST
            ),
            A.GridDistortion(
                num_steps=config['dog']['grid_distortion']['num_steps'],
                distort_limit=config['dog']['grid_distortion']['distort_limit'],
                p=config['dog']['grid_distortion']['prob'],
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST
            ),
        ], p=config['dog']['elastic_transform_prob']),
        
        # Perspective for dogs
        A.Perspective(
            scale=tuple(config['dog']['perspective']['scale']),
            keep_size=True,
            pad_mode=cv2.BORDER_CONSTANT,
            mask_pad_val=0,
            fit_output=False,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            p=config['dog']['perspective']['prob']
        ),
        
        # Pixel-level Transforms for dogs
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=config['dog']['brightness_contrast']['brightness_limit'],
                contrast_limit=config['dog']['brightness_contrast']['contrast_limit'],
                p=config['dog']['brightness_contrast']['prob']
            ),
            A.HueSaturationValue(
                hue_shift_limit=config['dog']['hsv']['hue_shift_limit'],
                sat_shift_limit=config['dog']['hsv']['sat_shift_limit'],
                val_shift_limit=config['dog']['hsv']['val_shift_limit'],
                p=config['dog']['hsv']['prob']
            ),
            A.RGBShift(
                r_shift_limit=config['dog']['rgb_shift']['r_shift_limit'],
                g_shift_limit=config['dog']['rgb_shift']['g_shift_limit'],
                b_shift_limit=config['dog']['rgb_shift']['b_shift_limit'],
                p=config['dog']['rgb_shift']['prob']
            ),
        ], p=config['dog']['color_transform_prob']),
        
        # Additional Color Transformations for dogs
        A.OneOf([
            A.CLAHE(
                clip_limit=config['dog']['clahe_equalize']['clahe_clip_limit'], 
                tile_grid_size=tuple(config['dog']['clahe_equalize']['clahe_tile_grid_size']), 
                p=config['dog']['clahe_equalize']['clahe_prob']
            ),
            A.Equalize(
                mode='cv', 
                by_channels=True, 
                p=config['dog']['clahe_equalize']['equalize_prob']
            ),
            A.ToGray(
                p=config['dog']['clahe_equalize']['to_gray_prob']
            )
        ], p=config['dog']['clahe_equalize']['prob']),
        
        # Noise/blur for dogs
        A.OneOf([
            A.GaussNoise(
                var_limit=config['dog']['gauss_noise']['var_limit'],
                p=config['dog']['gauss_noise']['prob']
            ),
            A.GaussianBlur(
                blur_limit=config['dog']['gaussian_blur']['blur_limit'],
                p=config['dog']['gaussian_blur']['prob']
            ),
            A.MotionBlur(
                blur_limit=config['dog']['motion_blur']['blur_limit'],
                p=config['dog']['motion_blur']['prob']
            ),
        ], p=config['dog']['noise_transform_prob']),
        
        # Salt and Pepper Noise
        A.SaltAndPepper(
            always_apply=False,
            p=config['dog']['salt_pepper']['prob'],
            salt_p=config['dog']['salt_pepper']['salt_p'],
            pepper_p=config['dog']['salt_pepper']['pepper_p']
        ),
        
        # ISO Noise (camera sensor noise)
        A.ISONoise(
            color_shift=tuple(config['dog']['iso_noise']['color_shift']),
            intensity=tuple(config['dog']['iso_noise']['intensity']),
            p=config['dog']['iso_noise']['prob']
        ),
        
        # Lighting variations for dogs
        A.OneOf([
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                p=config['dog']['shadow']['prob']
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 1),
                angle_lower=0,
                angle_upper=1,
                p=config['dog']['sunflare']['prob']
            ),
            # Random Fog for dogs
            A.RandomFog(
                fog_coef_lower=config['dog']['fog']['fog_coef_lower'], 
                fog_coef_upper=config['dog']['fog']['fog_coef_upper'], 
                alpha_coef=config['dog']['fog']['alpha_coef'], 
                p=config['dog']['fog']['prob']
            ),
        ], p=config['dog']['lighting_transform_prob']),
    ])
    
    return cat_transform, dog_transform
def get_class_from_mask(mask: np.ndarray, filename: str = None) -> int:
    """
    Determine if a mask contains a cat (1) or dog (2) based on mask values and filename.
    
    Args:
        mask: Numpy array containing mask data
        filename: Original filename to check breed
        
    Returns:
        1 for cat, 2 for dog, 0 if neither could be determined
    """
    # First priority: Check mask values directly (most reliable)
    unique_values = np.unique(mask)
    
    if 1 in unique_values:
        return 1  # Cat
    elif 2 in unique_values:
        return 2  # Dog
    
    # If mask check is inconclusive, check filename
    if filename:
        # Complete list of cat breeds - using shortened forms to improve matching
        cat_breeds = [
            'abyssinian', 'bengal', 'birman', 'bombay', 
            'british', 'egyptian', 'maine', 
            'persian', 'ragdoll', 'russian', 'siamese', 'sphynx'
        ]
        
        # Complete list of dog breeds - using shortened forms to improve matching
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
    
    # If we get here, we couldn't determine the class
    return 0  # Unknown

def resize_mask_to_match_image(
    mask: np.ndarray, 
    target_shape: Tuple[int, int],
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Resize a mask to match the shape of the corresponding image
    using nearest neighbor interpolation to preserve label values.
    
    Args:
        mask: Original mask as numpy array
        target_shape: Target shape (height, width)
        logger: Optional logger for debugging
        
    Returns:
        Resized mask as numpy array
    """
    # Ensure mask is uint8
    mask = mask.astype(np.uint8)
    
    if logger:
        logger.debug(f"Resizing mask from {mask.shape} to {target_shape}")
        logger.debug(f"Original mask values: {np.unique(mask)}")
    
    # Resize mask using nearest neighbor interpolation to preserve label values
    resized_mask = cv2.resize(
        mask, (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    
    if logger:
        logger.debug(f"Resized mask values: {np.unique(resized_mask)}")
    
    return resized_mask


def load_image_and_mask(
    img_path: Path,
    mask_path: Path,
    logger: Optional[logging.Logger] = None,
    debug_mask: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an image and its corresponding mask.
    Resize mask to match image dimensions if needed.
    
    Args:
        img_path: Path to the image file
        mask_path: Path to the mask file
        logger: Optional logger for debugging
        debug_mask: Whether to print debug information for mask values
        
    Returns:
        Tuple of (image, mask) as numpy arrays with same dimensions
    
    Raises:
        ValueError: If the image or mask cannot be loaded
    """
    # Read image with OpenCV (BGR format)
    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Failed to read image: {img_path}")
    
    # Convert to RGB for consistency with most models
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Read mask with OpenCV to preserve label values
    # This is more reliable than PIL for specific values
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")
    
    if debug_mask and logger:
        logger.info(f"Loaded mask from {mask_path}, shape: {mask.shape}, values: {np.unique(mask)}")
    
    # Resize mask to match image dimensions if needed
    if image.shape[:2] != mask.shape[:2]:
        mask = resize_mask_to_match_image(mask, image.shape[:2], logger)
    
    print(f"DEBUG - Raw mask from {mask_path.name}, values: {np.unique(mask)}")

    return image, mask


def save_augmented_data(
    image: np.ndarray,
    mask: np.ndarray,
    output_img_path: Path,
    output_mask_path: Path,
    logger: Optional[logging.Logger] = None,
    debug_mask: bool = False
) -> None:
    """
    Save the augmented image and mask.
    
    Args:
        image: Augmented image as numpy array (RGB)
        mask: Augmented mask as numpy array
        output_img_path: Path to save the augmented image
        output_mask_path: Path to save the augmented mask
        logger: Optional logger for debugging
        debug_mask: Whether to print debug information for mask values
    """
    # Create output directories if they don't exist
    output_img_path.parent.mkdir(parents=True, exist_ok=True)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure mask is uint8
    mask = mask.astype(np.uint8)
    
    if debug_mask and logger:
        logger.info(f"Saving mask to {output_mask_path}, values before save: {np.unique(mask)}")
    
    # Convert image from RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Save image with OpenCV
    cv2.imwrite(str(output_img_path), image_bgr)
    
    # Save mask with OpenCV (most reliable for preserving values)
    cv2.imwrite(str(output_mask_path), mask)
    
    # Verify the saved mask values if in debug mode
    if debug_mask and logger:
        try:
            # Reload the mask and check values
            saved_mask = cv2.imread(str(output_mask_path), cv2.IMREAD_UNCHANGED)
            logger.info(f"Mask values after save/reload: {np.unique(saved_mask)}")
        except Exception as e:
            logger.error(f"Error verifying saved mask: {e}")


def visualize_augmentation(
    original_image: np.ndarray,
    original_mask: np.ndarray,
    augmented_image: np.ndarray,
    augmented_mask: np.ndarray,
    output_path: Path,
    index: int,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Visualize original and augmented image-mask pairs.
    
    Args:
        original_image: Original image as numpy array
        original_mask: Original mask as numpy array
        augmented_image: Augmented image as numpy array
        augmented_mask: Augmented mask as numpy array
        output_path: Directory to save visualization
        index: Index for the output filename
        logger: Optional logger for debugging
    """
    # Create visualization directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    if logger:
        logger.info(f"Creating visualization {index}, original mask values: {np.unique(original_mask)}, augmented mask values: {np.unique(augmented_mask)}")
    
    # Create a visualization with 2x2 grid:
    # [original image, augmented image]
    # [original mask, augmented mask]
    fig_height, fig_width = 12, 12
    fig = np.zeros((fig_height*2, fig_width*2, 3), dtype=np.uint8)
    
    # Resize images and masks for visualization
    def resize_for_viz(img, mask=False):
        h, w = img.shape[:2]
        scale = min(fig_height / h, fig_width / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        interpolation = cv2.INTER_NEAREST if mask else cv2.INTER_AREA
        resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        
        # If mask, colorize it for better visualization
        if mask:
            # Create a color map for visualization
            colored_mask = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            # Background (class 0) - Black
            # Cat (class 1 or 128) - Red
            colored_mask[resized == 1] = [255, 0, 0]
            colored_mask[resized == 128] = [255, 0, 0]  # Handle OpenCV loaded value
            # Dog (class 2) - Green
            colored_mask[resized == 2] = [0, 255, 0]
            # Border/Don't care (class 255) - White
            colored_mask[resized == 255] = [255, 255, 255]
            resized = colored_mask
        
        return resized
    
    # Resize and place images
    orig_img_resized = resize_for_viz(original_image)
    aug_img_resized = resize_for_viz(augmented_image)
    orig_mask_resized = resize_for_viz(original_mask, mask=True)
    aug_mask_resized = resize_for_viz(augmented_mask, mask=True)
    
    # Place images in the grid
    h1, w1 = orig_img_resized.shape[:2]
    h2, w2 = aug_img_resized.shape[:2]
    h3, w3 = orig_mask_resized.shape[:2]
    h4, w4 = aug_mask_resized.shape[:2]
    
    # Calculate offsets to center images in each quadrant
    img1_offset_y = (fig_height - h1) // 2
    img1_offset_x = (fig_width - w1) // 2
    
    img2_offset_y = (fig_height - h2) // 2
    img2_offset_x = fig_width + (fig_width - w2) // 2
    
    img3_offset_y = fig_height + (fig_height - h3) // 2
    img3_offset_x = (fig_width - w3) // 2
    
    img4_offset_y = fig_height + (fig_height - h4) // 2
    img4_offset_x = fig_width + (fig_width - w4) // 2
    
    # Place the resized images in the grid
    fig[img1_offset_y:img1_offset_y+h1, img1_offset_x:img1_offset_x+w1] = orig_img_resized
    fig[img2_offset_y:img2_offset_y+h2, img2_offset_x:img2_offset_x+w2] = aug_img_resized
    fig[img3_offset_y:img3_offset_y+h3, img3_offset_x:img3_offset_x+w3] = orig_mask_resized
    fig[img4_offset_y:img4_offset_y+h4, img4_offset_x:img4_offset_x+w4] = aug_mask_resized
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(fig, "Original Image", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(fig, "Augmented Image", (fig_width + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(fig, "Original Mask", (10, fig_height + 30), font, 1, (255, 255, 255), 2)
    cv2.putText(fig, "Augmented Mask", (fig_width + 10, fig_height + 30), font, 1, (255, 255, 255), 2)
    
    # Add color legend for masks
    legend_y = 2*fig_height - 60
    cv2.putText(fig, "Mask Legend:", (10, legend_y), font, 0.7, (255, 255, 255), 2)
    cv2.putText(fig, "Cat", (200, legend_y), font, 0.7, (0, 0, 255), 2)  # Red
    cv2.putText(fig, "Dog", (300, legend_y), font, 0.7, (0, 255, 0), 2)  # Green
    cv2.putText(fig, "Border/Don't Care", (400, legend_y), font, 0.7, (255, 255, 255), 2)  # White
    
    # Save the visualization
    cv2.imwrite(str(output_path / f"augmentation_example_{index}.jpg"), fig)
    
    if logger:
        logger.info(f"Saved visualization to {output_path / f'augmentation_example_{index}.jpg'}")


def augment_dataset(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Apply augmentation to the dataset.
    
    Args:
        args: Command-line arguments
        logger: Logger for output
    """
    # Load augmentation configuration
    config = load_config(args.config)
    logger.info(f"Loaded augmentation configuration from: {args.config}")
    
    # Set random seed for reproducibility
    seed_everything(args.seed)
    logger.info(f"Set random seed to: {args.seed}")
    
    # Set paths
    processed_dir = Path(args.processed_dir)
    train_dir = processed_dir / "Train"
    
    # Define output directories
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = train_dir / "augmented"
    
    output_img_dir = output_base / "images"
    output_mask_dir = output_base / "masks"
    vis_dir = output_base / "visualization"
    
    # Create output directories
    create_directory(output_img_dir)
    create_directory(output_mask_dir)
    if args.visualize > 0:
        create_directory(vis_dir)
    
    # Create augmentation pipelines
    cat_transform, dog_transform = create_augmentation_pipelines(config)
    logger.info("Created augmentation pipelines for cats and dogs")
    logger.info(f"Cat augmentations per image: {args.cat_augmentations}")
    logger.info(f"Dog augmentations per image: {args.dog_augmentations}")
    
    # Get train images and masks paths
    resized_img_dir = train_dir / "resized"
    resized_mask_dir = train_dir / "resized_label"  # Use resized masks instead of original
    
    # Check if resized_mask_dir exists
    if not resized_mask_dir.exists():
        logger.error(f"Resized mask directory not found: {resized_mask_dir}")
        logger.error("Please run preprocess_training_masks.py first to create resized masks")
        return
    
    train_images = list(resized_img_dir.glob("*.jpg"))
    mask_dict = {mask_path.stem: mask_path for mask_path in resized_mask_dir.glob("*.png")}
    
    logger.info(f"Found {len(train_images)} training images")
    logger.info(f"Found {len(mask_dict)} resized masks")
    
    # Counters for statistics
    cat_count = 0
    dog_count = 0
    augmented_cat_count = 0
    augmented_dog_count = 0
    visualization_count = 0
    unknown_class_count = 0
    
    # Process each image
    for img_idx, img_path in enumerate(tqdm(train_images, desc="Augmenting images")):
        try:
            # Get corresponding mask
            mask_path = mask_dict.get(img_path.stem)
            if not mask_path:
                logger.warning(f"No resized mask found for image: {img_path.name}, skipping")
                continue
            
            # Load image and mask
            try:
                image, mask = load_image_and_mask(
                    img_path, mask_path, logger, args.debug_mask
                )
            except ValueError as e:
                logger.error(f"Error loading {img_path.name}: {e}")
                continue
            
            # Determine class (cat or dog)
            class_id = get_class_from_mask(mask, img_path.name)
            
            # IMPORTANT: Set the correct number of augmentations based on class
            if class_id == 1:  # Cat
                cat_count += 1
                transform = cat_transform
                num_augmentations = int(args.cat_augmentations)
                logger.info(f"Cat image: {img_path.name}, will generate {num_augmentations} augmentations")
            elif class_id == 2:  # Dog
                dog_count += 1
                transform = dog_transform
                num_augmentations = int(args.dog_augmentations)
                logger.info(f"Dog image: {img_path.name}, will generate {num_augmentations} augmentations")
            else:
                unknown_class_count += 1
                logger.warning(f"Image {img_path.name} has unknown class (values: {np.unique(mask)}), skipping")
                continue
            
            # Generate augmentations
            augmentation_range = range(num_augmentations)
            logger.debug(f"Creating {len(augmentation_range)} augmentations for {img_path.name}")
            
            for aug_idx in augmentation_range:
                # Ensure mask is uint8 before augmentation
                mask_uint8 = mask.astype(np.uint8)
                
                # Apply augmentation with explicit seed to ensure different results
                augmentation_seed = args.seed + img_idx + aug_idx
                A.ReplayCompose.seed = augmentation_seed
                
                augmented = transform(image=image, mask=mask_uint8)
                augmented_image = augmented['image']
                augmented_mask = augmented['mask']
                
                # Generate output paths
                output_img_name = f"{img_path.stem}_aug{aug_idx+1}.jpg"
                output_mask_name = f"{img_path.stem}_aug{aug_idx+1}.png"
                
                output_img_path = output_img_dir / output_img_name
                output_mask_path = output_mask_dir / output_mask_name
                
                # Save augmented data
                save_augmented_data(
                    augmented_image, augmented_mask,
                    output_img_path, output_mask_path,
                    logger, args.debug_mask
                )
                
                # Update counters
                if class_id == 1:
                    augmented_cat_count += 1
                elif class_id == 2:
                    augmented_dog_count += 1
                
                # Visualize if requested
                if args.visualize > 0 and visualization_count < args.visualize:
                    visualize_augmentation(
                        image, mask, augmented_image, augmented_mask,
                        vis_dir, visualization_count + 1, logger
                    )
                    visualization_count += 1
        
        except Exception as e:
            logger.error(f"Error processing image {img_path.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Log final statistics with sanity checks
    cat_aug_ratio = augmented_cat_count / cat_count if cat_count > 0 else 0
    dog_aug_ratio = augmented_dog_count / dog_count if dog_count > 0 else 0
    
    logger.info("Augmentation complete!")
    logger.info(f"Original dataset: {cat_count} cats, {dog_count} dogs (total: {cat_count + dog_count})")
    logger.info(f"Unknown class count: {unknown_class_count}")
    logger.info(f"Augmented dataset: {augmented_cat_count} cat augmentations, {augmented_dog_count} dog augmentations")
    logger.info(f"Augmentation ratios - Cats: {cat_aug_ratio:.2f}x, Dogs: {dog_aug_ratio:.2f}x")
    logger.info(f"Total for training: {cat_count + dog_count + augmented_cat_count + augmented_dog_count} images")
    logger.info(f"Final class distribution: {cat_count + augmented_cat_count} cats, {dog_count + augmented_dog_count} dogs")
    
    # Create a simple report file
    report_path = output_base / "augmentation_report.txt"
    with open(report_path, 'w') as f:
        f.write("Augmentation Report\n")
        f.write("===================\n\n")
        f.write(f"Original dataset: {cat_count} cats, {dog_count} dogs (total: {cat_count + dog_count})\n")
        f.write(f"Unknown class count: {unknown_class_count}\n")
        f.write(f"Augmented dataset: {augmented_cat_count} cat augmentations, {augmented_dog_count} dog augmentations\n")
        f.write(f"Augmentation ratios - Cats: {cat_aug_ratio:.2f}x, Dogs: {dog_aug_ratio:.2f}x\n")
        f.write(f"Total for training: {cat_count + dog_count + augmented_cat_count + augmented_dog_count} images\n")
        f.write(f"Final class distribution: {cat_count + augmented_cat_count} cats, {dog_count + augmented_dog_count} dogs\n")
        f.write(f"\nAugmentation parameters are stored in: {args.config}\n")

def main() -> None:
    """Main function to augment the dataset."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    # Apply augmentation
    augment_dataset(args, logger)


if __name__ == "__main__":
    main()