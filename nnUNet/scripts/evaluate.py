#!/usr/bin/env python
"""
Script: evaluate.py

This script evaluates a trained UNet model on the test set,
calculating quantitative metrics and generating qualitative visualizations.

Example Usage:
    python scripts/evaluate.py \
        --model models/unet_pet_segmentation/best_model.pth \
        --test_dir data/processed/Test \
        --output_dir evaluation_results
"""

import os
import sys
import yaml
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
from PIL import Image

# Add project root to path to import our modules
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.models.unet import ModelConfig, UNetModel, create_transforms


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("evaluate")
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
        description="Evaluate trained UNet model on the test set"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model checkpoint"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model configuration file (optional if included in checkpoint)"
    )
    
    parser.add_argument(
        "--test_dir",
        type=str,
        default="data/processed/Test",
        help="Path to test data directory"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading"
    )
    
    parser.add_argument(
        "--visualize_n",
        type=int,
        default=10,
        help="Number of test samples to visualize"
    )
    
    return parser.parse_args()


class TestDataset(Dataset):
    """Dataset class for test evaluation."""
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Optional[A.Compose] = None
    ):
        """Initialize the dataset."""
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        
        # Store original image sizes
        self.original_sizes = {}
        for img_path in self.image_files:
            img = cv2.imread(str(img_path))
            self.original_sizes[img_path.stem] = img.shape[:2]  # (height, width)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset."""
        # Get image file path
        img_path = self.image_files[idx]
        img_stem = img_path.stem
        
        # Get corresponding mask file path
        mask_path = self.masks_dir / f"{img_stem}.png"
        
        # Load image and mask
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original image for visualization
        original_image = image.copy()
        
        # Load mask if it exists
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Create a dummy mask if not available
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Store original mask for evaluation
        original_mask = mask.copy()
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        return {
            "image": image,
            "mask": mask,
            "original_image": original_image,
            "original_mask": original_mask,
            "name": img_stem,
            "original_size": self.original_sizes[img_stem]
        }


def load_model_from_checkpoint(checkpoint_path: str, config_path: Optional[str] = None, device: str = "cuda") -> UNetModel:
    """Load model from checkpoint."""
    # Determine device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract configuration
    if "config" in checkpoint:
        # Config is included in the checkpoint
        config_dict = checkpoint["config"]
        config = ModelConfig(None)  # Create empty config
        
        # Fill config from dict
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    elif config_path:
        # Load config from file
        config = ModelConfig(config_path)
    else:
        raise ValueError("Config not found in checkpoint and no config file provided")
    
    # Create model
    model = UNetModel(config)
    
    # Load state dict
    if "model_state_dict" in checkpoint:
        model.model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume checkpoint is just the model state dict
        model.model.load_state_dict(checkpoint)
    
    model.model.to(device)
    model.model.eval()
    
    return model


def remove_padding_and_resize(padded_mask: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
    """
    Remove padding and resize mask to original dimensions.
    
    Args:
        padded_mask: Mask with padding (e.g., 512×512)
        original_size: Original size as (height, width)
        
    Returns:
        Mask with original dimensions
    """
    height, width = padded_mask.shape
    
    # Calculate scaling ratio
    if original_size[0] > original_size[1]:
        # Portrait orientation
        scale = height / original_size[0]
        new_height = height
        new_width = int(original_size[1] * scale)
    else:
        # Landscape or square orientation
        scale = width / original_size[1]
        new_width = width
        new_height = int(original_size[0] * scale)
    
    # Calculate padding offsets
    pad_y = (height - new_height) // 2
    pad_x = (width - new_width) // 2
    
    # Extract the non-padded region
    unpadded = padded_mask[pad_y:pad_y+new_height, pad_x:pad_x+new_width]
    
    # Resize to original dimensions
    original = cv2.resize(
        unpadded.astype(np.uint8),
        (original_size[1], original_size[0]),
        interpolation=cv2.INTER_NEAREST
    )
    
    return original


def calculate_metrics(predictions: List[np.ndarray], ground_truths: List[np.ndarray], num_classes: int = 3) -> Dict[str, float]:
    """
    Calculate evaluation metrics for segmentation.
    
    Args:
        predictions: List of prediction masks
        ground_truths: List of ground truth masks
        num_classes: Number of classes including background
        
    Returns:
        Dictionary with metrics
    """
    # Initialize accumulators for IoU
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    
    # Initialize accumulators for Dice
    dice_intersection = np.zeros(num_classes)
    dice_sum = np.zeros(num_classes)
    
    # Initialize accumulator for pixel accuracy
    total_pixels = 0
    correct_pixels = 0
    
    # Process each image
    for pred, gt in zip(predictions, ground_truths):
        # Ensure same shape
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Update pixel accuracy
        total_pixels += gt.size
        correct_pixels += np.sum(pred == gt)
        
        # Process each class
        for class_id in range(num_classes):
            pred_mask = (pred == class_id)
            gt_mask = (gt == class_id)
            
            # IoU
            intersection[class_id] += np.logical_and(pred_mask, gt_mask).sum()
            union[class_id] += np.logical_or(pred_mask, gt_mask).sum()
            
            # Dice
            dice_intersection[class_id] += 2.0 * np.logical_and(pred_mask, gt_mask).sum()
            dice_sum[class_id] += pred_mask.sum() + gt_mask.sum()
    
    # Calculate metrics
    iou = np.zeros(num_classes)
    dice = np.zeros(num_classes)
    
    for class_id in range(num_classes):
        if union[class_id] > 0:
            iou[class_id] = intersection[class_id] / union[class_id]
        
        if dice_sum[class_id] > 0:
            dice[class_id] = dice_intersection[class_id] / dice_sum[class_id]
    
    # Calculate pixel accuracy
    pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
    
    # Prepare metrics dictionary
    metrics = {
        "iou_background": float(iou[0]),
        "iou_cat": float(iou[1]),
        "iou_dog": float(iou[2]),
        "mean_iou": float(np.mean(iou)),
        "dice_background": float(dice[0]),
        "dice_cat": float(dice[1]),
        "dice_dog": float(dice[2]),
        "mean_dice": float(np.mean(dice)),
        "pixel_accuracy": float(pixel_accuracy)
    }
    
    return metrics


def create_visualization(
    original_image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    name: str,
    output_dir: str
) -> str:
    """
    Create visualization of segmentation results.
    
    Args:
        original_image: Original image
        ground_truth: Ground truth mask
        prediction: Predicted mask
        name: Image name
        output_dir: Output directory
        
    Returns:
        Path to saved visualization
    """
    # Create a figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define color maps for masks
    classes = ['Background', 'Cat', 'Dog']
    colors = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]  # Black, Red, Green
    cmap = ListedColormap(colors)
    
    # Plot original image
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image")
    axs[0].axis('off')
    
    # Plot ground truth
    masked_gt = np.ma.masked_where(ground_truth == 0, ground_truth)
    axs[1].imshow(original_image)
    axs[1].imshow(masked_gt, cmap=cmap, alpha=0.5, vmin=0, vmax=2)
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')
    
    # Plot prediction
    masked_pred = np.ma.masked_where(prediction == 0, prediction)
    axs[2].imshow(original_image)
    axs[2].imshow(masked_pred, cmap=cmap, alpha=0.5, vmin=0, vmax=2)
    axs[2].set_title("Prediction")
    axs[2].axis('off')
    
    # Add a legend
    patches = [plt.Rectangle((0, 0), 1, 1, fc=colors[i]) for i in range(len(classes))]
    fig.legend(patches, classes, loc='lower center', ncol=len(classes))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f"{name}_visualization.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def create_error_visualization(
    original_image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    name: str,
    output_dir: str
) -> str:
    """
    Create visualization highlighting prediction errors.
    
    Args:
        original_image: Original image
        ground_truth: Ground truth mask
        prediction: Predicted mask
        name: Image name
        output_dir: Output directory
        
    Returns:
        Path to saved visualization
    """
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original image with ground truth
    masked_gt = np.ma.masked_where(ground_truth == 0, ground_truth)
    axs[0].imshow(original_image)
    axs[0].imshow(masked_gt, cmap=ListedColormap([(0, 0, 0), (1, 0, 0), (0, 1, 0)]), alpha=0.5, vmin=0, vmax=2)
    axs[0].set_title("Ground Truth")
    axs[0].axis('off')
    
    # Create error map
    error_map = np.zeros_like(ground_truth)
    
    # True positive: prediction and ground truth agree (non-zero and match)
    true_positive = (prediction > 0) & (ground_truth > 0) & (prediction == ground_truth)
    
    # False positive: prediction sees class where there isn't one
    false_positive = (prediction > 0) & (ground_truth == 0)
    
    # False negative: prediction misses a class that's there
    false_negative = (prediction == 0) & (ground_truth > 0)
    
    # Wrong class: prediction and ground truth both have a class but disagree
    wrong_class = (prediction > 0) & (ground_truth > 0) & (prediction != ground_truth)
    
    # Assign colors to error map
    error_map[true_positive] = 1  # Green (correct)
    error_map[false_positive] = 2  # Red (false positive)
    error_map[false_negative] = 3  # Blue (false negative)
    error_map[wrong_class] = 4  # Yellow (wrong class)
    
    # Create color map for error visualization
    error_colors = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 1, 0)]  # Black, Green, Red, Blue, Yellow
    error_cmap = ListedColormap(error_colors)
    
    # Plot error map
    masked_error = np.ma.masked_where(error_map == 0, error_map)
    axs[1].imshow(original_image)
    axs[1].imshow(masked_error, cmap=error_cmap, alpha=0.5, vmin=0, vmax=4)
    axs[1].set_title("Error Analysis")
    axs[1].axis('off')
    
    # Add a legend
    error_classes = ['Background', 'Correct', 'False Positive', 'False Negative', 'Wrong Class']
    patches = [plt.Rectangle((0, 0), 1, 1, fc=error_colors[i]) for i in range(len(error_classes))]
    fig.legend(patches, error_classes, loc='lower center', ncol=len(error_classes))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f"{name}_error_analysis.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def evaluate_model(
    model: UNetModel,
    test_loader: DataLoader,
    output_dir: str,
    visualize_n: int = 10,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Evaluate the model on the test set.
    
    Args:
        model: UNetModel to evaluate
        test_loader: DataLoader for test data
        output_dir: Directory to save results
        visualize_n: Number of samples to visualize
        logger: Logger for output
        
    Returns:
        Dictionary with evaluation results
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    
    # Lists to store predictions and ground truths
    all_predictions = []
    all_ground_truths = []
    all_names = []
    
    # Process each batch
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Get data
            images = batch["image"].to(model.device)
            original_images = batch["original_image"]
            original_masks = batch["original_mask"]
            names = batch["name"]
            original_sizes = batch["original_size"]
            
            # Forward pass
            outputs = model.model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Process each sample in the batch
            for i in range(len(names)):
                name = names[i]
                pred = preds[i].cpu().numpy()
                original_image = original_images[i]
                original_mask = original_masks[i]
                original_size = (original_sizes[i][0].item(), original_sizes[i][1].item())
                
                # Resize prediction to original size
                pred_resized = remove_padding_and_resize(pred, original_size)
                
                # Save prediction
                pred_path = os.path.join(pred_dir, f"{name}_pred.png")
                cv2.imwrite(pred_path, pred_resized)
                
                # Store for metrics calculation
                all_predictions.append(pred_resized)
                all_ground_truths.append(original_mask)
                all_names.append(name)
                
                # Create visualizations for a subset of samples
                if len(all_names) <= visualize_n:
                    viz_path = create_visualization(
                        original_image, original_mask, pred_resized, name, viz_dir
                    )
                    error_path = create_error_visualization(
                        original_image, original_mask, pred_resized, name, viz_dir
                    )
                    
                    if logger:
                        logger.info(f"Created visualizations for {name}: {viz_path}, {error_path}")
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_ground_truths)
    
    # Print metrics
    if logger:
        logger.info("Evaluation Metrics:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    return {
        "metrics": metrics,
        "predictions": all_predictions,
        "ground_truths": all_ground_truths,
        "names": all_names
    }


def main() -> None:
    """Main function for model evaluation."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    log_file = os.path.join(args.output_dir, "evaluation.log")
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(log_file)
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model = load_model_from_checkpoint(args.model, args.config)
    
    # Create test dataset
    test_imgs_dir = os.path.join(args.test_dir, "resized")
    test_masks_dir = os.path.join(args.test_dir, "processed_labels")
    
    # Create transforms
    transform = A.Compose([
        A.Resize(height=model.config.patch_size[0], width=model.config.patch_size[1]),
        A.Normalize(),
        ToTensorV2()
    ])
    
    # Create dataset
    test_dataset = TestDataset(
        images_dir=test_imgs_dir,
        masks_dir=test_masks_dir,
        transform=transform
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    logger.info(f"Evaluating model on {len(test_dataset)} test samples")
    
    # Evaluate model
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        output_dir=args.output_dir,
        visualize_n=args.visualize_n,
        logger=logger
    )
    
    # Compare to baseline
    baseline_iou = 0.33
    achieved_iou = results["metrics"]["mean_iou"]
    
    if achieved_iou > baseline_iou:
        logger.info(f"Model outperforms baseline! Mean IoU: {achieved_iou:.4f} (Baseline: {baseline_iou:.4f})")
    else:
        logger.info(f"Model does not outperform baseline. Mean IoU: {achieved_iou:.4f} (Baseline: {baseline_iou:.4f})")
    
    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()