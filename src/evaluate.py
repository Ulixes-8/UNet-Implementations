#!/usr/bin/env python
"""
Script: evaluate.py

This script evaluates the trained UNet model on the test set of the Oxford-IIIT Pet Dataset.
It calculates performance metrics using accumulated values across the test set for accurate
per-class evaluation, and generates visualizations for qualitative analysis.

Example usage:
    python src/evaluate.py --model_path models/unet_pet_segmentation/best_model.pth \
                          --data_dir data/processed \
                          --output_dir evaluation_results
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import local modules
from src.models.unet import UNet
from src.utils.metrics import evaluate_masks
from src.utils.visualize import (
    visualize_image_triplets,
    visualize_error_analysis,
    visualize_metrics_bar_chart,
    visualize_confusion_matrix
)


class PetTestDataset(Dataset):
    """
    Dataset class for testing UNet on the Oxford-IIIT Pet test set.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        transform=None
    ):
        """
        Initialize the test dataset.
        
        Args:
            data_dir: Path to the processed data directory
            transform: Optional transform to apply to images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Set paths
        self.image_dir = self.data_dir / "Test" / "resized"
        self.mask_dir = self.data_dir / "Test" / "processed_labels"
        self.original_image_dir = self.data_dir / "Test" / "color"
        
        # Get test image files
        self.image_files = sorted(list(self.image_dir.glob("*.jpg")))
        
        # Verify images have corresponding masks
        valid_files = []
        for img_path in self.image_files:
            mask_path = self.mask_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                valid_files.append(img_path)
        
        self.image_files = valid_files
    
    def __len__(self) -> int:
        """Return the number of test samples."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a test sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with image, mask, image path, mask path, and original size
        """
        # Get paths
        img_path = self.image_files[idx]
        mask_path = self.mask_dir / f"{img_path.stem}.png"
        original_img_path = self.original_image_dir / f"{img_path.stem}.jpg"
        
        # Load images
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Get original size
        if original_img_path.exists():
            original_size = cv2.imread(str(original_img_path)).shape[:2]
        else:
            # If original image not available, use mask size
            original_size = mask.shape
        
        # Store original image and mask
        original_image = image.copy()
        original_mask = mask.copy()
        
        # Apply transforms if provided
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        return {
            "image": image,
            "mask": mask,
            "original_image": original_image,
            "original_mask": original_mask,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "original_size": original_size,
            "filename": img_path.stem
        }


def create_transforms():
    """
    Create transforms for test images.
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        transform = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])
        return transform
    except ImportError:
        print("Albumentations not installed. Using simple normalization.")
        return None


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Optional path to log file
        
    Returns:
        Configured logger
    """
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
        description="Evaluate UNet model on the test set"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Path to the processed data directory"
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
        "--num_vis_samples",
        type=int,
        default=5,
        help="Number of samples to visualize"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Load trained UNet model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load the model on
        
    Returns:
        Loaded UNet model
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    if "model_state_dict" in checkpoint:
        # Newer checkpoint format
        if "epoch" in checkpoint:
            print(f"Loading model from epoch {checkpoint['epoch']}")
        
        # Try to get model configuration if available
        if "config" in checkpoint:
            config = checkpoint["config"]
            model = UNet(
                in_channels=config.get("in_channels", 3),
                num_classes=config.get("num_classes", 3),
                n_stages=config.get("n_stages", 8),
                features_per_stage=config.get("features_per_stage", [32, 64, 128, 256, 512, 512, 512, 512]),
                kernel_sizes=config.get("kernel_sizes", [[3, 3]] * 8),
                strides=config.get("strides", [[1, 1]] + [[2, 2]] * 7),
                n_conv_per_stage=config.get("n_conv_per_stage", [2] * 8),
                n_conv_per_stage_decoder=config.get("n_conv_per_stage_decoder", [2] * 7),
                conv_bias=config.get("conv_bias", True),
                norm_op=nn.InstanceNorm2d,
                norm_op_kwargs=config.get("norm_op_kwargs", {"eps": 1e-5, "affine": True}),
                nonlin=nn.LeakyReLU,
                nonlin_kwargs=config.get("nonlin_kwargs", {"inplace": True}),
                deep_supervision=config.get("deep_supervision", True)
            )
        else:
            # Use default architecture
            model = UNet(
                in_channels=3,
                num_classes=3,
                deep_supervision=True
            )
        
        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume checkpoint is just the model state dict
        model = UNet(
            in_channels=3,
            num_classes=3,
            deep_supervision=True
        )
        model.load_state_dict(checkpoint)
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    return model


def resize_mask_to_original(
    mask: np.ndarray,
    original_size: Tuple[int, int],
    interpolation: int = cv2.INTER_NEAREST
) -> np.ndarray:
    """
    Resize a mask to its original dimensions.
    
    Args:
        mask: Mask to resize
        original_size: Original size as (height, width)
        interpolation: Interpolation method
        
    Returns:
        Resized mask
    """
    # Check if resizing is necessary
    if mask.shape[0] == original_size[0] and mask.shape[1] == original_size[1]:
        return mask
    
    # Resize mask to original dimensions
    resized_mask = cv2.resize(
        mask.astype(np.uint8),
        (original_size[1], original_size[0]),  # cv2.resize expects (width, height)
        interpolation=interpolation
    )
    
    return resized_mask


def predict_test_set(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: str,
    logger: logging.Logger
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Predict on the test set and save results.
    
    Args:
        model: UNet model
        dataloader: Test data loader
        device: Device to run inference on
        output_dir: Directory to save prediction masks
        logger: Logger for output
        
    Returns:
        Tuple of (predictions, ground_truths, filenames)
    """
    # Create output directory for predictions
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    
    # Lists to store results
    predictions = []
    ground_truths = []
    filenames = []
    
    # Run inference
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move data to device
            images = batch["image"].to(device)
            original_masks = batch["original_mask"]
            original_sizes = batch["original_size"]
            batch_filenames = batch["filename"]
            
            # Forward pass
            outputs = model(images)
            
            # If using deep supervision, take the first output (highest resolution)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            
            # Get predicted class (argmax)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Process each prediction in the batch
            for i in range(len(batch_filenames)):
                pred_mask = preds[i]
                gt_mask = original_masks[i].numpy()
                filename = batch_filenames[i]
                original_size = (original_sizes[i][0].item(), original_sizes[i][1].item())
                
                # Resize prediction to original size
                pred_mask_resized = resize_mask_to_original(pred_mask, original_size)
                
                # Save prediction mask
                pred_path = os.path.join(pred_dir, f"{filename}.png")
                cv2.imwrite(pred_path, pred_mask_resized)
                
                # Store for evaluation
                predictions.append(pred_mask_resized)
                ground_truths.append(gt_mask)
                filenames.append(filename)
    
    logger.info(f"Saved prediction masks to {pred_dir}")
    return predictions, ground_truths, filenames


def analyze_results(
    predictions: List[np.ndarray],
    ground_truths: List[np.ndarray],
    output_dir: str,
    logger: logging.Logger
) -> Dict:
    """
    Analyze prediction results and calculate metrics.
    
    Args:
        predictions: List of prediction masks
        ground_truths: List of ground truth masks
        output_dir: Directory to save analysis results
        logger: Logger for output
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Set class names
    class_names = ["background", "cat", "dog"]
    
    # Evaluate masks
    metrics = evaluate_masks(predictions, ground_truths, num_classes=3, class_names=class_names)
    
    # Print metrics
    logger.info("Evaluation Metrics:")
    logger.info(f"  Mean IoU: {metrics['mean_iou']:.4f}")
    logger.info(f"  Mean Dice: {metrics['mean_dice']:.4f}")
    logger.info(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    
    for class_name in class_names:
        logger.info(f"  {class_name.capitalize()}:")
        logger.info(f"    IoU: {metrics['class_metrics'][class_name]['iou']:.4f}")
        logger.info(f"    Dice: {metrics['class_metrics'][class_name]['dice']:.4f}")
    
    # Check if model outperforms baseline
    baseline_iou = 0.33  # As specified in the requirements
    if metrics["mean_iou"] > baseline_iou:
        logger.info(f"Model outperforms baseline! Mean IoU: {metrics['mean_iou']:.4f} vs Baseline: {baseline_iou:.4f}")
    else:
        logger.info(f"Model does not outperform baseline. Mean IoU: {metrics['mean_iou']:.4f} vs Baseline: {baseline_iou:.4f}")
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        # Convert numpy values to Python types for serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if key == "confusion_matrix":
                serializable_metrics[key] = value.tolist()
            elif key == "class_metrics":
                serializable_metrics[key] = {}
                for class_name, class_metrics in value.items():
                    serializable_metrics[key][class_name] = {}
                    for metric_name, metric_value in class_metrics.items():
                        serializable_metrics[key][class_name][metric_name] = float(metric_value)
            else:
                serializable_metrics[key] = float(value)
        
        json.dump(serializable_metrics, f, indent=4)
    
    logger.info(f"Saved metrics to {metrics_file}")
    
    return metrics


def create_visualizations(
    predictions: List[np.ndarray],
    ground_truths: List[np.ndarray],
    filenames: List[str],
    data_dir: str,
    output_dir: str,
    metrics: Dict,
    num_samples: int,
    seed: int,
    logger: logging.Logger
) -> None:
    """
    Create visualizations for qualitative analysis.
    
    Args:
        predictions: List of prediction masks
        ground_truths: List of ground truth masks
        filenames: List of filenames
        data_dir: Path to data directory
        output_dir: Directory to save visualizations
        metrics: Dictionary with evaluation metrics
        num_samples: Number of samples to visualize
        seed: Random seed for reproducibility
        logger: Logger for output
    """
    # Create visualizations directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Paths for visualization
    pred_dir = os.path.join(output_dir, "predictions")
    gt_dir = os.path.join(data_dir, "Test", "processed_labels")
    img_dir = os.path.join(data_dir, "Test", "resized")
    
    # Create sample visualizations
    logger.info(f"Creating sample visualizations (n={num_samples})...")
    
    # Get mask filenames
    mask_filenames = [f"{filename}.png" for filename in filenames]
    
    # Create image triplet visualizations
    visualize_image_triplets(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        img_dir=img_dir,
        filenames=mask_filenames,
        output_dir=vis_dir,
        num_samples=num_samples,
        seed=seed
    )
    
    # Create error analysis visualizations
    visualize_error_analysis(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        img_dir=img_dir,
        filenames=mask_filenames,
        output_dir=vis_dir,
        num_samples=num_samples,
        seed=seed
    )
    
    # Create metrics visualization
    visualize_metrics_bar_chart(
        metrics=metrics,
        output_path=os.path.join(vis_dir, "metrics_chart.png")
    )
    
    # Create confusion matrix visualization
    visualize_confusion_matrix(
        confusion_matrix=metrics["confusion_matrix"],
        class_names=["background", "cat", "dog"],
        output_path=os.path.join(vis_dir, "confusion_matrix.png")
    )
    
    logger.info(f"Saved visualizations to {vis_dir}")


def generate_report(
    metrics: Dict,
    output_dir: str,
    logger: logging.Logger
) -> None:
    """
    Generate a comprehensive evaluation report.
    
    Args:
        metrics: Dictionary with evaluation metrics
        output_dir: Directory to save the report
        logger: Logger for output
    """
    report_path = os.path.join(output_dir, "evaluation_report.md")
    
    with open(report_path, "w") as f:
        # Title
        f.write("# UNet Semantic Segmentation Evaluation Report\n\n")
        
        # Date and time
        f.write(f"**Date:** {time.strftime('%Y-%m-%d')}\n\n")
        
        # Overview
        f.write("## Overview\n\n")
        f.write("This report presents the evaluation results of the UNet model for semantic segmentation ")
        f.write("on the Oxford-IIIT Pet Dataset test set. The model segments images into three classes: ")
        f.write("background, cat, and dog.\n\n")
        
        # Summary metrics
        f.write("## Summary Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Mean IoU | {metrics['mean_iou']:.4f} |\n")
        f.write(f"| Mean Dice | {metrics['mean_dice']:.4f} |\n")
        f.write(f"| Pixel Accuracy | {metrics['pixel_accuracy']:.4f} |\n")
        
        # Baseline comparison
        baseline_iou = 0.33
        f.write("\n### Baseline Comparison\n\n")
        if metrics["mean_iou"] > baseline_iou:
            f.write(f"✅ **Model outperforms baseline!** Mean IoU: {metrics['mean_iou']:.4f} vs Baseline: {baseline_iou:.4f}\n\n")
            improvement = (metrics["mean_iou"] - baseline_iou) / baseline_iou * 100
            f.write(f"This represents a **{improvement:.2f}%** improvement over the baseline.\n\n")
        else:
            f.write(f"❌ **Model does not outperform baseline.** Mean IoU: {metrics['mean_iou']:.4f} vs Baseline: {baseline_iou:.4f}\n\n")
            gap = (baseline_iou - metrics["mean_iou"]) / baseline_iou * 100
            f.write(f"This represents a **{gap:.2f}%** gap below the baseline.\n\n")
        
        # Per-class metrics
        f.write("## Per-Class Metrics\n\n")
        f.write("| Class | IoU | Dice |\n")
        f.write("|-------|-----|------|\n")
        
        for class_name in ["background", "cat", "dog"]:
            iou = metrics["class_metrics"][class_name]["iou"]
            dice = metrics["class_metrics"][class_name]["dice"]
            f.write(f"| {class_name.capitalize()} | {iou:.4f} | {dice:.4f} |\n")
        
        # Analysis
        f.write("\n## Analysis\n\n")
        
        # Class performance analysis
        f.write("### Class Performance Analysis\n\n")
        
        # Find best and worst performing classes
        class_ious = [(name, metrics["class_metrics"][name]["iou"]) for name in ["background", "cat", "dog"]]
        best_class = max(class_ious, key=lambda x: x[1])
        worst_class = min(class_ious, key=lambda x: x[1])
        
        f.write(f"- **Best performing class:** {best_class[0].capitalize()} (IoU: {best_class[1]:.4f})\n")
        f.write(f"- **Worst performing class:** {worst_class[0].capitalize()} (IoU: {worst_class[1]:.4f})\n\n")
        
        # Analyze class imbalance if present
        cat_iou = metrics["class_metrics"]["cat"]["iou"]
        dog_iou = metrics["class_metrics"]["dog"]["iou"]
        class_diff = abs(cat_iou - dog_iou)
        
        if class_diff > 0.05:
            f.write("**Class Imbalance Analysis:**\n\n")
            if cat_iou > dog_iou:
                f.write(f"There is a significant performance gap between cats and dogs (difference: {class_diff:.4f}). ")
                f.write("The model performs better on cats, which might indicate:\n\n")
            else:
                f.write(f"There is a significant performance gap between dogs and cats (difference: {class_diff:.4f}). ")
                f.write("The model performs better on dogs, which might indicate:\n\n")
            
            f.write("1. Class distribution imbalance in the training data\n")
            f.write("2. Different levels of difficulty in segmenting the two classes\n")
            f.write("3. More visual variety within the worse-performing class\n\n")
        
        # General observations
        f.write("### General Observations\n\n")
        
        if metrics["mean_iou"] > 0.7:
            f.write("- The model achieves excellent segmentation performance overall.\n")
        elif metrics["mean_iou"] > 0.5:
            f.write("- The model achieves good segmentation performance, but there is room for improvement.\n")
        else:
            f.write("- The model's segmentation performance is moderate, suggesting that further improvements are needed.\n")
        
        if metrics["mean_iou"] - metrics["mean_dice"] / 2 > 0.1:
            f.write("- There is a notable difference between IoU and Dice scores, suggesting that the model produces fragmented segmentations.\n")
        
        # Error analysis
        f.write("\n### Error Analysis\n\n")
        
        # Calculate error rates from confusion matrix
        confusion = np.array(metrics["confusion_matrix"])
        total = confusion.sum()
        correct = np.diag(confusion).sum()
        error_rate = 1 - (correct / total)
        
        f.write(f"- Overall error rate: {error_rate:.2%}\n")
        
        # Calculate misclassification patterns
        bg_as_cat = confusion[0, 1] / confusion[0, :].sum() if confusion[0, :].sum() > 0 else 0
        bg_as_dog = confusion[0, 2] / confusion[0, :].sum() if confusion[0, :].sum() > 0 else 0
        cat_as_bg = confusion[1, 0] / confusion[1, :].sum() if confusion[1, :].sum() > 0 else 0
        cat_as_dog = confusion[1, 2] / confusion[1, :].sum() if confusion[1, :].sum() > 0 else 0
        dog_as_bg = confusion[2, 0] / confusion[2, :].sum() if confusion[2, :].sum() > 0 else 0
        dog_as_cat = confusion[2, 1] / confusion[2, :].sum() if confusion[2, :].sum() > 0 else 0
        
        f.write("- Common misclassification patterns:\n")
        f.write(f"  - Background misclassified as cat: {bg_as_cat:.2%}\n")
        f.write(f"  - Background misclassified as dog: {bg_as_dog:.2%}\n")
        f.write(f"  - Cat misclassified as background: {cat_as_bg:.2%}\n")
        f.write(f"  - Cat misclassified as dog: {cat_as_dog:.2%}\n")
        f.write(f"  - Dog misclassified as background: {dog_as_bg:.2%}\n")
        f.write(f"  - Dog misclassified as cat: {dog_as_cat:.2%}\n\n")
        
        # Most common error analysis
        error_types = [
            ("Background as cat", bg_as_cat),
            ("Background as dog", bg_as_dog),
            ("Cat as background", cat_as_bg),
            ("Cat as dog", cat_as_dog),
            ("Dog as background", dog_as_bg),
            ("Dog as cat", dog_as_cat)
        ]
        most_common_error = max(error_types, key=lambda x: x[1])
        
        f.write(f"- **Most common error:** {most_common_error[0]} ({most_common_error[1]:.2%})\n\n")
        
        # Draw conclusions
        if cat_as_dog > 0.1 or dog_as_cat > 0.1:
            f.write("- The model sometimes confuses cats and dogs, suggesting that more distinctive features need to be learned.\n")
        
        if cat_as_bg > 0.1 or dog_as_bg > 0.1:
            f.write("- There are significant false negatives (animals missed entirely), which could be improved with better boundary detection.\n")
        
        if bg_as_cat > 0.1 or bg_as_dog > 0.1:
            f.write("- There are significant false positives (background classified as animals), which might indicate oversegmentation.\n")
        
        # Conclusions and recommendations
        f.write("\n## Conclusions and Recommendations\n\n")
        
        if metrics["mean_iou"] > baseline_iou:
            f.write("- ✅ The model successfully outperforms the baseline, demonstrating effective learning of pet segmentation.\n")
        else:
            f.write("- ❌ The model does not meet the baseline requirement, suggesting that further improvements are needed.\n")
        
        f.write("\n### Potential Improvements\n\n")
        
        f.write("1. **Data Augmentation:** ")
        if class_diff > 0.05:
            f.write(f"Enhance augmentation for the {worst_class[0]} class to improve balance.\n")
        else:
            f.write("Increase diversity in the training data through more aggressive augmentations.\n")
        
        f.write("2. **Architecture Adjustments:** Consider deeper or wider networks, attention mechanisms, or ensemble methods.\n")
        
        f.write("3. **Loss Function:** ")
        if cat_as_bg > 0.1 or dog_as_bg > 0.1:
            f.write("Use a loss function that penalizes false negatives more heavily.\n")
        elif bg_as_cat > 0.1 or bg_as_dog > 0.1:
            f.write("Use a loss function that penalizes false positives more heavily.\n")
        else:
            f.write("Experiment with different loss functions tailored to segmentation tasks.\n")
        
        f.write("4. **Postprocessing:** Apply morphological operations or conditional random fields to refine boundaries.\n")
        
        f.write("5. **Training Strategy:** Extend training time, adjust learning rate scheduling, or implement curriculum learning.\n")
        
        # Closing
        f.write("\n## Visualizations\n\n")
        f.write("Please refer to the `visualizations` directory for qualitative examples of the model's performance, ")
        f.write("including sample predictions, error maps, and performance charts.\n")
    
    logger.info(f"Generated evaluation report: {report_path}")


def main() -> None:
    """Main function for evaluation."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = output_dir / "evaluation.log"
    logger = setup_logging(log_file)
    
    # Log configuration
    logger.info("Evaluation Configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, device)
    
    # Create transforms
    transform = create_transforms()
    
    # Create dataset
    logger.info(f"Creating test dataset from {args.data_dir}")
    test_dataset = PetTestDataset(args.data_dir, transform=transform)
    logger.info(f"Test dataset size: {len(test_dataset)} images")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Predict on test set
    logger.info("Running inference on test set...")
    predictions, ground_truths, filenames = predict_test_set(
        model, test_loader, device, output_dir, logger
    )
    
    # Analyze results
    logger.info("Analyzing results...")
    metrics = analyze_results(predictions, ground_truths, output_dir, logger)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    create_visualizations(
        predictions=predictions,
        ground_truths=ground_truths,
        filenames=filenames,
        data_dir=args.data_dir,
        output_dir=output_dir,
        metrics=metrics,
        num_samples=args.num_vis_samples,
        seed=args.seed,
        logger=logger
    )
    
    # Generate report
    logger.info("Generating evaluation report...")
    generate_report(metrics, output_dir, logger)
    
    logger.info(f"Evaluation complete. All results saved to {output_dir}")


if __name__ == "__main__":
    main()