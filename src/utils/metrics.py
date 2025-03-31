"""
Metrics utility functions for semantic segmentation evaluation.

This module provides functions for calculating evaluation metrics for semantic segmentation,
including IoU (Intersection over Union) and Dice coefficient, with proper handling of
per-class accumulation across an entire dataset.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional


def calculate_confusion_matrix(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    num_classes: int = 3
) -> np.ndarray:
    """
    Calculate confusion matrix for a single image.
    
    Args:
        pred_mask: Predicted segmentation mask
        gt_mask: Ground truth segmentation mask
        num_classes: Number of classes
        
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    mask = (gt_mask >= 0) & (gt_mask < num_classes)
    confusion = np.bincount(
        num_classes * gt_mask[mask].astype(int) + pred_mask[mask],
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    
    return confusion


def calculate_iou_per_class(
    confusion_matrix: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Calculate IoU for each class from a confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix of shape (num_classes, num_classes)
        
    Returns:
        Tuple of (per-class IoU array, mean IoU)
    """
    # True positives are on the diagonal
    true_pos = np.diag(confusion_matrix)
    
    # Sum over rows gives total ground truth pixels for each class
    gt_pixels = np.sum(confusion_matrix, axis=1)
    
    # Sum over columns gives total predicted pixels for each class
    pred_pixels = np.sum(confusion_matrix, axis=0)
    
    # IoU = true_pos / (gt_pixels + pred_pixels - true_pos)
    denominator = gt_pixels + pred_pixels - true_pos
    
    # Handle division by zero
    iou = np.zeros_like(true_pos, dtype=np.float32)
    mask = denominator > 0
    iou[mask] = true_pos[mask] / denominator[mask]
    
    # Calculate mean IoU
    valid_classes = mask.sum()
    mean_iou = iou[mask].sum() / valid_classes if valid_classes > 0 else 0
    
    return iou, mean_iou


def calculate_dice_per_class(
    confusion_matrix: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Calculate Dice coefficient for each class from a confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix of shape (num_classes, num_classes)
        
    Returns:
        Tuple of (per-class Dice array, mean Dice)
    """
    # True positives are on the diagonal
    true_pos = np.diag(confusion_matrix)
    
    # Sum over rows gives total ground truth pixels for each class
    gt_pixels = np.sum(confusion_matrix, axis=1)
    
    # Sum over columns gives total predicted pixels for each class
    pred_pixels = np.sum(confusion_matrix, axis=0)
    
    # Dice = (2 * true_pos) / (gt_pixels + pred_pixels)
    denominator = gt_pixels + pred_pixels
    
    # Handle division by zero
    dice = np.zeros_like(true_pos, dtype=np.float32)
    mask = denominator > 0
    dice[mask] = (2 * true_pos[mask]) / denominator[mask]
    
    # Calculate mean Dice
    valid_classes = mask.sum()
    mean_dice = dice[mask].sum() / valid_classes if valid_classes > 0 else 0
    
    return dice, mean_dice


def calculate_pixel_accuracy(
    confusion_matrix: np.ndarray
) -> float:
    """
    Calculate pixel accuracy from a confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix of shape (num_classes, num_classes)
        
    Returns:
        Pixel accuracy as a float
    """
    # True positives are on the diagonal
    true_pos = np.diag(confusion_matrix).sum()
    
    # Total pixels are the sum of the entire matrix
    total_pixels = confusion_matrix.sum()
    
    # Pixel accuracy = true_pos / total_pixels
    accuracy = true_pos / total_pixels if total_pixels > 0 else 0
    
    return accuracy


def evaluate_masks(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    num_classes: int = 3,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Evaluate prediction masks against ground truth using accumulated metrics.
    
    Args:
        pred_masks: List of predicted segmentation masks
        gt_masks: List of ground truth segmentation masks
        num_classes: Number of classes
        class_names: Optional list of class names
        
    Returns:
        Dictionary with evaluation metrics
    """
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # Accumulate confusion matrix across all images
    for pred, gt in zip(pred_masks, gt_masks):
        confusion_matrix += calculate_confusion_matrix(pred, gt, num_classes)
    
    # Calculate IoU
    iou_per_class, mean_iou = calculate_iou_per_class(confusion_matrix)
    
    # Calculate Dice
    dice_per_class, mean_dice = calculate_dice_per_class(confusion_matrix)
    
    # Calculate pixel accuracy
    pixel_accuracy = calculate_pixel_accuracy(confusion_matrix)
    
    # Create metrics dictionary
    metrics = {
        "confusion_matrix": confusion_matrix,
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "class_metrics": {}
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(class_names):
        metrics["class_metrics"][class_name] = {
            "iou": iou_per_class[i],
            "dice": dice_per_class[i]
        }
    
    return metrics