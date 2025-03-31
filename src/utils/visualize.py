"""
Visualization utilities for semantic segmentation evaluation.

This module provides functions for visualizing segmentation results, including
colorization of masks, side-by-side comparison of predictions and ground truth,
and error analysis visualizations.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from PIL import Image


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert a segmentation mask with values {0,1,2} into an RGB color image.
    - 0 → black (0, 0, 0)
    - 1 → red (255, 0, 0)
    - 2 → green (0, 255, 0)
    
    Args:
        mask (np.ndarray): 2D array of shape (H, W) with class values.
        
    Returns:
        np.ndarray: RGB image of shape (H, W, 3)
    
    Example:
        >>> color_mask = colorize_mask(np.array([[0, 1], [2, 1]]))
    """
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb_mask[mask == 1] = [255, 0, 0]  # Red for class 1 (cat)
    rgb_mask[mask == 2] = [0, 255, 0]  # Green for class 2 (dog)
    return rgb_mask


def visualize_image_triplets(
    pred_dir: str,
    gt_dir: str,
    img_dir: str,
    filenames: List[str],
    output_dir: Optional[str] = None,
    num_samples: int = 5,
    seed: int = 42
) -> None:
    """
    Visualize the original image, ground truth mask, and predicted mask side-by-side.
    
    Args:
        pred_dir (str): Path to prediction masks.
        gt_dir (str): Path to ground truth masks.
        img_dir (str): Path to original images.
        filenames (List[str]): List of prediction mask filenames.
        output_dir (str, optional): Directory to save visualizations. If None, will show interactively.
        num_samples (int): Number of samples to visualize.
        seed (int): Random seed for reproducibility.
        
    Returns:
        None
    
    Example:
        >>> visualize_image_triplets(PRED_MASK_DIR, GT_MASK_DIR, IMAGE_DIR, pred_mask_files)
    """
    random.seed(seed)
    sample_files = random.sample(filenames, min(num_samples, len(filenames)))
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    for fname in sample_files:
        basename = fname.replace(".png", "")
        pred_path = os.path.join(pred_dir, fname)
        gt_path = os.path.join(gt_dir, fname)
        img_path = os.path.join(img_dir, f"{basename}.jpg")  # Assuming .jpg extension
        
        # Check if files exist
        if not os.path.exists(pred_path) or not os.path.exists(gt_path) or not os.path.exists(img_path):
            # Try alternative image extension
            img_path = os.path.join(img_dir, f"{basename}.png")
            if not os.path.exists(img_path):
                print(f"Warning: Skipping {basename} - one or more files missing")
                continue
        
        # Load images
        pred_mask = np.array(Image.open(pred_path))
        gt_mask = np.array(Image.open(gt_path))
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Colorize masks
        colored_pred = colorize_mask(pred_mask)
        colored_gt = colorize_mask(gt_mask)
        
        # Create subplot
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(image)
        axs[0].set_title(f"Original Image\n{basename}")
        axs[0].axis("off")
        
        axs[1].imshow(colored_gt)
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")
        
        axs[2].imshow(colored_pred)
        axs[2].set_title("Prediction")
        axs[2].axis("off")
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{basename}_comparison.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def create_error_visualization(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    image: np.ndarray
) -> np.ndarray:
    """
    Create visualization highlighting prediction errors.
    
    Args:
        pred_mask: Predicted segmentation mask
        gt_mask: Ground truth segmentation mask
        image: Original RGB image
        
    Returns:
        RGB visualization image
    """
    # Create empty RGB image
    error_map = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    
    # True positive: prediction and ground truth agree (non-zero and match)
    true_positive = (pred_mask > 0) & (gt_mask > 0) & (pred_mask == gt_mask)
    
    # False positive: model predicted a class when there isn't one
    false_positive = (pred_mask > 0) & (gt_mask == 0)
    
    # False negative: model missed a class that's there
    false_negative = (pred_mask == 0) & (gt_mask > 0)
    
    # Wrong class: both prediction and ground truth have a class but disagree
    wrong_class = (pred_mask > 0) & (gt_mask > 0) & (pred_mask != gt_mask)
    
    # Assign colors
    error_map[true_positive] = [0, 255, 0]    # Green (correct)
    error_map[false_positive] = [255, 0, 0]   # Red (false positive)
    error_map[false_negative] = [0, 0, 255]   # Blue (false negative)
    error_map[wrong_class] = [255, 255, 0]    # Yellow (wrong class)
    
    # Create a blended image with the original
    alpha = 0.5
    blended = (image * (1 - alpha) + error_map * alpha).astype(np.uint8)
    
    return blended


def visualize_error_analysis(
    pred_dir: str,
    gt_dir: str,
    img_dir: str,
    filenames: List[str],
    output_dir: Optional[str] = None,
    num_samples: int = 5,
    seed: int = 42
) -> None:
    """
    Visualize error analysis for segmentation predictions.
    
    Args:
        pred_dir: Path to prediction masks
        gt_dir: Path to ground truth masks
        img_dir: Path to original images
        filenames: List of prediction mask filenames
        output_dir: Directory to save visualizations. If None, will show interactively
        num_samples: Number of samples to visualize
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    sample_files = random.sample(filenames, min(num_samples, len(filenames)))
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    for fname in sample_files:
        basename = fname.replace(".png", "")
        pred_path = os.path.join(pred_dir, fname)
        gt_path = os.path.join(gt_dir, fname)
        img_path = os.path.join(img_dir, f"{basename}.jpg")  # Assuming .jpg extension
        
        # Check if files exist
        if not os.path.exists(pred_path) or not os.path.exists(gt_path) or not os.path.exists(img_path):
            # Try alternative image extension
            img_path = os.path.join(img_dir, f"{basename}.png")
            if not os.path.exists(img_path):
                print(f"Warning: Skipping {basename} - one or more files missing")
                continue
        
        # Load images
        pred_mask = np.array(Image.open(pred_path))
        gt_mask = np.array(Image.open(gt_path))
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Create error visualization
        error_viz = create_error_visualization(pred_mask, gt_mask, image)
        
        # Create subplot
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].imshow(image)
        axs[0].set_title(f"Original Image\n{basename}")
        axs[0].axis("off")
        
        axs[1].imshow(error_viz)
        axs[1].set_title("Error Analysis")
        axs[1].axis("off")
        
        # Add legend
        legend_items = [
            plt.Rectangle((0, 0), 1, 1, color='g', label='Correct'),
            plt.Rectangle((0, 0), 1, 1, color='r', label='False Positive'),
            plt.Rectangle((0, 0), 1, 1, color='b', label='False Negative'),
            plt.Rectangle((0, 0), 1, 1, color='y', label='Wrong Class')
        ]
        axs[1].legend(
            handles=legend_items,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.05),
            ncol=4
        )
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{basename}_error_analysis.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def create_class_visualization(mask: np.ndarray, class_id: int) -> np.ndarray:
    """
    Create binary visualization for a specific class.
    
    Args:
        mask: Segmentation mask
        class_id: Class ID to visualize
        
    Returns:
        Binary mask for the specified class
    """
    binary_mask = (mask == class_id).astype(np.uint8) * 255
    return binary_mask


def visualize_metrics_bar_chart(metrics: Dict, output_path: Optional[str] = None) -> None:
    """
    Create a bar chart visualization of performance metrics.
    
    Args:
        metrics: Dictionary with evaluation metrics
        output_path: Path to save the visualization. If None, will show interactively
    """
    class_names = list(metrics["class_metrics"].keys())
    iou_values = [metrics["class_metrics"][cls]["iou"] for cls in class_names]
    dice_values = [metrics["class_metrics"][cls]["dice"] for cls in class_names]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, iou_values, width, label='IoU')
    rects2 = ax.bar(x + width/2, dice_values, width, label='Dice')
    
    # Add mean values
    ax.axhline(y=metrics["mean_iou"], color='b', linestyle='--', alpha=0.7)
    ax.axhline(y=metrics["mean_dice"], color='orange', linestyle='--', alpha=0.7)
    
    # Add baseline
    baseline = 0.33  # As specified in the requirements
    ax.axhline(y=baseline, color='r', linestyle=':', label=f'Baseline IoU: {baseline:.2f}')
    
    # Add some text for labels, title and custom x-axis tick labels
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    
    # Add value annotations
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    # Add text for mean values
    ax.text(len(class_names) - 0.7, metrics["mean_iou"] + 0.01, 
            f'Mean IoU: {metrics["mean_iou"]:.3f}', 
            color='b', ha='center')
    
    ax.text(len(class_names) - 0.7, metrics["mean_dice"] + 0.01, 
            f'Mean Dice: {metrics["mean_dice"]:.3f}', 
            color='orange', ha='center')
    
    # Set y-axis limit
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def visualize_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    output_path: Optional[str] = None
) -> None:
    """
    Visualize confusion matrix for segmentation results.
    
    Args:
        confusion_matrix: Confusion matrix
        class_names: List of class names
        output_path: Path to save the visualization. If None, will show interactively
    """
    # Normalize confusion matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        norm_cm = np.nan_to_num(norm_cm)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(norm_cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Show all ticks
    ax.set(xticks=np.arange(norm_cm.shape[1]),
           yticks=np.arange(norm_cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    fmt = '.2f'
    thresh = norm_cm.max() / 2.
    for i in range(norm_cm.shape[0]):
        for j in range(norm_cm.shape[1]):
            ax.text(j, i, format(norm_cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if norm_cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()