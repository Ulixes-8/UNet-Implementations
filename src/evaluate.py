#!/usr/bin/env python
"""
Script: evaluate.py

This script evaluates a trained UNet model for pet segmentation on the test dataset.
It calculates metrics and visualizes the model's performance with confidence maps.

Example Usage:
    python src/evaluate.py --model_path models/unet_pet_segmentation/best_model.pth --data_dir data/processed
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add the project root to the Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import local modules
from src.models.unet import UNet
from src.utils.metrics import SegmentationMetrics, evaluate_model_metrics
from src.utils.visualize import (
    visualize_prediction_batch,
    visualize_confidence_maps_batch,
    visualize_error_analysis_batch,
    plot_class_distributions,
    visualize_gradcam,
    plot_confusion_matrix
)
from src.train import PetSegmentationDataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a UNet model for pet segmentation"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/unet_pet_segmentation/best_model.pth",
        help="Path to the model checkpoint"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Path to the processed data directory"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Path to store evaluation results"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use (cuda:0, cuda:1, etc. or empty for automatic)"
    )
    
    parser.add_argument(
        "--visualize_samples",
        type=int,
        default=3,
        help="Number of sample batches to visualize (set to 0 to disable visualization)"
    )
    
    return parser.parse_args()


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Load the trained UNet model from a checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model onto
        
    Returns:
        Loaded model
    """
    # Create the model with the same architecture used during training
    model = UNet(
        in_channels=3,
        num_classes=3,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 512, 512],
        kernel_sizes=[[3, 3]] * 6,
        strides=[[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
        n_conv_per_stage=[2] * 6,
        n_conv_per_stage_decoder=[2] * 5,
        conv_bias=True,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        encoder_dropout_rates=[0.0, 0.0, 0.1, 0.2, 0.3, 0.3],
        decoder_dropout_rates=[0.3, 0.2, 0.2, 0.1, 0.0]
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load state dict - handle both formats (just state_dict or full checkpoint)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # Move to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_model(
    model: nn.Module, 
    test_loader: DataLoader, 
    device: torch.device,
    visualize_samples: int = 3
) -> Dict:
    """
    Evaluate the model on the test dataset and return metrics.
    
    Args:
        model: Trained UNet model
        test_loader: DataLoader for test dataset
        device: Device to run the model on
        visualize_samples: Number of sample batches to visualize (0 to disable)
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    # Initialize SegmentationMetrics
    metrics = SegmentationMetrics(num_classes=3, ignore_index=255)
    
    # Collect all ground truth and prediction masks for confusion matrix
    all_preds = []
    all_gts = []
    
    # Evaluate model on test set
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            original_dims = batch["original_dims"]
            
            # Forward pass
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Update metrics for each image in the batch
            for j in range(preds.size(0)):
                orig_h, orig_w = original_dims[j]
                
                # Resize prediction to original dimensions
                pred_resized = F.interpolate(
                    preds[j][None, None].float(),
                    size=(orig_h, orig_w),
                    mode="nearest"
                ).squeeze().cpu().numpy().astype(np.uint8)
                
                # Resize mask to original dimensions
                mask_resized = F.interpolate(
                    masks[j][None, None].float(),
                    size=(orig_h, orig_w),
                    mode="nearest"
                ).squeeze().cpu().numpy().astype(np.uint8)
                
                # Update metrics
                metrics.update(pred_resized, mask_resized)
                
                # Collect for confusion matrix
                all_preds.append(pred_resized)
                all_gts.append(mask_resized)
            
            # Visualize sample batches if requested
            if i < visualize_samples:
                print(f"\nSample Batch {i+1}:")
                print("Prediction Visualization:")
                visualize_prediction_batch(model, batch, device)
                
                print("Confidence Maps Visualization:")
                visualize_confidence_maps_batch(model, batch, device)
                
                print("Error Analysis Visualization:")
                visualize_error_analysis_batch(model, batch, device)
                
                # Visualize grad-CAM for the "Cat" class
                print("Grad-CAM Visualization for Cat Class:")
                target_layer = model.decoder_stages[0].conv_block.block[0]  # First conv in decoder
                visualize_gradcam(model, batch, device, target_class=1, target_layer=target_layer)
    
    # Plot confusion matrix
    if visualize_samples > 0:
        print("\nConfusion Matrix:")
        plot_confusion_matrix(all_preds, all_gts, class_names=["Background", "Cat", "Dog"])
        
        print("\nClass Distribution:")
        plot_class_distributions(all_gts, class_names=["Background", "Cat", "Dog"])
    
    # Calculate and collect all metrics
    results = {
        "pixel_accuracy": metrics.compute_pixel_accuracy(),
        "mean_iou": metrics.compute_mean_iou(),
        "background": {
            "dice": metrics.compute_dice(0),
            "iou": metrics.compute_iou(0),
            "precision": metrics.compute_precision(0),
            "recall": metrics.compute_recall(0)
        },
        "cat": {
            "dice": metrics.compute_dice(1),
            "iou": metrics.compute_iou(1),
            "precision": metrics.compute_precision(1),
            "recall": metrics.compute_recall(1)
        },
        "dog": {
            "dice": metrics.compute_dice(2),
            "iou": metrics.compute_iou(2),
            "precision": metrics.compute_precision(2),
            "recall": metrics.compute_recall(2)
        }
    }
    
    # Calculate mean foreground Dice
    results["mean_foreground_dice"] = np.nanmean([
        results["cat"]["dice"], 
        results["dog"]["dice"]
    ])
    
    return results


def main() -> None:
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, device)
    print("Model loaded successfully")
    
    # Create test dataset and dataloader
    test_dataset = PetSegmentationDataset(
        images_dir=os.path.join(args.data_dir, "Test", "resized"),
        masks_dir=os.path.join(args.data_dir, "Test", "processed_labels"),
        include_augmented=False,
        target_size=(512, 512)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)} images")
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(
        model, 
        test_loader, 
        device,
        visualize_samples=args.visualize_samples
    )
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Pixel Accuracy: {results['pixel_accuracy']:.4f}")
    print(f"Mean IoU: {results['mean_iou']:.4f}")
    print(f"Mean Foreground Dice: {results['mean_foreground_dice']:.4f}")
    
    print("\nClass-wise Metrics:")
    for cls_name in ["background", "cat", "dog"]:
        cls_metrics = results[cls_name]
        print(f"{cls_name.capitalize():<10} | " +
              f"Precision: {cls_metrics['precision']:.4f} | " +
              f"Recall: {cls_metrics['recall']:.4f} | " +
              f"IoU: {cls_metrics['iou']:.4f} | " +
              f"Dice: {cls_metrics['dice']:.4f}")
    
    # Save results to JSON
    import json
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()