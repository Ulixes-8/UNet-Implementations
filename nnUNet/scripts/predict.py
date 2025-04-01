#!/usr/bin/env python
"""
Script: predict.py

This script runs inference with a trained UNet model for pet segmentation.
It supports single image prediction or batch processing of a directory of images.

Features:
- Support for multiple model architectures
- Visualization of prediction results with overlays
- Batch processing of multiple images
- Confidence map generation
- Optional postprocessing for smoother segmentation
- Metrics calculation when ground truth is available

Example Usage:
    python predict.py \
        --model models/unet_pet_segmentation/best_model.pth \
        --input path/to/image_or_directory \
        --output results \
        --visualize
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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Optional path to log file
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("predict")
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
        description="Run inference with a trained UNet model for pet segmentation"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model checkpoint"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image or directory of images"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output directory"
    )
    
    parser.add_argument(
        "--ground_truth",
        type=str,
        default=None,
        help="Path to ground truth masks (optional, for evaluation)"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations of predictions"
    )
    
    parser.add_argument(
        "--confidence_maps",
        action="store_true",
        help="Generate confidence maps for predictions"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing multiple images"
    )
    
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Apply postprocessing to the predictions (removes small objects)"
    )
    
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Target size for input images (height width)"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading"
    )
    
    return parser.parse_args()


class ModelConfig:
    """Configuration class for the model."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize the configuration from a dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters
        """
        self.model_name = config_dict.get("model_name", "unet")
        self.encoder_name = config_dict.get("encoder_name", "resnet34")
        self.encoder_weights = config_dict.get("encoder_weights", "imagenet")
        self.in_channels = config_dict.get("in_channels", 3)
        self.classes = config_dict.get("classes", 3)
        self.patch_size = config_dict.get("patch_size", [512, 512])


class InferenceDataset(Dataset):
    """Dataset class for inference."""
    
    def __init__(
        self,
        image_paths: List[Path],
        transform: A.Compose
    ):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to images
            transform: Albumentations transformations to apply
        """
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dict containing image and path
        """
        image_path = self.image_paths[idx]
        
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original size
        original_size = image.shape[:2]
        
        # Apply transformations
        transformed = self.transform(image=image)
        image_tensor = transformed["image"]
        
        return {
            "image": image_tensor,
            "path": str(image_path),
            "original_size": original_size
        }


def create_transform(target_size: Tuple[int, int]) -> A.Compose:
    """
    Create transformation for inference.
    
    Args:
        target_size: Target size (height, width) for input images
        
    Returns:
        Albumentations Compose object
    """
    transform = A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.Normalize(),
        ToTensorV2()
    ])
    
    return transform


def create_model(config: ModelConfig, device: torch.device) -> nn.Module:
    """
    Create UNet model based on configuration.
    
    Args:
        config: Model configuration
        device: Device to load the model on
        
    Returns:
        PyTorch UNet model
    """
    if config.model_name.lower() == "unet":
        model = smp.Unet(
            encoder_name=config.encoder_name,
            encoder_weights=None,  # No need for pretrained weights during inference
            in_channels=config.in_channels,
            classes=config.classes,
            activation=None  # We'll apply softmax manually
        )
    elif config.model_name.lower() == "unet++":
        model = smp.UnetPlusPlus(
            encoder_name=config.encoder_name,
            encoder_weights=None,
            in_channels=config.in_channels,
            classes=config.classes,
            activation=None
        )
    elif config.model_name.lower() == "deeplabv3+":
        model = smp.DeepLabV3Plus(
            encoder_name=config.encoder_name,
            encoder_weights=None,
            in_channels=config.in_channels,
            classes=config.classes,
            activation=None
        )
    else:
        # Default to UNet
        model = smp.Unet(
            encoder_name=config.encoder_name,
            encoder_weights=None,
            in_channels=config.in_channels,
            classes=config.classes,
            activation=None
        )
    
    return model.to(device)


def load_model(model_path: str, device: torch.device) -> Tuple[nn.Module, ModelConfig]:
    """
    Load model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Tuple of (model, config)
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract configuration
    if "config" in checkpoint:
        # Full checkpoint with configuration
        config_dict = checkpoint["config"]
        config = ModelConfig(config_dict)
        
        # Create model
        model = create_model(config, device)
        
        # Load state dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        # Assume it's a simple state dict
        # Use default configuration
        config = ModelConfig({})
        model = create_model(config, device)
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, config


def get_input_files(input_path: str) -> List[Path]:
    """
    Get list of input image files.
    
    Args:
        input_path: Path to input file or directory
        
    Returns:
        List of image file paths
    """
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Single file
        return [input_path]
    elif input_path.is_dir():
        # Directory of files
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        image_files = []
        
        for ext in extensions:
            image_files.extend(list(input_path.glob(f"*{ext}")))
            image_files.extend(list(input_path.glob(f"*{ext.upper()}")))
        
        return sorted(image_files)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def predict_batch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    postprocess: bool = False
) -> List[Dict[str, Any]]:
    """
    Run prediction on a batch of images.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for inference
        device: Device to run inference on
        postprocess: Whether to apply postprocessing
        
    Returns:
        List of dictionaries with prediction results
    """
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            # Get input
            images = batch["image"].to(device)
            paths = batch["path"]
            original_sizes = batch["original_size"]
            
            # Run inference
            outputs = model(images)
            
            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)
            
            # Get class predictions
            preds = torch.argmax(probs, dim=1)
            
            # Process each prediction in the batch
            for i in range(len(paths)):
                # Get prediction for this sample
                pred = preds[i].cpu().numpy()
                prob = probs[i].cpu().numpy()
                
                # Resize prediction to original size if needed
                original_size = (original_sizes[i][0].item(), original_sizes[i][1].item())
                if pred.shape[:2] != original_size:
                    pred_resized = cv2.resize(
                        pred.astype(np.uint8),
                        (original_size[1], original_size[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                else:
                    pred_resized = pred
                
                # Apply postprocessing if requested
                if postprocess:
                    pred_resized = apply_postprocessing(pred_resized)
                
                # Add to results
                results.append({
                    "path": paths[i],
                    "prediction": pred_resized,
                    "probabilities": prob,
                    "original_size": original_size
                })
    
    return results


def apply_postprocessing(pred: np.ndarray) -> np.ndarray:
    """
    Apply postprocessing to prediction.
    
    Args:
        pred: Prediction as numpy array
        
    Returns:
        Postprocessed prediction
    """
    # Create a copy of the prediction
    processed = pred.copy()
    
    # Apply morphological operations to remove small objects
    for class_id in range(1, 3):  # Process each class (cat, dog)
        # Create binary mask for this class
        mask = (processed == class_id).astype(np.uint8)
        
        # Apply morphological opening to remove small objects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_opened, connectivity=8)
        
        # Remove small components (less than 1% of the image area)
        min_size = 0.01 * mask.shape[0] * mask.shape[1]
        for label in range(1, num_labels):  # Skip background (0)
            if stats[label, cv2.CC_STAT_AREA] < min_size:
                mask_opened[labels == label] = 0
        
        # Update the processed prediction
        processed[mask == 1] = 0  # Reset this class
        processed[mask_opened == 1] = class_id  # Set the filtered class
    
    return processed


def create_visualization(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create visualization of prediction.
    
    Args:
        image: Original image as numpy array
        prediction: Prediction as numpy array
        ground_truth: Optional ground truth mask
        
    Returns:
        Visualization as numpy array
    """
    # Create a color map for visualization
    color_map = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    
    # Background: transparent
    # Class 1 (cat): Red with alpha
    color_map[prediction == 1] = [255, 0, 0]  # BGR for OpenCV
    
    # Class 2 (dog): Green with alpha
    color_map[prediction == 2] = [0, 255, 0]  # BGR for OpenCV
    
    # Create a blended image with alpha=0.5
    alpha = 0.5
    blended = cv2.addWeighted(image, 1.0, color_map, alpha, 0)
    
    # Add ground truth comparison if provided
    if ground_truth is not None:
        # Create a color map for ground truth
        gt_color_map = np.zeros((ground_truth.shape[0], ground_truth.shape[1], 3), dtype=np.uint8)
        
        # Background: transparent
        # Class 1 (cat): Blue with alpha
        gt_color_map[ground_truth == 1] = [0, 0, 255]  # BGR for OpenCV
        
        # Class 2 (dog): Cyan with alpha
        gt_color_map[ground_truth == 2] = [255, 255, 0]  # BGR for OpenCV
        
        # Create ground truth blend
        gt_blended = cv2.addWeighted(image, 1.0, gt_color_map, alpha, 0)
        
        # Create side-by-side comparison
        h, w = image.shape[:2]
        comparison = np.zeros((h, w*3, 3), dtype=np.uint8)
        comparison[:, :w] = image
        comparison[:, w:2*w] = blended
        comparison[:, 2*w:] = gt_blended
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Prediction", (w + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Ground Truth", (2*w + 10, 30), font, 1, (255, 255, 255), 2)
        
        return comparison
    else:
        # Create side-by-side comparison without ground truth
        h, w = image.shape[:2]
        comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
        comparison[:, :w] = image
        comparison[:, w:] = blended
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Prediction", (w + 10, 30), font, 1, (255, 255, 255), 2)
        
        return comparison


def create_confidence_map(
    probabilities: np.ndarray,
    prediction: np.ndarray
) -> np.ndarray:
    """
    Create confidence map visualization.
    
    Args:
        probabilities: Prediction probabilities (C, H, W)
        prediction: Class prediction (H, W)
        
    Returns:
        Confidence map as numpy array
    """
    # Get confidence for the predicted class
    confidence = np.zeros(prediction.shape, dtype=np.float32)
    
    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            class_id = prediction[i, j]
            confidence[i, j] = probabilities[class_id, i, j]
    
    # Convert to heatmap
    confidence_map = (confidence * 255).astype(np.uint8)
    confidence_colormap = cv2.applyColorMap(confidence_map, cv2.COLORMAP_JET)
    
    return confidence_colormap


def load_ground_truth(
    ground_truth_dir: str,
    image_path: str
) -> Optional[np.ndarray]:
    """
    Load ground truth mask if available.
    
    Args:
        ground_truth_dir: Directory with ground truth masks
        image_path: Path to input image
        
    Returns:
        Ground truth mask or None if not found
    """
    if not ground_truth_dir:
        return None
    
    # Get filename without extension
    image_filename = Path(image_path).stem
    
    # Check for mask file
    ground_truth_dir = Path(ground_truth_dir)
    mask_path = ground_truth_dir / f"{image_filename}.png"
    
    if not mask_path.exists():
        # Try alternatives
        alternatives = list(ground_truth_dir.glob(f"{image_filename}.*"))
        if not alternatives:
            return None
        mask_path = alternatives[0]
    
    # Load mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    return mask


def calculate_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray
) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        prediction: Prediction as numpy array
        ground_truth: Ground truth mask
        
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # Calculate Dice coefficient for each class
    for class_id in range(1, 3):  # Skip background
        pred_mask = (prediction == class_id)
        gt_mask = (ground_truth == class_id)
        
        # Calculate intersection and union
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum()
        
        # Calculate Dice coefficient
        if union > 0:
            dice = (2.0 * intersection) / union
        else:
            dice = 1.0 if pred_mask.sum() == 0 else 0.0
        
        metrics[f"dice_class_{class_id}"] = float(dice)
    
    # Calculate average Dice coefficient (macro)
    metrics["dice_macro"] = sum(metrics.values()) / len(metrics)
    
    # Calculate pixel accuracy
    correct = (prediction == ground_truth).sum()
    total = prediction.size
    pixel_acc = correct / total
    metrics["pixel_accuracy"] = float(pixel_acc)
    
    return metrics


def main() -> None:
    """Main function for prediction."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    log_file = os.path.join(args.output, "prediction.log")
    os.makedirs(args.output, exist_ok=True)
    logger = setup_logging(log_file)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model, config = load_model(args.model, device)
    logger.info(f"Loaded {config.model_name} model with {config.encoder_name} encoder")
    
    # Get input files
    logger.info(f"Getting input files from {args.input}")
    input_files = get_input_files(args.input)
    logger.info(f"Found {len(input_files)} input files")
    
    # Create output directories
    predictions_dir = os.path.join(args.output, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    if args.visualize:
        visualizations_dir = os.path.join(args.output, "visualizations")
        os.makedirs(visualizations_dir, exist_ok=True)
    
    if args.confidence_maps:
        confidence_dir = os.path.join(args.output, "confidence_maps")
        os.makedirs(confidence_dir, exist_ok=True)
    
    # Create transform
    target_size = tuple(args.target_size)
    transform = create_transform(target_size)
    
    # Create dataset and dataloader
    dataset = InferenceDataset(input_files, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Run prediction
    logger.info("Running prediction...")
    results = predict_batch(model, dataloader, device, args.postprocess)
    
    # Process results
    metrics_list = []
    
    for result in tqdm(results, desc="Processing results"):
        # Get result data
        image_path = result["path"]
        prediction = result["prediction"]
        probabilities = result["probabilities"]
        original_size = result["original_size"]
        
        # Get filename
        filename = Path(image_path).stem
        
        # Save prediction
        prediction_path = os.path.join(predictions_dir, f"{filename}.png")
        cv2.imwrite(prediction_path, prediction.astype(np.uint8))
        
        # Load original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image if needed
        if image.shape[:2] != original_size:
            image = cv2.resize(image, (original_size[1], original_size[0]))
        
        # Load ground truth if available
        ground_truth = None
        if args.ground_truth:
            ground_truth = load_ground_truth(args.ground_truth, image_path)
        
        # Create and save visualization if requested
        if args.visualize:
            visualization = create_visualization(image, prediction, ground_truth)
            visualization_path = os.path.join(visualizations_dir, f"{filename}_viz.jpg")
            cv2.imwrite(visualization_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        
        # Create and save confidence map if requested
        if args.confidence_maps:
            # Resize probabilities to original size if needed
            if probabilities.shape[1:] != original_size:
                resized_probs = np.zeros((probabilities.shape[0], *original_size), dtype=np.float32)
                for c in range(probabilities.shape[0]):
                    resized_probs[c] = cv2.resize(
                        probabilities[c],
                        (original_size[1], original_size[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                probabilities = resized_probs
            
            confidence_map = create_confidence_map(probabilities, prediction)
            confidence_path = os.path.join(confidence_dir, f"{filename}_conf.jpg")
            cv2.imwrite(confidence_path, confidence_map)
        
        # Calculate metrics if ground truth is available
        if ground_truth is not None:
            metrics = calculate_metrics(prediction, ground_truth)
            metrics["filename"] = filename
            metrics_list.append(metrics)
    
    # Save metrics if available
    if metrics_list:
        metrics_path = os.path.join(args.output, "metrics.json")
        
        # Calculate average metrics
        avg_metrics = {}
        for key in metrics_list[0].keys():
            if key != "filename":
                avg_metrics[f"avg_{key}"] = np.mean([m[key] for m in metrics_list])
        
        # Save all metrics
        with open(metrics_path, "w") as f:
            json.dump({
                "metrics_per_image": metrics_list,
                "average_metrics": avg_metrics
            }, f, indent=4)
        
        logger.info(f"Evaluation metrics:")
        for key, value in avg_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
    
    logger.info(f"Prediction completed. Results saved to {args.output}")


if __name__ == "__main__":
    main()