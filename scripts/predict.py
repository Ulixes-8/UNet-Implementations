#!/usr/bin/env python
"""
Script for running inference with the trained UNet model.
"""

import os
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.unet import UNetModel, ModelConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with trained UNet model"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model configuration file"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
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
        "--visualize",
        action="store_true",
        help="Generate visualization of predictions"
    )
    
    return parser.parse_args()


def create_visualization(image, mask, output_path):
    """
    Create a visualization of the prediction.
    
    Args:
        image: Original input image
        mask: Predicted segmentation mask
        output_path: Path to save the visualization
    """
    # Create a color map for visualization
    color_map = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # Background: Black
    # Class 1 (cat): Red
    color_map[mask == 1] = [255, 0, 0]
    # Class 2 (dog): Green
    color_map[mask == 2] = [0, 255, 0]
    
    # Create a blended image
    alpha = 0.5
    blended = cv2.addWeighted(image, 1.0, color_map, alpha, 0)
    
    # Create a side-by-side comparison
    h, w = image.shape[:2]
    comparison = np.zeros((h, w*3, 3), dtype=np.uint8)
    comparison[:, :w] = image
    comparison[:, w:2*w] = color_map
    comparison[:, 2*w:] = blended
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Prediction", (w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Blended", (2*w + 10, 30), font, 1, (255, 255, 255), 2)
    
    # Add legend
    legend_y = h - 30
    cv2.putText(comparison, "Cat (Red)", (w + 10, legend_y), font, 0.7, (0, 0, 255), 2)
    cv2.putText(comparison, "Dog (Green)", (w + 200, legend_y), font, 0.7, (0, 255, 0), 2)
    
    # Save the visualization
    cv2.imwrite(output_path, comparison)


def main():
    """Main inference function."""
    args = parse_args()
    
    # Load configuration
    config = ModelConfig(args.config)
    
    # Create model and load checkpoint
    model = UNetModel(config)
    epoch, metric = model.load_checkpoint(args.checkpoint)
    print(f"Loaded checkpoint from epoch {epoch} with metric {metric:.4f}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Check if input is a file or directory
    input_path = Path(args.input)
    if input_path.is_file():
        # Process a single image
        file_paths = [input_path]
    else:
        # Process all images in directory
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
        file_paths = []
        for ext in extensions:
            file_paths.extend(list(input_path.glob(f"*{ext}")))
        file_paths.sort()
    
    print(f"Found {len(file_paths)} images to process")
    
    # Process each image
    for file_path in tqdm(file_paths, desc="Processing images"):
        try:
            # Load image
            image = cv2.imread(str(file_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Make prediction
            prediction = model.predict(image)
            
            # Save prediction as a grayscale image
            output_filename = file_path.stem + "_pred.png"
            output_path = Path(args.output) / output_filename
            cv2.imwrite(str(output_path), prediction)
            
            # Create visualization if requested
            if args.visualize:
                viz_filename = file_path.stem + "_viz.jpg"
                viz_path = Path(args.output) / viz_filename
                create_visualization(image, prediction, str(viz_path))
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Inference completed. Results saved to {args.output}")


if __name__ == "__main__":
    main()