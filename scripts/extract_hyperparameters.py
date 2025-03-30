#!/usr/bin/env python
"""
This script extracts the optimal hyperparameters from nnU-Net plans
and updates our UNet configuration file.
"""

import os
import json
import yaml
import argparse
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract hyperparameters from nnU-Net plans and update config"
    )
    
    parser.add_argument(
        "--plans_file",
        type=str,
        required=True,
        help="Path to nnU-Net plans.json file"
    )
    
    parser.add_argument(
        "--output_config",
        type=str,
        default="config/model_config.yaml",
        help="Path to save the updated configuration"
    )
    
    parser.add_argument(
        "--template_config",
        type=str,
        default="config/model_config_template.yaml",
        help="Path to the template configuration file"
    )
    
    return parser.parse_args()


def extract_hyperparameters(plans_file):
    """
    Extract relevant hyperparameters from nnU-Net plans.json.
    
    Args:
        plans_file: Path to the plans.json file
        
    Returns:
        Dictionary with extracted hyperparameters
    """
    # Load the plans file
    with open(plans_file, 'r') as f:
        plans = json.load(f)
    
    hyperparameters = {}
    
    # Extract patch size
    if "plans_per_stage" in plans and len(plans["plans_per_stage"]) > 0:
        stage_plans = plans["plans_per_stage"][0]  # Use first stage for 2D
        
        # Extract patch size
        if "patch_size" in stage_plans:
            hyperparameters["patch_size"] = stage_plans["patch_size"]
            # For 2D, we only need the first two dimensions
            if len(hyperparameters["patch_size"]) > 2:
                hyperparameters["patch_size"] = hyperparameters["patch_size"][:2]
        
        # Extract batch size
        if "batch_size" in stage_plans:
            hyperparameters["batch_size"] = stage_plans["batch_size"]
    
    # Extract learning rate
    if "UNet_base_num_features" in plans:
        # Use the UNet base features to determine the starting network width
        hyperparameters["base_num_features"] = plans["UNet_base_num_features"]
    
    # Extract network depth
    if "UNet_max_num_features" in plans:
        # Calculate network depth based on max features
        max_features = plans["UNet_max_num_features"]
        base_features = hyperparameters.get("base_num_features", 32)
        depth = 1
        while base_features * (2 ** (depth - 1)) < max_features:
            depth += 1
        hyperparameters["depth"] = depth
    
    # Extract data augmentation parameters
    # Note: nnU-Net uses a fixed set of augmentations that we could adapt
    hyperparameters["use_augmentation"] = True
    
    # Extract learning rate (nnU-Net typically uses 1e-4 for Adam or 1e-2 for SGD)
    # We'll set a default and adjust based on optimizer
    hyperparameters["learning_rate"] = 1e-4
    
    # nnU-Net uses different loss functions based on the task
    # For segmentation, it typically uses a combination of Dice and CE loss
    hyperparameters["loss"] = "combined"
    
    return hyperparameters


def update_config(template_path, hyperparameters, output_path):
    """
    Update the template config with extracted hyperparameters.
    
    Args:
        template_path: Path to template config
        hyperparameters: Dictionary with hyperparameters
        output_path: Path to save updated config
    """
    # Load template config
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with hyperparameters
    for key, value in hyperparameters.items():
        if key in config:
            config[key] = value
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save updated config
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Updated config saved to {output_path}")


def main():
    """Main function to extract hyperparameters and update config."""
    args = parse_args()
    
    print(f"Extracting hyperparameters from {args.plans_file}")
    hyperparameters = extract_hyperparameters(args.plans_file)
    
    print("Extracted hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"  {key}: {value}")
    
    print(f"Updating config using template {args.template_config}")
    update_config(args.template_config, hyperparameters, args.output_config)


if __name__ == "__main__":
    main()