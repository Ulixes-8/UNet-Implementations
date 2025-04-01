#!/usr/bin/env python
"""
Script: extract_hyperparameters.py

This script extracts optimal hyperparameters from nnU-Net plans and translates them
into a configuration format compatible with our UNet implementation using
segmentation-models-pytorch.

The script extracts key parameters such as:
- Network architecture (depth, filters)
- Optimization parameters (learning rate, batch size)
- Data processing parameters (patch size)
- Loss function configuration

Example Usage:
    python extract_hyperparameters.py \
        --plans_file $nnUNet_results/Dataset001_PetSegmentation/nnUNetTrainer__nnUNetPlans__2d/plans.json \
        --output_config config/model_config.yaml \
        --template_config config/model_config_template.yaml
"""

import os
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Optional path to log file
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("extract_hyperparameters")
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
        description="Extract hyperparameters from nnU-Net plans and update config"
    )
    
    parser.add_argument(
        "--plans_file",
        type=str,
        required=True,
        help="Path to nnU-Net plans.json file"
    )
    
    parser.add_argument(
        "--debug_file",
        type=str,
        default=None,
        help="Path to nnU-Net debug.json file (optional, for additional parameters)"
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
    
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to log file (optional)"
    )
    
    return parser.parse_args()


def load_json_file(file_path: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Load a JSON file safely.
    
    Args:
        file_path: Path to the JSON file
        logger: Logger for output
        
    Returns:
        Dictionary with the loaded JSON data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON file: {file_path}")
        raise


def load_yaml_file(file_path: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Load a YAML file safely.
    
    Args:
        file_path: Path to the YAML file
        logger: Logger for output
        
    Returns:
        Dictionary with the loaded YAML data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If the file is not valid YAML
    """
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except yaml.YAMLError:
        logger.error(f"Invalid YAML file: {file_path}")
        raise


def extract_hyperparameters(
    plans_data: Dict[str, Any],
    debug_data: Optional[Dict[str, Any]] = None,
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Extract relevant hyperparameters from nnU-Net plans.
    
    Args:
        plans_data: Dictionary with the plans data
        debug_data: Optional dictionary with debug data for additional parameters
        logger: Logger for output
        
    Returns:
        Dictionary with extracted hyperparameters
    """
    hyperparameters = {}
    
    # Initialize with default values
    hyperparameters["model_name"] = "unet"
    hyperparameters["encoder_name"] = "resnet34"
    hyperparameters["encoder_weights"] = "imagenet"
    hyperparameters["in_channels"] = 3  # RGB images for pet dataset
    hyperparameters["classes"] = 3  # Background, cat, dog
    
    # Extract network architecture parameters
    hyperparameters["base_num_features"] = plans_data.get("UNet_base_num_features", 32)
    hyperparameters["max_num_features"] = plans_data.get("UNet_max_num_features", 512)
    
    # Calculate depth based on base and max features
    base_features = hyperparameters["base_num_features"]
    max_features = hyperparameters["max_num_features"]
    depth = 1
    features = base_features
    while features < max_features:
        features *= 2
        depth += 1
    hyperparameters["depth"] = depth
    
    # Extract patch size from the first stage of plans
    if "plans_per_stage" in plans_data and len(plans_data["plans_per_stage"]) > 0:
        stage_plans = plans_data["plans_per_stage"][0]
        
        # Get patch size - for 2D, we only need the first two dimensions
        if "patch_size" in stage_plans:
            patch_size = stage_plans["patch_size"]
            # Ensure we have at least 2 dimensions
            if len(patch_size) >= 2:
                hyperparameters["patch_size"] = patch_size[:2]
            else:
                hyperparameters["patch_size"] = [512, 512]  # Default if not found
                if logger:
                    logger.warning("Invalid patch size in plans, using default [512, 512]")
        
        # Extract batch size
        if "batch_size" in stage_plans:
            hyperparameters["batch_size"] = stage_plans["batch_size"]
    
    # Extract optimizer parameters from debug data if available
    if debug_data is not None:
        blueprint = debug_data.get("blueprint", {})
        
        # Extract learning rate
        lr = blueprint.get("learning_rate", 1e-4)
        hyperparameters["learning_rate"] = float(lr)
        
        # Extract optimizer
        optimizer_name = blueprint.get("optimizer_name", "SGD")
        if optimizer_name.lower() == "sgd":
            hyperparameters["optimizer"] = "SGD"
        else:
            hyperparameters["optimizer"] = "Adam"
        
        # Extract loss type
        loss_name = blueprint.get("loss", "DC_and_CE_loss")
        if "ce" in loss_name.lower() and "dc" in loss_name.lower():
            hyperparameters["loss"] = "combined"
        elif "ce" in loss_name.lower():
            hyperparameters["loss"] = "ce"
        elif "dc" in loss_name.lower() or "dice" in loss_name.lower():
            hyperparameters["loss"] = "dice"
        else:
            hyperparameters["loss"] = "combined"  # Default
        
        # Extract weight decay
        weight_decay = blueprint.get("weight_decay", 3e-5)
        hyperparameters["weight_decay"] = float(weight_decay)
    else:
        # Default values if debug data is not available
        hyperparameters["learning_rate"] = 1e-4
        hyperparameters["optimizer"] = "Adam"
        hyperparameters["loss"] = "combined"
        hyperparameters["weight_decay"] = 3e-5
    
    # Extract augmentation parameters
    # For 2D, we enable augmentation by default
    hyperparameters["use_augmentation"] = True
    
    # Number of epochs
    hyperparameters["num_epochs"] = 1000  # Default, can be adjusted
    
    return hyperparameters


def map_to_smp_architecture(
    hyperparameters: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Map nnU-Net hyperparameters to segmentation-models-pytorch configuration.
    
    Args:
        hyperparameters: Dictionary with extracted hyperparameters
        logger: Logger for output
        
    Returns:
        Dictionary with mapped hyperparameters for SMP
    """
    smp_config = hyperparameters.copy()
    
    # Map encoder name based on depth and other parameters
    # Default to resnet34 as it's a good balance of performance and speed
    depth = hyperparameters.get("depth", 5)
    base_features = hyperparameters.get("base_num_features", 32)
    
    # Choose an appropriate encoder based on network depth
    if depth <= 5:
        if base_features <= 32:
            smp_config["encoder_name"] = "resnet34"
        else:
            smp_config["encoder_name"] = "resnet50"
    else:
        # For deeper networks, use a more powerful encoder
        smp_config["encoder_name"] = "resnet101"
    
    logger.info(f"Mapped network depth {depth} to encoder: {smp_config['encoder_name']}")
    
    # Clean up parameters not needed by SMP
    if "base_num_features" in smp_config:
        del smp_config["base_num_features"]
    if "max_num_features" in smp_config:
        del smp_config["max_num_features"]
    if "depth" in smp_config:
        del smp_config["depth"]
    
    return smp_config


def update_config(
    template: Dict[str, Any],
    hyperparameters: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Update the template config with extracted hyperparameters.
    
    Args:
        template: Template configuration dictionary
        hyperparameters: Dictionary with extracted hyperparameters
        logger: Logger for output
        
    Returns:
        Updated configuration dictionary
    """
    # Create a deep copy of the template
    updated_config = template.copy()
    
    # Update with hyperparameters
    for key, value in hyperparameters.items():
        if key in updated_config:
            if key == "patch_size" and not isinstance(value, list):
                # Ensure patch_size is a list
                updated_config[key] = [value, value]
            else:
                updated_config[key] = value
    
    # Log changes
    for key, value in updated_config.items():
        if key in template and template[key] != updated_config[key]:
            logger.info(f"Updated {key}: {template[key]} -> {updated_config[key]}")
    
    return updated_config


def main() -> None:
    """Main function to extract hyperparameters and update config."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    # Load plans file
    logger.info(f"Loading plans file: {args.plans_file}")
    plans_data = load_json_file(args.plans_file, logger)
    
    # Load debug file if provided
    debug_data = None
    if args.debug_file:
        logger.info(f"Loading debug file: {args.debug_file}")
        debug_data = load_json_file(args.debug_file, logger)
    
    # Load template config
    logger.info(f"Loading template config: {args.template_config}")
    template_config = load_yaml_file(args.template_config, logger)
    
    # Extract hyperparameters
    logger.info("Extracting hyperparameters from nnU-Net plans")
    hyperparameters = extract_hyperparameters(plans_data, debug_data, logger)
    
    # Map to SMP architecture
    logger.info("Mapping parameters to segmentation-models-pytorch format")
    smp_config = map_to_smp_architecture(hyperparameters, logger)
    
    # Update config
    logger.info("Updating configuration with extracted parameters")
    updated_config = update_config(template_config, smp_config, logger)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save updated config
    logger.info(f"Saving updated config to: {args.output_config}")
    with open(args.output_config, 'w') as f:
        yaml.dump(updated_config, f, default_flow_style=False)
    
    logger.info("Hyperparameter extraction and configuration update complete")
    logger.info("You can now train your UNet model with the optimized configuration")


if __name__ == "__main__":
    main()