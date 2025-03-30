#!/usr/bin/env python
"""
Main training script for the UNet model with optimal hyperparameters
derived from nnU-Net.
"""

import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models.unet import UNetModel, ModelConfig, PetSegmentationDataset, create_transforms, train_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train UNet model using hyperparameters from nnU-Net"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Path to model configuration file"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save trained models"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of epochs to train (overrides config if provided)"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = ModelConfig(args.config)
    
    # Override num_epochs if provided
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    
    # Print configuration
    print("Model Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    
    # Setup paths
    data_dir = Path(args.data_dir)
    train_imgs_dir = data_dir / "Train" / "resized"
    train_masks_dir = data_dir / "Train" / "resized_label"
    val_imgs_dir = data_dir / "Val" / "resized"
    val_masks_dir = data_dir / "Val" / "processed_labels"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create datasets
    train_transforms = create_transforms(config, is_train=True)
    val_transforms = create_transforms(config, is_train=False)
    
    train_dataset = PetSegmentationDataset(
        images_dir=train_imgs_dir,
        masks_dir=train_masks_dir,
        transform=train_transforms
    )
    
    val_dataset = PetSegmentationDataset(
        images_dir=val_imgs_dir,
        masks_dir=val_masks_dir,
        transform=val_transforms
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Print dataset info
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create model
    model = UNetModel(config)
    
    # Resume training if checkpoint is provided
    start_epoch = 0
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        start_epoch, best_metric = model.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch} with metric {best_metric:.4f}")
        start_epoch += 1  # Start from the next epoch
    
    # Train model
    print(f"Starting training for {config.num_epochs} epochs...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        num_epochs=config.num_epochs
    )
    
    print(f"Training completed. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()