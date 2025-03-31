#!/usr/bin/env python
"""
Script: train.py

This script trains a UNet model for pet segmentation using the Oxford-IIIT Pet Dataset.
It handles the training loop, validation, checkpointing, and logging.

Example Usage:
    python src/train.py --data_dir data --output_dir models/unet_pet_segmentation
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
from tqdm import tqdm

# Import local modules
from models.losses import DC_and_CE_loss, DeepSupervisionWrapper
from models.unet import UNet


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a UNet model for pet segmentation"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/ulixes/segmentation_cv/unet/data/processed",
        help="Path to the processed data directory"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/unet_pet_segmentation",
        help="Path to store model checkpoints and logs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,  # From nnUNet configuration
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,  # From nnUNet configuration
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,  # From nnUNet configuration
        help="Initial learning rate"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=3e-5,  # From nnUNet configuration
        help="Weight decay for optimizer"
    )
    
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.99,  # From nnUNet configuration
        help="Momentum for SGD optimizer"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading"
    )
    
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,  # From nnUNet configuration
        help="Save a checkpoint every N epochs"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use (cuda:0, cuda:1, etc. or empty for automatic)"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to a checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision training"
    )
    
    return parser.parse_args()


class PetSegmentationDataset(Dataset):
    """Dataset class for the Oxford-IIIT Pet segmentation dataset."""
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        include_augmented: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            images_dir: Directory containing images
            masks_dir: Directory containing mask annotations
            include_augmented: Whether to include augmented images (if available)
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        
        # Get all image files from the directory
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        
        # Check for augmented data directory
        if include_augmented and (self.images_dir.parent / "augmented" / "images").exists():
            aug_images_dir = self.images_dir.parent / "augmented" / "images"
            aug_masks_dir = self.images_dir.parent / "augmented" / "masks"
            
            # Add augmented images to the dataset
            aug_image_files = sorted(list(aug_images_dir.glob("*.jpg")))
            self.aug_image_files = aug_image_files
            self.aug_masks_dir = aug_masks_dir
            
            self.image_files.extend(aug_image_files)
        else:
            self.aug_image_files = []
            self.aug_masks_dir = None
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dict containing image and mask tensors
        """
        # Get image file path
        img_path = self.image_files[idx]
        
        # Determine if this is an augmented image
        is_augmented = img_path in self.aug_image_files if self.aug_image_files else False
        
        # Get corresponding mask file path
        if is_augmented and self.aug_masks_dir:
            mask_path = self.aug_masks_dir / f"{img_path.stem}.png"
        else:
            mask_path = self.masks_dir / f"{img_path.stem}.png"
        
        # Load image and mask
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Convert 255 (ignore label) to 250 (a new ignore label that won't cause index errors)
        # This is a simpler approach to handle invalid indices in the loss function
        mask = np.where(mask == 255, 250, mask)
        
        # Ensure other values are within valid range (0, 1, 2)
        mask = np.where((mask > 2) & (mask != 250), 0, mask)
        
        # Convert image to tensor and normalize (0-1)
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        
        # Apply standardization (approximately equivalent to ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        # Convert mask to tensor
        mask = torch.from_numpy(mask).long()
        
        return {"image": image, "mask": mask}


def create_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int,
    num_workers: int
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Set directories for training data
    train_imgs_dir = Path(data_dir) / "Train" / "resized"
    train_masks_dir = Path(data_dir) / "Train" / "resized_label"
    
    # Set directories for validation data
    val_imgs_dir = Path(data_dir) / "Val" / "resized"
    val_masks_dir = Path(data_dir) / "Val" / "processed_labels"
    
    # Create datasets
    train_dataset = PetSegmentationDataset(
        images_dir=train_imgs_dir,
        masks_dir=train_masks_dir,
        include_augmented=True
    )
    
    val_dataset = PetSegmentationDataset(
        images_dir=val_imgs_dir,
        masks_dir=val_masks_dir,
        include_augmented=False
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # To ensure consistent batch sizes for batch norm
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_model(device: torch.device) -> nn.Module:
    """
    Create the UNet model based on nnUNet hyperparameters.
    
    Args:
        device: Device to use for the model
        
    Returns:
        Initialized model
    """
    # Create UNet with nnUNet hyperparameters
    model = UNet(
        in_channels=3,  # RGB images
        num_classes=3,  # background, cat, dog
        n_stages=8,
        features_per_stage=[32, 64, 128, 256, 512, 512, 512, 512],
        kernel_sizes=[[3, 3]] * 8,
        strides=[[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
        n_conv_per_stage=[2] * 8,
        n_conv_per_stage_decoder=[2] * 7,
        conv_bias=True,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=True
    )
    
    # Move model to device
    model = model.to(device)
    
    return model


def create_optimizer(model: nn.Module, lr: float, weight_decay: float, momentum: float) -> optim.Optimizer:
    """
    Create an optimizer for the model.
    
    Args:
        model: The model to optimize
        lr: Learning rate
        weight_decay: Weight decay factor
        momentum: Momentum factor for SGD
        
    Returns:
        Configured optimizer
    """
    # Use SGD with momentum and nesterov, as in nnUNet
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=True
    )
    
    return optimizer


def create_lr_scheduler(optimizer: optim.Optimizer, max_epochs: int) -> optim.lr_scheduler._LRScheduler:
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: The optimizer to schedule
        max_epochs: Maximum number of epochs
        
    Returns:
        Learning rate scheduler
    """
    # Use polynomial learning rate decay as in nnUNet
    def poly_lr_lambda(current_epoch: int) -> float:
        """Polynomial learning rate decay function."""
        return (1 - current_epoch / max_epochs) ** 0.9
    
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=poly_lr_lambda
    )
    
    return scheduler


def get_loss_function() -> nn.Module:
    """
    Create the loss function with deep supervision wrapper.
    
    Returns:
        Configured loss function
    """
    # Set up Dice and Cross-Entropy loss
    soft_dice_kwargs = {
        'batch_dice': True,
        'do_bg': False,  # Don't include background in Dice
        'smooth': 1e-5
    }
    
    ce_kwargs = {}
    
    # Create combined loss
    loss_func = DC_and_CE_loss(
        soft_dice_kwargs=soft_dice_kwargs,
        ce_kwargs=ce_kwargs
    )
    
    # Wrap with deep supervision
    loss_func = DeepSupervisionWrapper(loss_func)
    
    return loss_func


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    ignore_label: int = 250  # New ignore label
) -> Tuple[float, Dict[str, float]]:
    """
    Validate the model on the validation set.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        loss_function: Loss function
        device: Device to use
        ignore_label: Label to ignore in metrics (border class)
        
    Returns:
        Tuple of (validation_loss, metrics_dict)
    """
    model.eval()
    
    # Initialize metrics
    val_loss = 0.0
    dice_scores = {
        "background": 0.0,
        "cat": 0.0,
        "dog": 0.0,
        "mean_foreground": 0.0
    }
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = loss_function(outputs, masks)
            val_loss += loss.item()
            
            # For metrics, use the main output if deep supervision is used
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            
            # Convert outputs to predictions
            preds = torch.argmax(outputs, dim=1, keepdim=True)
            
            # Calculate Dice scores for each class
            for cls_idx, cls_name in [(0, "background"), (1, "cat"), (2, "dog")]:
                # Create binary masks for predictions and ground truth
                pred_cls = (preds == cls_idx).float()
                mask_cls = (masks == cls_idx).float()
                
                # Ignore border pixels (class 250)
                ignore_mask = (masks != ignore_label).float()
                pred_cls = pred_cls * ignore_mask
                mask_cls = mask_cls * ignore_mask
                
                # Calculate intersection and union
                intersection = (pred_cls * mask_cls).sum()
                union = pred_cls.sum() + mask_cls.sum()
                
                # Calculate Dice score
                if union > 0:
                    dice = (2.0 * intersection) / (union + 1e-5)
                else:
                    dice = torch.tensor(1.0, device=device)  # Perfect score if both are empty
                
                dice_scores[cls_name] += dice.item()
            
    # Average metrics
    num_batches = len(val_loader)
    val_loss /= num_batches
    
    for cls_name in dice_scores:
        dice_scores[cls_name] /= num_batches
    
    # Calculate mean foreground Dice (cat and dog)
    dice_scores["mean_foreground"] = (dice_scores["cat"] + dice_scores["dog"]) / 2.0
    
    return val_loss, dice_scores


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        loss_function: Loss function
        device: Device to use
        scaler: Optional grad scaler for mixed precision training
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    
    epoch_loss = 0.0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with optional mixed precision
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = loss_function(outputs, masks)
                
            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision training
            outputs = model(images)
            loss = loss_function(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(train_loader)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_dice: float,
    output_dir: Path,
    is_best: bool = False
) -> None:
    """
    Save a checkpoint of the model.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: LR scheduler state to save
        epoch: Current epoch number
        best_dice: Best validation Dice score so far
        output_dir: Directory to save to
        is_best: Whether this is the best model so far
    """
    # Create the checkpoint directory
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the checkpoint dictionary
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_dice": best_dice
    }
    
    # Save the checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save as best model if requested
    if is_best:
        best_path = output_dir / "best_model.pth"
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")


def main() -> None:
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training configuration
    with open(output_dir / "training_config.json", "w") as f:
        config = vars(args)
        json.dump(config, f, indent=4)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        args.data_dir, 
        args.batch_size, 
        args.num_workers
    )
    
    # Create model
    model = create_model(device)
    print(f"Created model with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Create optimizer
    optimizer = create_optimizer(
        model, 
        args.lr, 
        args.weight_decay, 
        args.momentum
    )
    
    # Create learning rate scheduler
    scheduler = create_lr_scheduler(optimizer, args.epochs)
    
    # Create loss function
    loss_function = get_loss_function()
    
    # Initialize training state
    start_epoch = 0
    best_dice = 0.0
    
    # Set up mixed precision training if requested
    scaler = torch.cuda.amp.GradScaler() if args.amp and torch.cuda.is_available() else None
    
    # Resume from checkpoint if provided
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            start_epoch = checkpoint["epoch"] + 1
            best_dice = checkpoint["best_dice"]
            
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            print(f"Resumed training from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}, starting from scratch")
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs")
    
    # Create log file
    log_file = output_dir / "training_log.csv"
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,val_loss,dice_background,dice_cat,dice_dog,dice_mean_foreground,learning_rate\n")
    
    # Main training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch
        train_loss = train_one_epoch(
            model, 
            train_loader, 
            optimizer, 
            loss_function, 
            device, 
            scaler
        )
        
        # Validate
        val_loss, dice_scores = validate(
            model, 
            val_loader, 
            loss_function, 
            device
        )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Step the scheduler
        scheduler.step()
        
        # Log metrics
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Dice Scores:")
        print(f"    Background: {dice_scores['background']:.4f}")
        print(f"    Cat: {dice_scores['cat']:.4f}")
        print(f"    Dog: {dice_scores['dog']:.4f}")
        print(f"    Mean Foreground: {dice_scores['mean_foreground']:.4f}")
        print(f"  Learning Rate: {current_lr:.7f}")
        
        # Log to file
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},"
                   f"{dice_scores['background']:.6f},{dice_scores['cat']:.6f},"
                   f"{dice_scores['dog']:.6f},{dice_scores['mean_foreground']:.6f},"
                   f"{current_lr:.7f}\n")
        
        # Check if this is the best model so far
        is_best = dice_scores['mean_foreground'] > best_dice
        if is_best:
            best_dice = dice_scores['mean_foreground']
            print(f"  New best model with mean foreground Dice: {best_dice:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or is_best:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch + 1,
                best_dice,
                output_dir,
                is_best
            )
    
    print("Training complete!")
    print(f"Best mean foreground Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()