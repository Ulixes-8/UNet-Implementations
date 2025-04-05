#!/usr/bin/env python
"""
Script: train.py

This script trains a UNet model for pet segmentation using the Oxford-IIIT Pet Dataset.
It handles the training loop, validation, checkpointing, and logging.

Example Usage:
    python src/train.py --data_dir data/processed --output_dir models/unet_pet_segmentation
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
from models.losses import SimpleLoss
from models.unet import UNet


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with additional loss function parameters."""
    parser = argparse.ArgumentParser(
        description="Train a UNet model for pet segmentation with anti-overfitting measures"
    )

    parser.add_argument(
        "--pretrained_encoder",
        type=str,
        default="",
        help="Path to pre-trained autoencoder encoder weights"
    )
    
    # Original arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
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
        default=32,  # Increased from 12 to 32
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,  # Reduced from 1000 to 200
        help="Maximum number of training epochs (early stopping may reduce this)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,  # Reduced from 0.01 to 0.005
        help="Initial learning rate"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,  # Increased from 3e-5 to 1e-4 for stronger regularization
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
        default=10,  # Save more frequently (changed from 50 to 10)
        help="Save a checkpoint every N epochs"
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Patience for early stopping (number of epochs without improvement)"
    )
    
    parser.add_argument(
        "--reduced_complexity",
        action="store_true",
        default=True,  # Enable by default
        help="Use reduced complexity model to prevent overfitting"
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
    
    # New arguments for loss function
    parser.add_argument(
        "--weighted_ce",
        action="store_true",
        default=True,  # Enable by default
        help="Use class weighting in cross entropy loss based on inverse frequency"
    )
    
    parser.add_argument(
        "--static_weights",
        action="store_true",
        help="Compute class weights once at the beginning instead of dynamically per batch"
    )
    
    parser.add_argument(
        "--dice_weight",
        type=float,
        default=1.0,
        help="Weight for the Dice loss component"
    )
    
    parser.add_argument(
        "--ce_weight",
        type=float,
        default=1.0,
        help="Weight for the Cross Entropy loss component"
    )
    
    return parser.parse_args()

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Stops training when a monitored metric has stopped improving.
    
    Args:
        patience: Number of epochs with no improvement after which training will be stopped
        mode: One of {'min', 'max'}. 'min' -> monitored metric should decrease, 'max' -> increase
        min_delta: Minimum change in monitored metric to qualify as improvement
        verbose: Whether to print messages
    """
    def __init__(self, patience=10, mode='max', min_delta=0.001, verbose=True):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = float('inf') if mode == 'min' else -float('inf')
    
    def __call__(self, val_score):
        score = -val_score if self.mode == 'min' else val_score

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            
        return self.early_stop
    
class PetSegmentationDataset(Dataset):
    """Dataset class for the Oxford-IIIT Pet segmentation dataset."""
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        include_augmented: bool = True,
        target_size: Tuple[int, int] = (512, 512)
    ):
        """
        Initialize the dataset.
        
        Args:
            images_dir: Directory containing images
            masks_dir: Directory containing mask annotations
            include_augmented: Whether to include augmented images (if available)
            target_size: Target size for images and masks (height, width)
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.target_size = target_size
        
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
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask: {mask_path}")
                
            # Store original dimensions before resizing
            original_dims = mask.shape[:2]  # (height, width)
        except Exception as e:
            print(f"Error loading image or mask: {e}")
            # Return a blank sample as fallback
            image = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            mask = np.zeros(self.target_size, dtype=np.uint8)
            original_dims = self.target_size
        
        # Ensure image and mask have the correct dimensions
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)
        
        if mask.shape != self.target_size:
            mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # Keep 255 as is - we'll handle it properly in the loss function
        # Just ensure other values are within valid range (0, 1, 2)
        mask = np.where((mask > 2) & (mask != 255), 0, mask)
        
        # Convert image to tensor and normalize (0-1)
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        
        # Apply standardization (approximately equivalent to ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        # Convert mask to tensor
        mask = torch.from_numpy(mask).long()
        
        # Store original dimensions as tensor
        original_dims = torch.tensor(original_dims)
        
        return {
            "image": image, 
            "mask": mask,
            "original_dims": original_dims
        }


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
    
    # Target size for all images and masks
    target_size = (512, 512)
    
    # Create datasets
    train_dataset = PetSegmentationDataset(
        images_dir=train_imgs_dir,
        masks_dir=train_masks_dir,
        include_augmented=True,
        target_size=target_size
    )
    
    val_dataset = PetSegmentationDataset(
        images_dir=val_imgs_dir,
        masks_dir=val_masks_dir,
        include_augmented=False,
        target_size=target_size
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders with a custom worker init function
    def worker_init_fn(worker_id):
        # Set a unique seed for each worker to ensure reproducibility
        np.random.seed(np.random.get_state()[1][0] + worker_id)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # To ensure consistent batch sizes for batch norm
        worker_init_fn=worker_init_fn,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    return train_loader, val_loader

def create_model(device: torch.device, pretrained_encoder_path: Optional[str] = None) -> nn.Module:
    """
    Create the UNet model with optional pre-trained encoder.
    
    Args:
        device: Device to use for the model
        pretrained_encoder_path: Path to pre-trained encoder weights
        
    Returns:
        Initialized model
    """
    # Create UNet with pre-trained encoder option
    model = UNet(
        in_channels=3,  # RGB images
        num_classes=3,  # background, cat, dog
        n_stages=6,     # Reduced from 8 to 6
        features_per_stage=[32, 64, 128, 256, 512, 512],  # Reduced complexity
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
        # Add spatial dropout with increasing rates in encoder
        encoder_dropout_rates=[0.0, 0.0, 0.1, 0.2, 0.3, 0.3],
        # Add decreasing rates in decoder
        decoder_dropout_rates=[0.3, 0.2, 0.2, 0.1, 0.0],
        # Add pre-trained encoder path
        pretrained_encoder_path=pretrained_encoder_path
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


def get_loss_function(use_weighted_ce=True, dynamic_weights=True):
    """
    Create a loss function with optional class weighting based on inverse frequency.
    
    Args:
        use_weighted_ce: Whether to use class weighting for cross entropy
        dynamic_weights: Whether to compute class weights dynamically for each batch
        
    Returns:
        Loss function instance
    """
    if use_weighted_ce:
        # Initialize with dynamic weights (will be computed during forward pass)
        return SimpleLoss(
            weight_dice=1.0,
            weight_ce=1.0,
            ignore_index=255,
            smooth=1e-5,
            class_weights=None,  # Will be computed dynamically if dynamic_weights=True
            dynamic_weights=dynamic_weights
        )
    else:
        # Use original unweighted loss
        return SimpleLoss(
            weight_dice=1.0,
            weight_ce=1.0,
            ignore_index=255,
            smooth=1e-5
        )

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    ignore_label: int = 255  # Ignore border class
) -> Tuple[float, Dict[str, float]]:
    """
    Validate the model on the full validation set.
    
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
            
            # Compute loss on model outputs (before any resizing)
            loss = loss_function(outputs, masks)
            val_loss += loss.item()
            
            # Convert outputs to predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Calculate Dice scores for each class
            for cls_idx, cls_name in [(0, "background"), (1, "cat"), (2, "dog")]:
                # Create binary masks
                pred_cls = (preds == cls_idx).float()
                mask_cls = (masks == cls_idx).float()
                
                # Ignore border pixels (class 255)
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
    
    # Average metrics over all batches
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
    
    # Timing variables
    data_loading_time = 0
    forward_time = 0
    loss_calc_time = 0
    backward_time = 0
    
    start_time = time.time()
    for batch in tqdm(train_loader, desc="Training", leave=False):
        # Time data loading
        data_time = time.time() - start_time
        data_loading_time += data_time
        
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Time forward pass
        forward_start = time.time()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                
                # Time loss calculation
                loss_start = time.time()
                loss = loss_function(outputs, masks)
                loss_calc_time += time.time() - loss_start
                
            # Time backward pass
            backward_start = time.time()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            backward_time += time.time() - backward_start
        else:
            outputs = model(images)
            
            # Time loss calculation
            loss_start = time.time()
            loss = loss_function(outputs, masks)
            loss_calc_time += time.time() - loss_start
            
            # Time backward pass
            backward_start = time.time()
            loss.backward()
            optimizer.step()
            backward_time += time.time() - backward_start
        
        # Calculate forward time (excluding loss calculation)
        forward_time += (time.time() - forward_start) - (time.time() - loss_start)
        
        epoch_loss += loss.item()
        start_time = time.time()
    
    # Print timing information
    print(f"  Data loading time: {data_loading_time:.2f}s")
    print(f"  Forward pass time: {forward_time:.2f}s")
    print(f"  Loss calculation time: {loss_calc_time:.2f}s")
    print(f"  Backward pass time: {backward_time:.2f}s")
    print(f"  Total time: {data_loading_time + forward_time + loss_calc_time + backward_time:.2f}s")
    
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
        "best_dice": best_dice,
        "config": {
            "in_channels": 3,
            "num_classes": 3,
            "n_stages": 8,
            "features_per_stage": [32, 64, 128, 256, 512, 512, 512, 512],
            "kernel_sizes": [[3, 3]] * 8,
            "strides": [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
            "n_conv_per_stage": [2] * 8,
            "n_conv_per_stage_decoder": [2] * 7,
            "conv_bias": True,
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "nonlin_kwargs": {"inplace": True}
        }
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
    """Main training function with anti-overfitting measures and weighted loss."""
    # Parse arguments
    args = parse_args()
    
    # Override batch size to 32 (from command line)
    args.batch_size = 32
    
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
    print(f"Using mixed precision: {args.amp}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        args.data_dir, 
        args.batch_size, 
        args.num_workers
    )
    
    # Create reduced complexity model with dropout
# Create reduced complexity model with pre-trained encoder
    model = create_model(
        device, 
        pretrained_encoder_path=args.pretrained_encoder if args.pretrained_encoder else None
    )
        
    # Move model to device
    model = model.to(device)
    
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
    
    # Create loss function with class weighting
    if args.weighted_ce:
        print("Using weighted cross entropy loss based on inverse class frequency")
        
        if args.static_weights:
            print("Using static class weights computed once at start")
            # If static weights are requested, we need to compute them now
            # by scanning through the dataset
            
            # First, let's compute class frequencies across the training set
            class_pixels = torch.zeros(3, device=device)  # 3 classes: background, cat, dog
            total_valid_pixels = 0
            
            print("Computing class frequencies across training set...")
            with torch.no_grad():
                for batch in tqdm(train_loader, desc="Computing class weights"):
                    masks = batch["mask"].to(device)
                    # Create a mask for valid pixels (not ignore_index/border class)
                    valid_mask = (masks != 255)
                    valid_masks = masks * valid_mask.long()
                    
                    # Count pixels for each class
                    for c in range(3):
                        class_pixels[c] += torch.sum((valid_masks == c) & valid_mask).float()
                    
                    total_valid_pixels += torch.sum(valid_mask).float()
            
            # Compute weights: total / class_count
            # Avoid division by zero
            class_pixels = torch.clamp(class_pixels, min=1.0)
            static_weights = total_valid_pixels / class_pixels
            
            # Normalize weights to sum to num_classes
            static_weights = static_weights * (3 / static_weights.sum())
            
            print(f"Computed class weights: {static_weights.cpu().numpy()}")
            
            # Create loss with static weights
            loss_function = SimpleLoss(
                weight_dice=args.dice_weight,
                weight_ce=args.ce_weight,
                ignore_index=255,
                smooth=1e-5,
                class_weights=static_weights,
                dynamic_weights=False
            )
        else:
            print("Using dynamic class weights computed per batch")
            # Use dynamic weights computed during forward pass
            loss_function = SimpleLoss(
                weight_dice=args.dice_weight,
                weight_ce=args.ce_weight,
                ignore_index=255,
                smooth=1e-5,
                class_weights=None,
                dynamic_weights=True
            )
    else:
        print("Using unweighted loss function")
        # Use unweighted loss
        loss_function = SimpleLoss(
            weight_dice=args.dice_weight,
            weight_ce=args.ce_weight,
            ignore_index=255,
            smooth=1e-5
        )
    
    # Initialize training state
    start_epoch = 0
    best_dice = 0.0
    
    # Set up mixed precision training if requested
    # scaler = torch.amp.GradScaler(device_type='cuda') if args.amp and torch.cuda.is_available() else None
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
    
    # Set up early stopping with patience of 15 epochs
    early_stopping = EarlyStopping(patience=15, mode='max', verbose=True)
    
    # Create log file
    log_file = output_dir / "training_log.csv"
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,val_loss,dice_background,dice_cat,dice_dog,dice_mean_foreground,learning_rate,epoch_time\n")
    
    # Main training loop
    print(f"Starting training for up to {args.epochs} epochs with early stopping")
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
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
        
        # Validate on full validation set
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
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log metrics
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Dice Scores:")
        print(f"    Background: {dice_scores['background']:.4f}")
        print(f"    Cat: {dice_scores['cat']:.4f}")
        print(f"    Dog: {dice_scores['dog']:.4f}")
        print(f"    Mean Foreground: {dice_scores['mean_foreground']:.4f}")
        print(f"  Learning Rate: {current_lr:.7f}")
        print(f"  Epoch Time: {epoch_time:.2f}s")
        
        # Log to file
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},"
                   f"{dice_scores['background']:.6f},{dice_scores['cat']:.6f},"
                   f"{dice_scores['dog']:.6f},{dice_scores['mean_foreground']:.6f},"
                   f"{current_lr:.7f},{epoch_time:.2f}\n")
        
        # Check if this is the best model so far
        is_best = dice_scores['mean_foreground'] > best_dice
        if is_best:
            best_dice = dice_scores['mean_foreground']
            print(f"  New best model with mean foreground Dice: {best_dice:.4f}")
        
        # Save checkpoint more frequently (every 10 epochs)
        if (epoch + 1) % 10 == 0 or is_best:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch + 1,
                best_dice,
                output_dir,
                is_best
            )
        
        # Check for early stopping using mean foreground Dice
        if early_stopping(dice_scores['mean_foreground']):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print("Training complete!")
    print(f"Best mean foreground Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()