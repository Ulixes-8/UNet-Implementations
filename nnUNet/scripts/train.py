#!/usr/bin/env python
"""
Script: train.py

This script trains a UNet model for pet segmentation using hyperparameters
optimized by nnU-Net. It leverages segmentation-models-pytorch for the model
implementation and supports various training configurations.

Features:
- Configurable model architecture (encoder, decoder)
- Learning rate scheduling with early stopping
- Mixed precision training for faster computation
- Validation metrics with Dice coefficient
- TensorBoard logging for monitoring training progress
- Checkpoint saving and resuming

Example Usage:
    python train.py \
        --config config/model_config.yaml \
        --data_dir data/processed \
        --output_dir models/unet_pet_segmentation \
        --num_workers 4
"""

import os
import sys
import time
import yaml
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
from PIL import Image


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Optional path to log file
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("train")
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
        "--log_dir",
        type=str,
        default=None,
        help="Directory to save training logs (defaults to output_dir/logs)"
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
    
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Patience for early stopping (epochs without improvement)"
    )
    
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Use mixed precision training (faster on compatible GPUs)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (less data, faster iterations)"
    )
    
    return parser.parse_args()


class ModelConfig:
    """Configuration class to store model hyperparameters."""
    
    def __init__(self, config_path: str):
        """
        Initialize the configuration from a YAML file.
        
        Args:
            config_path: Path to a YAML configuration file
        """
        self.model_name = "unet"
        self.encoder_name = "resnet34"
        self.encoder_weights = "imagenet"
        self.in_channels = 3
        self.classes = 3
        
        # Training params
        self.batch_size = 16
        self.patch_size = [512, 512]
        self.num_epochs = 1000
        self.learning_rate = 1e-4
        self.optimizer = "Adam"
        self.loss = "combined"
        self.weight_decay = 3e-5
        
        # Augmentation params
        self.use_augmentation = True
        
        # Load from file
        self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to a YAML configuration file
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class PetSegmentationDataset(Dataset):
    """Dataset class for the Oxford-IIIT Pet segmentation dataset."""
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Optional[A.Compose] = None,
        include_augmented: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            images_dir: Directory containing images
            masks_dir: Directory containing mask annotations
            transform: Albumentations transforms to apply
            include_augmented: Whether to include augmented images (if available)
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        
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
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        return {"image": image, "mask": mask}


def create_transforms(config: ModelConfig, is_train: bool = True) -> A.Compose:
    """
    Create Albumentations transforms.
    
    Args:
        config: Model configuration
        is_train: Whether to create transforms for training or validation
        
    Returns:
        Albumentations Compose object
    """
    if is_train and config.use_augmentation:
        transforms = A.Compose([
            A.Resize(height=config.patch_size[0], width=config.patch_size[1]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.15, 
                rotate_limit=15, 
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.OneOf([
                A.ElasticTransform(p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(p=0.5)
            ], p=0.3),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.3),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        transforms = A.Compose([
            A.Resize(height=config.patch_size[0], width=config.patch_size[1]),
            A.Normalize(),
            ToTensorV2()
        ])
    
    return transforms


def create_model(config: ModelConfig) -> nn.Module:
    """
    Create UNet model based on configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        PyTorch UNet model
    """
    if config.model_name.lower() == "unet":
        model = smp.Unet(
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            in_channels=config.in_channels,
            classes=config.classes,
            activation=None  # We'll apply activation in the loss function
        )
    elif config.model_name.lower() == "unet++":
        model = smp.UnetPlusPlus(
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            in_channels=config.in_channels,
            classes=config.classes,
            activation=None
        )
    elif config.model_name.lower() == "deeplabv3+":
        model = smp.DeepLabV3Plus(
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            in_channels=config.in_channels,
            classes=config.classes,
            activation=None
        )
    else:
        # Default to UNet
        model = smp.Unet(
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            in_channels=config.in_channels,
            classes=config.classes,
            activation=None
        )
    
    return model


def create_loss_function(config: ModelConfig) -> nn.Module:
    """
    Create loss function based on configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        PyTorch loss function
    """
    if config.loss == "dice":
        return smp.losses.DiceLoss(mode='multiclass')
    elif config.loss == "ce":
        return nn.CrossEntropyLoss()
    elif config.loss == "combined":
        dice_loss = smp.losses.DiceLoss(mode='multiclass')
        ce_loss = nn.CrossEntropyLoss()
        return lambda pred, target: dice_loss(pred, target) + ce_loss(pred, target)
    else:
        # Default to combined loss
        dice_loss = smp.losses.DiceLoss(mode='multiclass')
        ce_loss = nn.CrossEntropyLoss()
        return lambda pred, target: dice_loss(pred, target) + ce_loss(pred, target)


def create_optimizer(config: ModelConfig, model: nn.Module) -> optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        config: Model configuration
        model: PyTorch model
        
    Returns:
        PyTorch optimizer
    """
    if config.optimizer.lower() == "adam":
        return optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    else:
        # Default to Adam
        return optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )


def create_scheduler(optimizer: optim.Optimizer, patience: int = 10) -> optim.lr_scheduler.ReduceLROnPlateau:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        patience: Number of epochs without improvement to wait
        
    Returns:
        PyTorch learning rate scheduler
    """
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=patience,
        verbose=True
    )


class Trainer:
    """Class for training and validating a UNet model."""
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.ReduceLROnPlateau,
        device: torch.device,
        output_dir: str,
        mixed_precision: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model
            loss_fn: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            output_dir: Directory to save checkpoints and logs
            mixed_precision: Whether to use mixed precision training
            logger: Logger for output
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.mixed_precision = mixed_precision
        self.logger = logger
        
        # Create directories
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None
        
        # Initialize training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Dict with training metrics
        """
        self.model.train()
        running_loss = 0.0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, masks)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
                
                loss.backward()
                self.optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})
        
        # Calculate average loss
        avg_loss = running_loss / len(train_loader)
        
        return {"loss": avg_loss}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        running_loss = 0.0
        running_dice = 0.0
        num_samples = 0
        
        # Progress bar
        pbar = tqdm(val_loader, desc=f"Epoch {self.epoch+1} [Valid]")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
                
                # Calculate Dice coefficient for each class and sample
                preds = torch.argmax(outputs, dim=1)
                
                # Calculate average Dice coefficient (exclude background)
                dice_coeffs = []
                for c in range(1, outputs.shape[1]):  # Skip background class
                    dice_class = self.calculate_dice(preds == c, masks == c)
                    dice_coeffs.append(dice_class)
                
                avg_dice = sum(dice_coeffs) / len(dice_coeffs)
                
                # Update statistics
                running_loss += loss.item() * images.size(0)
                running_dice += avg_dice.item() * images.size(0)
                num_samples += images.size(0)
                
                # Update progress bar
                pbar.set_postfix({"loss": loss.item(), "dice": avg_dice.item()})
        
        # Calculate average metrics
        avg_loss = running_loss / num_samples
        avg_dice = running_dice / num_samples
        
        return {"val_loss": avg_loss, "val_dice": avg_dice}
    
    @staticmethod
    def calculate_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """
        Calculate Dice coefficient.
        
        Args:
            pred: Prediction tensor
            target: Target tensor
            smooth: Smoothing factor
            
        Returns:
            Dice coefficient
        """
        intersection = (pred & target).float().sum((1, 2))
        union = pred.float().sum((1, 2)) + target.float().sum((1, 2))
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.mean()
    
    def save_checkpoint(self, is_best: bool = False) -> str:
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
            
        Returns:
            Path to the saved checkpoint
        """
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            if self.logger:
                self.logger.info(f"Saved best model to {best_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.patience_counter = checkpoint.get("patience_counter", 0)
        
        if self.logger:
            self.logger.info(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        patience: int = 20
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train
            patience: Patience for early stopping
            
        Returns:
            Dict with training history
        """
        start_epoch = self.epoch
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "learning_rates": []
        }
        
        if self.logger:
            self.logger.info(f"Starting training from epoch {start_epoch + 1}")
            self.logger.info(f"Training on device: {self.device}")
            self.logger.info(f"Mixed precision: {self.mixed_precision}")
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            history["learning_rates"].append(current_lr)
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader)
            history["train_loss"].append(train_metrics["loss"])
            
            # Validate
            val_metrics = self.validate(val_loader)
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_dice"].append(val_metrics["val_dice"])
            
            # Update learning rate
            self.scheduler.step(val_metrics["val_loss"])
            
            # Check if this is the best model
            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0 or epoch == start_epoch + num_epochs - 1:
                self.save_checkpoint()
            
            # Log metrics
            epoch_time = time.time() - epoch_start_time
            if self.logger:
                self.logger.info(
                    f"Epoch {epoch+1}/{start_epoch+num_epochs} "
                    f"[{epoch_time:.1f}s] - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Val Dice: {val_metrics['val_dice']:.4f}, "
                    f"LR: {current_lr:.8f}"
                )
            
            # TensorBoard logging
            self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            self.writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
            self.writer.add_scalar("Dice/val", val_metrics["val_dice"], epoch)
            self.writer.add_scalar("LR", current_lr, epoch)
            
            # Early stopping
            if self.patience_counter >= patience:
                if self.logger:
                    self.logger.info(
                        f"Early stopping triggered after {epoch+1} epochs "
                        f"({patience} epochs without improvement)"
                    )
                break
        
        # Load best model before returning
        best_model_path = self.checkpoint_dir / "best_model.pth"
        if best_model_path.exists():
            self.load_checkpoint(str(best_model_path))
        
        # Close TensorBoard writer
        self.writer.close()
        
        return history


def main() -> None:
    """Main function to train the UNet model."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    log_file = os.path.join(args.output_dir, "training.log")
    logger = setup_logging(log_file)
    
    # Load configuration
    config = ModelConfig(args.config)
    
    # Override num_epochs if provided
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    
    # Set log directory
    log_dir = args.log_dir if args.log_dir else os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Print configuration
    logger.info("Model Configuration:")
    for key, value in config.__dict__.items():
        logger.info(f"  {key}: {value}")
    
    # Setup paths
    data_dir = Path(args.data_dir)
    train_imgs_dir = data_dir / "Train" / "resized"
    train_masks_dir = data_dir / "Train" / "resized_label"
    val_imgs_dir = data_dir / "Val" / "resized"
    val_masks_dir = data_dir / "Val" / "processed_labels"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create transforms
    train_transforms = create_transforms(config, is_train=True)
    val_transforms = create_transforms(config, is_train=False)
    
    # Create datasets
    train_dataset = PetSegmentationDataset(
        images_dir=train_imgs_dir,
        masks_dir=train_masks_dir,
        transform=train_transforms,
        include_augmented=True
    )
    
    val_dataset = PetSegmentationDataset(
        images_dir=val_imgs_dir,
        masks_dir=val_masks_dir,
        transform=val_transforms,
        include_augmented=False
    )
    
    # Debug mode - use subset of data
    if args.debug:
        train_indices = np.random.choice(len(train_dataset), min(100, len(train_dataset)), replace=False)
        val_indices = np.random.choice(len(val_dataset), min(20, len(val_dataset)), replace=False)
        
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        
        logger.info(f"Debug mode: Using {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Print dataset info
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    logger.info(f"Created {config.model_name} model with {config.encoder_name} encoder")
    
    # Create loss function
    loss_fn = create_loss_function(config)
    logger.info(f"Using {config.loss} loss function")
    
    # Create optimizer
    optimizer = create_optimizer(config, model)
    logger.info(f"Using {config.optimizer} optimizer with LR={config.learning_rate}")
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, patience=5)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        mixed_precision=args.mixed_precision,
        logger=logger
    )
    
    # Resume training if checkpoint is provided
    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    logger.info(f"Starting training for {config.num_epochs} epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.num_epochs,
        patience=args.patience
    )
    
    # Save final model if not saved already
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "history": history
    }, final_model_path)
    
    logger.info(f"Training completed. Final model saved to {final_model_path}")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()