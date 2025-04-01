#!/usr/bin/env python
"""
Script: train.py

This script trains a convolutional autoencoder for image reconstruction using the Oxford-IIIT Pet Dataset.
It handles the training loop, validation, checkpointing, and logging.

Example Usage:
    python AE_pretrained/train.py --data_dir data/processed --output_dir models/ae_pet_reconstruction
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
from models.losses import ReconstructionLoss
from models.autoencoder import Autoencoder


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for autoencoder training."""
    parser = argparse.ArgumentParser(
        description="Train a convolutional autoencoder for image reconstruction"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Path to the processed data directory"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/ae_pet_reconstruction",
        help="Path to store model checkpoints and logs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum number of training epochs (early stopping may reduce this)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for optimizer"
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
        default=10,
        help="Save a checkpoint every N epochs"
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Patience for early stopping (number of epochs without improvement)"
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
    
    # Loss function arguments
    parser.add_argument(
        "--mse_weight",
        type=float,
        default=1.0,
        help="Weight for MSE loss component"
    )
    
    parser.add_argument(
        "--perceptual_weight",
        type=float,
        default=0.1,
        help="Weight for perceptual loss component"
    )
    
    parser.add_argument(
        "--ssim_weight",
        type=float,
        default=0.1,
        help="Weight for SSIM loss component"
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
    def __init__(self, patience=10, mode='min', min_delta=0.001, verbose=True):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = float('inf') if mode == 'min' else -float('inf')
    
    def __call__(self, val_score):
        score = val_score if self.mode == 'min' else -val_score

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            
        return self.early_stop
    
    
class PetReconstructionDataset(Dataset):
    """Dataset class for the Oxford-IIIT Pet image reconstruction task."""
    
    def __init__(
        self,
        images_dir: str,
        include_augmented: bool = True,
        target_size: Tuple[int, int] = (512, 512)
    ):
        """
        Initialize the dataset.
        
        Args:
            images_dir: Directory containing images
            include_augmented: Whether to include augmented images (if available)
            target_size: Target size for images (height, width)
        """
        self.images_dir = Path(images_dir)
        self.target_size = target_size
        
        # Get all image files from the directory
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        
        # Check for augmented data directory
        if include_augmented and (self.images_dir.parent / "augmented" / "images").exists():
            aug_images_dir = self.images_dir.parent / "augmented" / "images"
            
            # Add augmented images to the dataset
            aug_image_files = sorted(list(aug_images_dir.glob("*.jpg")))
            self.aug_image_files = aug_image_files
            
            self.image_files.extend(aug_image_files)
        else:
            self.aug_image_files = []
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dict containing image tensors
        """
        # Get image file path
        img_path = self.image_files[idx]
        
        # Load image
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Store original dimensions before resizing
            original_dims = image.shape[:2]  # (height, width)
        except Exception as e:
            print(f"Error loading image: {e}")
            # Return a blank sample as fallback
            image = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            original_dims = self.target_size
        
        # Ensure image has the correct dimensions
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # Convert image to tensor and normalize (0-1)
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        
        # Store original dimensions as tensor
        original_dims = torch.tensor(original_dims)
        
        # For autoencoder, input and target are the same image
        return {
            "image": image,
            "target": image,  # Target is same as input for reconstruction
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
    
    # Set directories for validation data
    val_imgs_dir = Path(data_dir) / "Val" / "resized"
    
    # Target size for all images
    target_size = (512, 512)
    
    # Create datasets
    train_dataset = PetReconstructionDataset(
        images_dir=train_imgs_dir,
        include_augmented=True,
        target_size=target_size
    )
    
    val_dataset = PetReconstructionDataset(
        images_dir=val_imgs_dir,
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


def create_model(device: torch.device) -> nn.Module:
    """
    Create the Autoencoder model.
    
    Args:
        device: Device to use for the model
        
    Returns:
        Initialized model
    """
    # Create Autoencoder with same architecture as UNet
    model = Autoencoder(
        in_channels=3,   # RGB images
        out_channels=3,  # RGB reconstructed images
        n_stages=6,      # Same architecture as UNet
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
        # Lower dropout for reconstruction task
        encoder_dropout_rates=[0.0, 0.0, 0.05, 0.1, 0.15, 0.15],
        decoder_dropout_rates=[0.15, 0.1, 0.1, 0.05, 0.0]
    )
    
    # Move model to device
    model = model.to(device)
    
    return model


def create_optimizer(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
    """
    Create an optimizer for the model.
    
    Args:
        model: The model to optimize
        lr: Learning rate
        weight_decay: Weight decay factor
        
    Returns:
        Configured optimizer
    """
    # Use Adam optimizer for autoencoder training
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
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
    # Use cosine annealing scheduler with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=1e-6
    )
    
    return scheduler


def get_loss_function(device=None):
    """
    Create a simple MSE loss function for reconstruction.
    
    Args:
        device: Device to use for the loss function
        
    Returns:
        MSE loss function
    """
    # Use simple MSE loss for autoencoder training
    loss_fn = nn.MSELoss()
    
    # Move to device if specified
    if device is not None:
        loss_fn = loss_fn.to(device)
    
    return loss_fn


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """
    Validate the model on the full validation set.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        loss_function: Loss function
        device: Device to use
        
    Returns:
        Tuple of (validation_loss, metrics_dict)
    """
    model.eval()
    
    # Initialize metrics
    val_loss = 0.0
    total_mse = 0.0
    total_psnr = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute MSE loss
            loss = loss_function(outputs, targets)
            val_loss += loss.item()
            
            # Calculate MSE per image
            mse = ((outputs - targets) ** 2).mean(dim=(1, 2, 3))
            total_mse += mse.sum().item()
            
            # Calculate PSNR per image
            psnr = 10 * torch.log10(1.0 / mse)
            total_psnr += psnr.sum().item()
    
    # Average metrics over all batches
    num_samples = len(val_loader.dataset)
    val_loss /= len(val_loader)
    avg_mse = total_mse / num_samples
    avg_psnr = total_psnr / num_samples
    
    # Return metrics
    metrics = {
        "loss": val_loss,
        "mse": avg_mse,
        "psnr": avg_psnr
    }
    
    return val_loss, metrics


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
        targets = batch["target"].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if enabled
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = loss_function(outputs, targets)
            
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = loss_function(outputs, targets)
            
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
    best_loss: float,
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
        best_loss: Best validation loss so far
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
        "best_loss": best_loss,
        "config": {
            "in_channels": 3,
            "out_channels": 3,
            "n_stages": 6,
            "features_per_stage": [32, 64, 128, 256, 512, 512],
            "kernel_sizes": [[3, 3]] * 6,
            "strides": [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
            "n_conv_per_stage": [2] * 6,
            "n_conv_per_stage_decoder": [2] * 5,
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


def save_reconstructed_images(
    model: nn.Module,
    val_loader: DataLoader,
    output_dir: Path,
    device: torch.device,
    num_images: int = 8
) -> None:
    """
    Save sample reconstructed images for visualization.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        output_dir: Directory to save images
        device: Device to use
        num_images: Number of images to save
    """
    model.eval()
    
    # Create output directory
    vis_dir = output_dir / "reconstructions"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Get a single batch
    batch = next(iter(val_loader))
    images = batch["image"].to(device)[:num_images]
    
    with torch.no_grad():
        # Forward pass
        reconstructed = model(images)
    
    # Convert to numpy and denormalize
    images = images.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    
    # Transpose from (B, C, H, W) to (B, H, W, C) and scale to [0, 255]
    images = np.transpose(images, (0, 2, 3, 1)) * 255
    reconstructed = np.transpose(reconstructed, (0, 2, 3, 1)) * 255
    
    # Save each image pair
    for i in range(min(num_images, len(images))):
        # Original image
        original = images[i].astype(np.uint8)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(vis_dir / f"original_{i}.jpg"), original_rgb)
        
        # Reconstructed image
        recon = reconstructed[i].astype(np.uint8)
        recon_rgb = cv2.cvtColor(recon, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(vis_dir / f"reconstructed_{i}.jpg"), recon_rgb)
        
        # Side-by-side comparison
        comparison = np.hstack((original, recon))
        comp_rgb = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(vis_dir / f"comparison_{i}.jpg"), comp_rgb)
    
    print(f"Saved {num_images} reconstructed images to {vis_dir}")


def main() -> None:
    """Main training function for the autoencoder."""
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
    print(f"Using mixed precision: {args.amp}")
    
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
        args.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = create_lr_scheduler(optimizer, args.epochs)
    
    # Create loss function (simple MSE for autoencoder)
    loss_function = get_loss_function(device=device)
    
    # Initialize training state
    start_epoch = 0
    best_loss = float('inf')
    
    # Set up mixed precision training if requested
    scaler = torch.amp.GradScaler(device_type='cuda') if args.amp and torch.cuda.is_available() else None
    
    # Resume from checkpoint if provided
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            start_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint["best_loss"]
            
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            print(f"Resumed training from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}, starting from scratch")
    
    # Set up early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='min', verbose=True)
    
    # Create log file
    log_file = output_dir / "training_log.csv"
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,val_loss,val_mse,val_psnr,learning_rate,epoch_time\n")
    
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
        val_loss, val_metrics = validate(
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
        print(f"  Val MSE: {val_metrics['mse']:.6f}")
        print(f"  Val PSNR: {val_metrics['psnr']:.2f} dB")
        print(f"  Learning Rate: {current_lr:.7f}")
        print(f"  Epoch Time: {epoch_time:.2f}s")
        
        # Log to file
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},"
                   f"{val_metrics['mse']:.6f},{val_metrics['psnr']:.2f},"
                   f"{current_lr:.7f},{epoch_time:.2f}\n")
        
        # Check if this is the best model so far
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            print(f"  New best model with validation loss: {best_loss:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_every == 0 or is_best:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch + 1,
                best_loss,
                output_dir,
                is_best
            )
            
            # Save sample reconstructions
            save_reconstructed_images(
                model,
                val_loader,
                output_dir,
                device
            )
        
        # Check for early stopping
        if early_stopping(val_loss):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print("Training complete!")
    print(f"Best validation loss: {best_loss:.4f}")
    
    # Save final reconstructed images
    save_reconstructed_images(
        model,
        val_loader,
        output_dir,
        device,
        num_images=16  # Save more images for final model
    )


if __name__ == "__main__":
    main()