"""
UNet implementation using segmentation-models-pytorch 
with hyperparameters guided by nnU-Net
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from PIL import Image


class ModelConfig:
    """Configuration class to store model hyperparameters."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration with default values or from a config file.
        
        Args:
            config_path: Optional path to a YAML configuration file
        """
        # Default values based on typical nnU-Net configurations
        self.model_name = "unet"
        self.encoder_name = "resnet34"  # nnU-Net typically uses a custom encoder, but resnet is a good default
        self.encoder_weights = "imagenet"  # Pretrained weights
        self.in_channels = 3  # RGB images
        self.classes = 3  # background, cat, dog
        
        # Training params
        self.batch_size = 16  # Will be adjusted based on nnU-Net
        self.patch_size = (512, 512)  # Will be adjusted based on nnU-Net
        self.num_epochs = 1000
        self.learning_rate = 1e-4  # Will be adjusted based on nnU-Net
        self.optimizer = "Adam"
        self.loss = "dice"  # Will be adjusted based on nnU-Net
        self.weight_decay = 3e-5
        
        # Augmentation params - we'll use our existing augmentation config
        self.use_augmentation = True
        
        # Load from file if provided
        if config_path is not None:
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
    
    def save_to_file(self, config_path: str) -> None:
        """
        Save the current configuration to a YAML file.
        
        Args:
            config_path: Path to save the YAML configuration
        """
        # Create a dictionary of all attributes
        config_dict = {key: value for key, value in self.__dict__.items()}
        
        # Save to file
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


class PetSegmentationDataset(Dataset):
    """Dataset class for the Oxford-IIIT Pet segmentation dataset."""
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Optional[A.Compose] = None,
        is_test: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            images_dir: Directory containing images
            masks_dir: Directory containing mask annotations
            transform: Albumentations transforms to apply
            is_test: Whether this is a test dataset
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.is_test = is_test
        
        # Get a list of all image files
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        
        # Ensure there are files in the directory
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {images_dir}")
    
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
        
        # Get corresponding mask file path
        mask_path = self.masks_dir / f"{img_path.stem}.png"
        
        # Load image and mask
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if not self.is_test:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # For test set, create a dummy mask if not available
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        return {"image": image, "mask": mask}


class UNetModel:
    """UNet model wrapper using segmentation-models-pytorch."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the UNet model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        
        # Initialize model
        self.model = self._create_model()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = self.model.to(self.device)
        
        # Initialize loss function
        self.loss_fn = self._create_loss()
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
    
    def _create_model(self) -> nn.Module:
        """
        Create and return the UNet model.
        
        Returns:
            PyTorch model
        """
        return smp.Unet(
            encoder_name=self.config.encoder_name,
            encoder_weights=self.config.encoder_weights,
            in_channels=self.config.in_channels,
            classes=self.config.classes,
            activation=None  # We'll apply softmax in the loss function
        )
    
    def _create_loss(self) -> nn.Module:
        """
        Create and return the loss function.
        
        Returns:
            PyTorch loss function
        """
        if self.config.loss == "dice":
            return smp.losses.DiceLoss(mode='multiclass')
        elif self.config.loss == "ce":
            return nn.CrossEntropyLoss()
        elif self.config.loss == "combined":
            return smp.losses.DiceLoss(mode='multiclass') + nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.config.loss}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create and return the optimizer.
        
        Returns:
            PyTorch optimizer
        """
        if self.config.optimizer == "Adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "SGD":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def save_checkpoint(self, path: str, epoch: int, best_metric: float) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint
            epoch: Current epoch
            best_metric: Best validation metric achieved so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': best_metric,
            'config': self.config.__dict__
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Tuple[int, float]:
        """
        Load model checkpoint.
        
        Args:
            path: Path to the checkpoint file
            
        Returns:
            Tuple of (epoch, best_metric)
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['best_metric']
    
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
        
        for batch in train_loader:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.loss_fn(outputs, masks)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
        
        return {'loss': running_loss / len(train_loader)}
    
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
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
                
                running_loss += loss.item()
        
        return {'val_loss': running_loss / len(val_loader)}
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Make prediction for a single image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Prediction mask as numpy array (H, W)
        """
        self.model.eval()
        
        # Preprocess the image
        transform = A.Compose([
            A.Resize(height=self.config.patch_size[0], width=self.config.patch_size[1]),
            A.Normalize(),
            ToTensorV2()
        ])
        
        transformed = transform(image=image)
        image_tensor = transformed["image"].unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            output = torch.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        # Resize back to original size if needed
        if image.shape[:2] != prediction.shape:
            prediction = cv2.resize(
                prediction.astype(np.uint8), 
                (image.shape[1], image.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        
        return prediction


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


def train_model(
    model: UNetModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str,
    num_epochs: int = None
) -> None:
    """
    Train the model.
    
    Args:
        model: UNetModel instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        output_dir: Directory to save checkpoints
        num_epochs: Number of epochs to train (overrides config if provided)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get number of epochs
    if num_epochs is None:
        num_epochs = model.config.num_epochs
    
    # Initialize training variables
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        train_metrics = model.train_epoch(train_loader)
        val_metrics = model.validate(val_loader)
        
        # Update learning rate
        model.scheduler.step(val_metrics['val_loss'])
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            model.save_checkpoint(
                os.path.join(output_dir, 'best_model.pth'),
                epoch,
                best_val_loss
            )
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.save_checkpoint(
                os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                epoch,
                best_val_loss
            )
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['val_loss']:.4f}")
    
    # Save final model
    model.save_checkpoint(
        os.path.join(output_dir, 'final_model.pth'),
        num_epochs - 1,
        best_val_loss
    )