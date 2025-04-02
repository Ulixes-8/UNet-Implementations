#!/usr/bin/env python
"""
Script: evaluate.py

This script evaluates a trained autoencoder model on a test dataset of pet images.
It computes quantitative metrics (PSNR, SSIM, MSE) and generates visualizations
comparing original images with their reconstructions.

Example Usage:
    python AE-pretrained/evaluate.py --model_path AE-pretrained/models/ae_pet_reconstruction/best_model.pth --data_dir data/processed
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add the project root to the Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import local modules from AE-pretrained
from models.autoencoder import Autoencoder
from utils.metrics import calculate_psnr, calculate_ssim, evaluate_reconstructions
from utils.visualize import (
    denormalize_image,
    create_comparison_image,
    save_comparison_grid,
    visualize_latent_space,
    plot_reconstruction_metrics
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for autoencoder evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained autoencoder for image reconstruction"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/ulixes/segmentation_cv/unet/AE-pretrained/models/ae_pet_reconstruction/best_model.pth",
        help="Path to the trained model checkpoint"
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
        default="evaluation_results",
        help="Path to store evaluation results"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use (cuda:0, cuda:1, etc. or empty for automatic)"
    )
    
    parser.add_argument(
        "--visualize_samples",
        type=int,
        default=16,
        help="Number of sample images to visualize"
    )
    
    parser.add_argument(
        "--analyze_latent_space",
        action="store_true",
        help="Whether to perform latent space analysis"
    )
    
    return parser.parse_args()

class PetReconstructionDataset(Dataset):
    """Dataset class for the Oxford-IIIT Pet image reconstruction task, with 3-class label retrieval."""
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        target_size: Tuple[int, int] = (512, 512)
    ):
        """
        Initialize the dataset.
        
        Args:
            images_dir: Directory containing images.
            masks_dir: Directory containing segmentation masks with 0=bg, 1=cat, 2=dog.
            target_size: Target size for images (height, width).
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.target_size = target_size
        
        # Get all image files from the directory
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        
        # For simplicity, we'll assume the mask filename matches the image filename except extension
        # E.g., "IMG_0001.jpg" -> "IMG_0001.png"
        # Adjust as necessary for your environment
        self.mask_files = [
            self.masks_dir / (img_file.stem + ".png") for img_file in self.image_files
        ]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve.
            
        Returns:
            Dict containing:
                - 'image': torch.Tensor of shape [C,H,W] in [0,1]
                - 'target': same as 'image' (for autoencoder)
                - 'original_dims': tensor of the original (height, width)
                - 'filename': file name (str)
                - 'label': an integer 0,1,2 representing background, cat, or dog
        """
        # Image file
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else np.zeros((512,512,3), dtype=np.uint8)
        original_dims = image.shape[:2]

        # Resize if needed
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)

        # Convert to tensor and normalize (0-1)
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

        # Try to load the corresponding mask
        mask_path = self.mask_files[idx]
        label_val = 0  # default to background
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            # If you want a single label for the entire image:
            # We'll assume the dominant (or max) class is the label
            label_val = int(mask.max())  # 0, 1, or 2
        except:
            pass  # mask might not exist or load failed; handle gracefully

        return {
            "image": image,
            "target": image,  # autoencoder: input == target
            "original_dims": torch.tensor(original_dims),
            "filename": img_path.name,
            "label": torch.tensor(label_val, dtype=torch.long),  
        }


def create_dataloader(
    data_dir: Union[str, Path],
    batch_size: int,
    num_workers: int
) -> DataLoader:
    """
    Create test dataloader.
    """
    test_imgs_dir = Path(data_dir) / "Test" / "resized"
    test_masks_dir = Path(data_dir) / "Test" / "masks"  # or wherever your test masks are stored
    target_size = (512, 512)
    
    test_dataset = PetReconstructionDataset(
        images_dir=test_imgs_dir,
        masks_dir=test_masks_dir,
        target_size=target_size
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return test_loader

def load_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Load the trained autoencoder model from a checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model onto
        
    Returns:
        Loaded model
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration from checkpoint
    config = checkpoint.get("config", {})
    
    # Create the model with the same architecture used during training
    model = Autoencoder(
        in_channels=config.get("in_channels", 3),
        out_channels=config.get("out_channels", 3),
        n_stages=config.get("n_stages", 6),
        features_per_stage=config.get("features_per_stage", [32, 64, 128, 256, 512, 512]),
        kernel_sizes=config.get("kernel_sizes", [[3, 3]] * 6),
        strides=config.get("strides", [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]),
        n_conv_per_stage=config.get("n_conv_per_stage", [2] * 6),
        n_conv_per_stage_decoder=config.get("n_conv_per_stage_decoder", [2] * 5),
        conv_bias=config.get("conv_bias", True),
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs=config.get("norm_op_kwargs", {"eps": 1e-5, "affine": True}),
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=config.get("nonlin_kwargs", {"inplace": True})
    )
    
    # Load model state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Move to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_reconstruction_quality(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    visualize_samples: int = 16
) -> Dict:
    """
    Evaluate reconstruction quality of the autoencoder.
    
    Args:
        model: Trained autoencoder model
        test_loader: Test data loader
        device: Device to run the model on
        output_dir: Directory to save results
        visualize_samples: Number of samples to visualize
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Create directory for visualizations
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize metrics
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    
    # Lists to store original and reconstructed images for visualization
    all_originals = []
    all_reconstructions = []
    all_filenames = []
    
    # Number of processed samples
    num_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Extract data
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            filenames = batch["filename"]
            
            # Forward pass
            reconstructions = model(images)
            
            # Compute MSE per image
            mse = ((reconstructions - targets) ** 2).mean(dim=(1, 2, 3))
            total_mse += mse.sum().item()
            
            # Compute PSNR per image
            psnr = calculate_psnr(reconstructions, targets)
            total_psnr += psnr.sum().item()
            
            # Compute SSIM per image
            ssim = calculate_ssim(reconstructions, targets)
            total_ssim += ssim.sum().item()
            
            # Update the total number of samples
            num_samples += images.size(0)
            
            # Store original and reconstructed images for later visualizations
            if batch_idx == 0 or (visualize_samples > 0 and len(all_originals) < visualize_samples):
                # Select a subset of images for visualization
                num_to_visualize = min(images.size(0), visualize_samples - len(all_originals))
                
                # Store images
                all_originals.extend([denormalize_image(img) for img in images[:num_to_visualize].cpu()])
                all_reconstructions.extend([denormalize_image(recon) for recon in reconstructions[:num_to_visualize].cpu()])
                all_filenames.extend(filenames[:num_to_visualize])
    
    # Calculate average metrics
    avg_mse = total_mse / num_samples
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    # Save side-by-side comparison of original and reconstructed images
    if visualize_samples > 0 and all_originals:
        # Create a grid of comparisons
        grid_path = vis_dir / "reconstruction_grid.png"
        
        # Convert to torch tensors for the grid function
        original_tensors = [torch.from_numpy(img.transpose(2, 0, 1) / 255.0) for img in all_originals]
        recon_tensors = [torch.from_numpy(img.transpose(2, 0, 1) / 255.0) for img in all_reconstructions]
        
        # Save as a grid
        save_comparison_grid(
            torch.stack(original_tensors),
            torch.stack(recon_tensors),
            str(grid_path),
            n_rows=min(4, len(original_tensors)),
            title="Original vs Reconstructed"
        )
        
        # Save individual comparisons
        for i, (original, recon, filename) in enumerate(zip(all_originals, all_reconstructions, all_filenames)):
            comparison = create_comparison_image(original, recon, error_map=True)
            cv2.imwrite(str(vis_dir / f"comparison_{i}_{filename}"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    # Return metrics
    metrics = {
        "mse": avg_mse,
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "num_samples": num_samples
    }
    
    return metrics

def analyze_latent_space(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path
) -> None:
    """
    Analyze the latent space of the autoencoder, grouping by labels (0,1,2).
    """
    # Create directory for latent space analysis
    latent_dir = output_dir / "latent_analysis"
    latent_dir.mkdir(parents=True, exist_ok=True)
    
    all_latents = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting Latents for Analysis"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)      # shape [B]
            
            # 1) Encode the images to get latents
            #    Adjust this line according to how your Autoencoder is structured.
            #    If your model doesn’t have a direct 'encode' method, you might do:
            #       z = model.encoder(images)
            #    or something like:
            #       z, _ = model.encode(images)
            #    Make sure you get the correct latent shape [B, latent_dim].
            latents = model.encode(images)
            
            all_latents.append(latents.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all latents and labels
    all_latents = torch.cat(all_latents, dim=0)  # shape [N, latent_dim]
    all_labels = torch.cat(all_labels, dim=0)    # shape [N]
    
    # Convert to numpy
    all_latents_np = all_latents.numpy()
    all_labels_np = all_labels.numpy()
    
    print("Visualizing latent space with PCA and t-SNE, color-coded by label...")

    # 2) PCA visualization
    visualize_latent_space(
        all_latents_np,
        all_labels_np,
        str(latent_dir / "latent_space_pca.png"),
        method='pca'
    )
    
    # 3) t-SNE visualization
    visualize_latent_space(
        all_latents_np,
        all_labels_np,
        str(latent_dir / "latent_space_tsne.png"),
        method='tsne'
    )
    
    print(f"Latent space visualizations saved to {latent_dir}")


def main() -> None:
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, device)
    print("Model loaded successfully")
    
    # Create test dataloader
    test_loader = create_dataloader(
        args.data_dir,
        args.batch_size,
        args.num_workers
    )
    
    # Evaluate reconstruction quality
    print("Evaluating reconstruction quality...")
    metrics = evaluate_reconstruction_quality(
        model,
        test_loader,
        device,
        output_dir,
        args.visualize_samples
    )
    
    # Print and save metrics
    print("\n=== Reconstruction Quality Metrics ===")
    print(f"Mean Squared Error (MSE): {metrics['mse']:.6f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {metrics['psnr']:.2f} dB")
    print(f"Structural Similarity Index (SSIM): {metrics['ssim']:.4f}")
    print(f"Number of samples: {metrics['num_samples']}")
    
    # Save metrics to JSON
    with open(output_dir / "reconstruction_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {output_dir / 'reconstruction_metrics.json'}")
    
    # Analyze latent space if requested
    if args.analyze_latent_space:
        print("\n=== Analyzing Latent Space ===")
        analyze_latent_space(model, test_loader, device, output_dir)
    
    # Plot training log if available
    training_log_path = Path(args.model_path).parent / "training_log.csv"
    if training_log_path.exists():
        print("\n=== Plotting Training Metrics ===")
        plot_reconstruction_metrics(
            str(training_log_path),
            str(output_dir)
        )
        print(f"Training plots saved to {output_dir / 'plots'}")


    if args.analyze_latent_space:
        print("\n=== Analyzing Latent Space ===")
        analyze_latent_space(model, test_loader, device, output_dir)

    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()