"""
Visualization utilities for autoencoder output.

This module provides functions for visualizing autoencoder reconstructions,
including side-by-side comparisons, error maps, and grid displays.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import cv2


def denormalize_image(
    image: torch.Tensor,
    mean: List[float] = None,
    std: List[float] = None
) -> np.ndarray:
    """
    Convert a normalized tensor to a displayable numpy image.
    
    Args:
        image: Image tensor of shape (C, H, W)
        mean: Optional mean values for denormalization
        std: Optional std values for denormalization
        
    Returns:
        Numpy array of shape (H, W, C) with values in [0, 255]
    """
    # Clone the tensor to avoid modifying the original
    img = image.clone().detach()
    
    # Denormalize if mean and std are provided
    if mean is not None and std is not None:
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
        img = img * std + mean
    
    # Clamp values to [0, 1]
    img = torch.clamp(img, 0, 1)
    
    # Convert to numpy and transpose from (C, H, W) to (H, W, C)
    img_np = img.cpu().numpy().transpose(1, 2, 0)
    
    # Scale to [0, 255] and convert to uint8
    img_np = (img_np * 255).astype(np.uint8)
    
    return img_np


def create_comparison_image(
    original: np.ndarray,
    reconstructed: np.ndarray,
    error_map: bool = True
) -> np.ndarray:
    """
    Create a side-by-side comparison of original and reconstructed images.
    
    Args:
        original: Original image as numpy array (H, W, C)
        reconstructed: Reconstructed image as numpy array (H, W, C)
        error_map: Whether to include an error map visualization
        
    Returns:
        Comparison image as numpy array
    """
    # Ensure the images have the same shape
    assert original.shape == reconstructed.shape, "Images must have the same shape"
    
    if error_map:
        # Calculate pixel-wise absolute error
        diff = np.abs(original.astype(np.float32) - reconstructed.astype(np.float32))
        
        # Normalize error to [0, 255] for visualization
        if diff.max() > 0:
            diff = (diff / diff.max() * 255).astype(np.uint8)
        else:
            diff = np.zeros_like(original)
        
        # Apply color map for better visualization
        if diff.shape[-1] == 3:  # RGB image
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            diff_color = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
        else:  # Already grayscale
            diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        
        # Create side-by-side comparison with error map
        comparison = np.hstack((original, reconstructed, diff_color))
    else:
        # Create side-by-side comparison without error map
        comparison = np.hstack((original, reconstructed))
    
    return comparison


def save_comparison_grid(
    original_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    output_path: str,
    n_rows: int = 4,
    title: str = "Original vs Reconstructed"
) -> None:
    """
    Create and save a grid of original and reconstructed image pairs.
    
    Args:
        original_images: Tensor of original images (B, C, H, W)
        reconstructed_images: Tensor of reconstructed images (B, C, H, W)
        output_path: Path to save the visualization
        n_rows: Number of rows in the grid
        title: Title for the plot
    """
    # Determine number of columns based on batch size and rows
    batch_size = original_images.size(0)
    n_cols = min(batch_size, 8)  # Maximum 8 columns
    n_rows = min(n_rows, (batch_size + n_cols - 1) // n_cols)  # Adjust rows based on batch size
    
    # Create figure
    fig, axes = plt.subplots(
        n_rows, n_cols * 2, 
        figsize=(n_cols * 4, n_rows * 2.5),
        gridspec_kw={'wspace': 0.05, 'hspace': 0.2}
    )
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    
    # Denormalize and plot images
    for i in range(min(n_rows * n_cols, batch_size)):
        row = i // n_cols
        col = (i % n_cols) * 2
        
        # Check if single row
        if n_rows == 1:
            ax_orig = axes[col]
            ax_recon = axes[col + 1]
        else:
            ax_orig = axes[row, col]
            ax_recon = axes[row, col + 1]
        
        # Get original and reconstructed images
        orig = denormalize_image(original_images[i])
        recon = denormalize_image(reconstructed_images[i])
        
        # Plot original
        ax_orig.imshow(orig)
        ax_orig.set_title("Original")
        ax_orig.axis('off')
        
        # Plot reconstructed
        ax_recon.imshow(recon)
        ax_recon.set_title("Reconstructed")
        ax_recon.axis('off')
    
    # Hide empty subplots
    for i in range(batch_size, n_rows * n_cols):
        row = i // n_cols
        col = (i % n_cols) * 2
        
        if n_rows == 1:
            axes[col].axis('off')
            axes[col + 1].axis('off')
        else:
            axes[row, col].axis('off')
            axes[row, col + 1].axis('off')
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for title
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

def visualize_latent_space(
    latents: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    method: str = 'pca',
    perplexity: float = 30.0
):
    """
    Visualize latent space (2D) for an array of embeddings (N, latent_dim).
    
    Args:
        latents: shape [N, latent_dim], all latent vectors.
        labels: shape [N], integer labels (0,1,2).
        save_path: path to save the figure.
        method: 'pca' or 'tsne'.
        perplexity: used for t-SNE (hyperparameter).
    """
    # Dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'pca' or 'tsne'.")
    
    reduced = reducer.fit_transform(latents)  # shape [N,2]
    
    # Plot
    plt.figure(figsize=(8, 6))
    
    # If you have exactly 3 labels {0,1,2}, you can color them easily:
    scatter = plt.scatter(
        reduced[:, 0], 
        reduced[:, 1], 
        c=labels, 
        alpha=0.7
    )
    plt.title(f"Latent Space ({method.upper()})")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    
    # Create a legend for the 3 classes if you want
    # A quick trick is to create a legend based on unique labels
    unique_labels = np.unique(labels)
    cbar = plt.colorbar(scatter)
    cbar.set_ticks(unique_labels)
    cbar.set_ticklabels([f"Class {u}" for u in unique_labels])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[{method.upper()}] Saved 2D embedding plot to {save_path}")


def plot_reconstruction_metrics(
    log_file: str,
    output_dir: str
) -> None:
    """
    Plot training and validation metrics from log file.
    
    Args:
        log_file: Path to training log CSV file
        output_dir: Directory to save plots
    """
    import pandas as pd
    
    # Load log data
    try:
        data = pd.read_csv(log_file)
    except Exception as e:
        print(f"Error loading log file: {e}")
        return
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(data['epoch'], data['train_loss'], label='Training Loss')
    plt.plot(data['epoch'], data['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'loss_curves.png'), dpi=150)
    plt.close()
    
    # Plot MSE and PSNR
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # MSE on left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE', color=color)
    ax1.plot(data['epoch'], data['val_mse'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # PSNR on right y-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('PSNR (dB)', color=color)
    ax2.plot(data['epoch'], data['val_psnr'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Validation MSE and PSNR')
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mse_psnr_curves.png'), dpi=150)
    plt.close()
    
    # Plot learning rate
    plt.figure(figsize=(10, 6))
    plt.plot(data['epoch'], data['learning_rate'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    plt.savefig(os.path.join(plots_dir, 'learning_rate.png'), dpi=150)
    plt.close()