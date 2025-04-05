"""
Metrics utility functions for image reconstruction evaluation.

This module provides functions for calculating evaluation metrics for image reconstruction,
including PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index),
and MS-SSIM (Multi-Scale Structural Similarity Index).
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional


def calculate_psnr(
    pred: torch.Tensor, 
    target: torch.Tensor,
    max_val: float = 1.0
) -> torch.Tensor:
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between predictions and targets.
    
    Args:
        pred: Predicted images, tensor of shape (B, C, H, W)
        target: Target images, tensor of shape (B, C, H, W)
        max_val: Maximum value of the signal
        
    Returns:
        PSNR values for each image in the batch
    """
    # Calculate MSE
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    
    # Avoid division by zero
    mse = torch.clamp(mse, min=1e-10)
    
    # Calculate PSNR
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    
    return psnr


def gaussian_kernel(size: int, sigma: float, channels: int = 1) -> torch.Tensor:
    """
    Create a Gaussian kernel for SSIM calculation.
    
    Args:
        size: Kernel size (should be odd)
        sigma: Standard deviation
        channels: Number of channels
        
    Returns:
        Gaussian kernel
    """
    # Ensure size is odd
    if size % 2 == 0:
        size += 1
    
    # Create 1D Gaussian kernel
    coords = torch.arange(size).float() - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    
    # Create 2D Gaussian kernel
    g_2d = g.view(1, -1) * g.view(-1, 1)
    
    # Expand to match channels
    g_2d = g_2d.view(1, 1, size, size)
    g_2d = g_2d.repeat(channels, 1, 1, 1)
    
    return g_2d


def calculate_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
    max_val: float = 1.0,
    reduction: str = 'none'
) -> torch.Tensor:
    """
    Calculate SSIM (Structural Similarity Index) between predictions and targets.
    
    Args:
        pred: Predicted images, tensor of shape (B, C, H, W)
        target: Target images, tensor of shape (B, C, H, W)
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation of the Gaussian kernel
        max_val: Maximum value of the signal
        reduction: How to reduce across batch ('none', 'mean', 'sum')
        
    Returns:
        SSIM values
    """
    # Get shapes
    B, C, H, W = pred.shape
    
    # Create Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma, 1).to(pred.device)
    
    # Prepare inputs
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    
    # Convert to single channel for pooling
    pred_flat = pred.reshape(B * C, 1, H, W)
    target_flat = target.reshape(B * C, 1, H, W)
    
    # Calculate means using convolution
    mu_pred = F.conv2d(pred_flat, kernel, padding=kernel_size//2, groups=1)
    mu_target = F.conv2d(target_flat, kernel, padding=kernel_size//2, groups=1)
    
    # Calculate variances and covariance
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    var_pred = F.conv2d(pred_flat ** 2, kernel, padding=kernel_size//2, groups=1) - mu_pred_sq
    var_target = F.conv2d(target_flat ** 2, kernel, padding=kernel_size//2, groups=1) - mu_target_sq
    cov_pred_target = F.conv2d(pred_flat * target_flat, kernel, padding=kernel_size//2, groups=1) - mu_pred_target
    
    # Calculate SSIM
    ssim_map = ((2 * mu_pred_target + c1) * (2 * cov_pred_target + c2)) / \
               ((mu_pred_sq + mu_target_sq + c1) * (var_pred + var_target + c2))
    
    # Reshape back to original dimensions
    ssim_map = ssim_map.reshape(B, C, H, W)
    
    # Reduce along spatial dimensions
    ssim_per_channel = ssim_map.mean(dim=(2, 3))
    
    # Average across channels
    ssim_per_image = ssim_per_channel.mean(dim=1)
    
    # Apply batch reduction if requested
    if reduction == 'mean':
        return ssim_per_image.mean()
    elif reduction == 'sum':
        return ssim_per_image.sum()
    else:  # 'none'
        return ssim_per_image


def evaluate_reconstructions(
    pred: torch.Tensor,
    target: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Evaluate reconstruction quality with multiple metrics.
    
    Args:
        pred: Predicted images, tensor of shape (B, C, H, W)
        target: Target images, tensor of shape (B, C, H, W)
        
    Returns:
        Dictionary of metrics
    """
    # Calculate PSNR
    psnr = calculate_psnr(pred, target)
    
    # Calculate SSIM
    ssim = calculate_ssim(pred, target)
    
    # Calculate MSE
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    
    # Create metrics dictionary
    metrics = {
        "psnr": psnr,
        "ssim": ssim,
        "mse": mse
    }
    
    return metrics