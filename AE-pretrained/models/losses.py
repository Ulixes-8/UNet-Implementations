"""
Loss functions for autoencoder training.
This module provides various loss functions for image reconstruction tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ReconstructionLoss(nn.Module):
    """
    Combined reconstruction loss for autoencoder training.
    This loss combines MSE, perceptual, and/or SSIM losses with configurable weights.
    """
    def __init__(
        self,
        mse_weight=1.0,
        perceptual_weight=0.0,
        ssim_weight=0.0,
        perceptual_layers=None
    ):
        """
        Initialize the reconstruction loss.
        
        Args:
            mse_weight: Weight for Mean Squared Error loss
            perceptual_weight: Weight for Perceptual loss (VGG features)
            ssim_weight: Weight for Structural Similarity loss
            perceptual_layers: List of VGG layers to use for perceptual loss
        """
        super(ReconstructionLoss, self).__init__()
        
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        
        # Initialize component losses
        self.mse_loss = nn.MSELoss()
        
        # Initialize perceptual loss if weight > 0
        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss(layers=perceptual_layers)
        else:
            self.perceptual_loss = None
        
        # Initialize SSIM loss if weight > 0
        if ssim_weight > 0:
            self.ssim_loss = SSIMLoss()
        else:
            self.ssim_loss = None
    
    def forward(self, output, target):
        """
        Calculate the combined reconstruction loss.
        
        Args:
            output: Model output tensor (reconstructed image)
            target: Target tensor (original image)
            
        Returns:
            Combined weighted loss
        """
        # Calculate MSE loss
        mse = self.mse_loss(output, target)
        total_loss = self.mse_weight * mse
        
        # Add perceptual loss if enabled
        if self.perceptual_weight > 0 and self.perceptual_loss is not None:
            perceptual = self.perceptual_loss(output, target)
            total_loss += self.perceptual_weight * perceptual
        
        # Add SSIM loss if enabled
        if self.ssim_weight > 0 and self.ssim_loss is not None:
            ssim = self.ssim_loss(output, target)
            total_loss += self.ssim_weight * ssim
        
        return total_loss


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 feature maps.
    """
    def __init__(self, layers=None):
        """
        Initialize the perceptual loss.
        
        Args:
            layers: List of VGG layers to use for feature extraction
        """
        super(PerceptualLoss, self).__init__()
        
        # Use default layers if none provided
        if layers is None:
            layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        
        # Load VGG model without pretrained weights
        vgg = models.vgg16(weights=None)
        
        # Define layer mapping for easier access
        layer_map = {
            'relu1_1': 1, 'relu1_2': 3,
            'relu2_1': 6, 'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15,
            'relu4_1': 18, 'relu4_2': 20, 'relu4_3': 22,
            'relu5_1': 25, 'relu5_2': 27, 'relu5_3': 29
        }
        
        # Create a sequential module with selected layers
        self.features = nn.ModuleDict()
        
        # Create a separate model for each layer
        # This allows us to calculate features at each depth separately
        for layer_name in layers:
            if layer_name in layer_map:
                layer_idx = layer_map[layer_name]
                self.features[layer_name] = nn.Sequential(
                    *list(vgg.features[:layer_idx+1])
                )
        
        # Freeze all VGG parameters
        for param in self.parameters():
            param.requires_grad = False
            
        # Move to evaluation mode
        self.eval()
        
        # Register VGG mean and std for normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _normalize(self, x):
        """Normalize input images for VGG."""
        return (x - self.mean) / self.std
    
    def forward(self, output, target):
        """
        Calculate perceptual loss based on VGG feature maps.
        
        Args:
            output: Model output tensor (reconstructed image)
            target: Target tensor (original image)
            
        Returns:
            Total perceptual loss
        """
        # Normalize inputs for VGG
        output = self._normalize(output)
        target = self._normalize(target)
        
        # Calculate MSE loss between feature maps
        loss = 0.0
        
        for layer_name, layer in self.features.items():
            output_features = layer(output)
            with torch.no_grad():
                target_features = layer(target)
            
            # Calculate MSE loss between feature maps
            layer_loss = F.mse_loss(output_features, target_features)
            loss += layer_loss
        
        # Average over number of layers
        loss /= len(self.features)
        
        return loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss.
    """
    def __init__(self, window_size=11, size_average=True):
        """
        Initialize SSIM loss.
        
        Args:
            window_size: Size of the window for SSIM calculation
            size_average: Whether to average over batch
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3  # RGB images
        self.window = self._create_window(window_size, self.channel)
    
    def _create_window(self, window_size, channel):
        """Create a Gaussian window for SSIM calculation."""
        def _gaussian(window_size, sigma):
            gauss = torch.Tensor([torch.exp(-(x - window_size//2)**2 / float(2*sigma**2)) 
                                 for x in range(window_size)])
            return gauss / gauss.sum()
        
        _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        """Calculate SSIM index between two images."""
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, output, target):
        """
        Calculate SSIM loss.
        
        Args:
            output: Model output tensor (reconstructed image)
            target: Target tensor (original image)
            
        Returns:
            1 - SSIM index (as we need a loss to minimize)
        """
        # Check if window needs to be moved to the same device as input
        if self.window.device != output.device:
            self.window = self.window.to(output.device)
        
        # Calculate SSIM
        ssim_value = self._ssim(
            output, target, 
            self.window, self.window_size, 
            self.channel, self.size_average
        )
        
        # Return loss (1 - SSIM, since we want to maximize SSIM)
        return 1 - ssim_value