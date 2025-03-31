#!/usr/bin/env python
"""
Test script for UNet model.
This script creates a UNet model with the specified hyperparameters
and runs a forward pass with a test input to ensure it works properly.
"""

import torch
import torch.nn as nn
import numpy as np
from models.unet import UNet
from models.losses import SimpleLoss

def test_unet():
    """Test the UNet model and loss function."""
    print("Testing UNet model...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = UNet(
        in_channels=3,
        num_classes=3,
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
        nonlin_kwargs={"inplace": True}
    )
    
    # Move model to device
    model = model.to(device)
    
    # Print model summary
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Create loss function
    loss_fn = SimpleLoss(weight_dice=1.0, weight_ce=1.0, ignore_index=255)
    
    # Create dummy input
    batch_size = 2
    input_shape = (batch_size, 3, 512, 512)
    input_tensor = torch.randn(input_shape, device=device)
    
    # Create dummy target (with some border pixels)
    target_shape = (batch_size, 512, 512)
    target_tensor = torch.zeros(target_shape, dtype=torch.long, device=device)
    
    # Add some foreground classes
    target_tensor[:, 100:200, 100:200] = 1  # Class 1 (cat)
    target_tensor[:, 300:400, 300:400] = 2  # Class 2 (dog)
    
    # Add border pixels (255)
    target_tensor[:, :10, :] = 255
    target_tensor[:, -10:, :] = 255
    
    # Print shapes
    print(f"Input shape: {input_tensor.shape}")
    print(f"Target shape: {target_tensor.shape}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        try:
            output = model(input_tensor)
            print(f"Output shape: {output.shape}")
            
            # Verify output dimensions
            assert output.shape == (batch_size, 3, 512, 512), "Output shape doesn't match expected shape"
            print("Output shape verification: PASSED")
            
            # Calculate loss
            loss = loss_fn(output, target_tensor)
            print(f"Loss: {loss.item()}")
            
            # Test with gradient
            model.train()
            output = model(input_tensor)
            loss = loss_fn(output, target_tensor)
            loss.backward()
            print("Backward pass: PASSED")
            
            print("All tests PASSED!")
            return True
            
        except Exception as e:
            print(f"Error during testing: {e}")
            return False

if __name__ == "__main__":
    test_unet()