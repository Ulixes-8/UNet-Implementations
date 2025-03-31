#!/usr/bin/env python
"""
Simple test script to verify the SimpleLoss function works correctly.
"""

import torch
from models.losses import SimpleLoss, DeepSupervisionWrapper

def test_simple_loss():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create dummy data
    batch_size = 2
    num_classes = 3  # Background, cat, dog
    height, width = 16, 16
    
    # Create random prediction logits
    predictions = torch.randn(batch_size, num_classes, height, width)
    
    # Create target with some border pixels
    target = torch.zeros(batch_size, height, width, dtype=torch.long)
    
    # Add some foreground classes
    target[:, 4:8, 4:8] = 1  # Add class 1 (cat)
    target[:, 10:14, 10:14] = 2  # Add class 2 (dog)
    
    # Add some border pixels (255)
    target[:, 0:2, :] = 255  # Top border
    target[:, -2:, :] = 255  # Bottom border
    target[:, :, 0:2] = 255  # Left border
    target[:, :, -2:] = 255  # Right border
    
    print(f"Target shape: {target.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Number of border pixels: {(target == 255).sum().item()}")
    
    # Create loss function
    print("Creating loss functions...")
    loss_fn = SimpleLoss(weight_dice=1.0, weight_ce=1.0, ignore_index=255)
    deep_loss_fn = DeepSupervisionWrapper(loss_fn)
    
    # Calculate loss
    print("Calculating loss...")
    loss = loss_fn(predictions, target)
    print(f"Loss value: {loss.item()}")
    
    # Test with gradients
    predictions.requires_grad = True
    loss = loss_fn(predictions, target)
    print(f"Loss with grad: {loss.item()}")
    
    # Test backward pass
    print("Testing backward pass...")
    loss.backward()
    print("Backward pass successful!")
    
    # Test with deep supervision
    print("Testing deep supervision...")
    deep_outputs = [predictions, predictions.detach().clone()]
    deep_loss = deep_loss_fn(deep_outputs, target)
    print(f"Deep supervision loss: {deep_loss.item()}")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_simple_loss()