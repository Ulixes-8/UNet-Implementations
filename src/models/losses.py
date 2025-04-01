import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLoss(nn.Module):
    """
    A combined Dice and weighted Cross Entropy loss with proper handling of border class (255).
    Implements class weighting for cross-entropy based on inverse frequency.
    """
    def __init__(self, weight_dice=1.0, weight_ce=1.0, ignore_index=255, smooth=1e-5, 
                 class_weights=None, dynamic_weights=True):
        super(SimpleLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.class_weights = class_weights
        self.dynamic_weights = dynamic_weights
        
        # Create Cross Entropy Loss with ignore_index
        # Note: class_weights will be calculated dynamically if None
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    
    def _compute_class_weights(self, target):
        """
        Compute class weights based on inverse pixel frequency in the current batch.
        Formula: w_n = total number of pixels / number of pixels in class
        
        Args:
            target: Target segmentation mask of shape (B, H, W)
            
        Returns:
            torch.Tensor: Class weights tensor
        """
        # Create a mask for valid pixels (not ignore_index/border class)
        mask = (target != self.ignore_index)
        valid_target = target * mask.long()
        
        # Get the number of classes
        num_classes = 3  # background, cat, dog
        
        # Count pixels per class
        batch_size = target.size(0)
        total_pixels = torch.sum(mask).float()
        
        # Initialize class pixel counts
        class_pixels = torch.zeros(num_classes, device=target.device)
        
        # Count pixels for each class
        for c in range(num_classes):
            class_pixels[c] = torch.sum((valid_target == c) & mask).float()
            # Avoid division by zero
            if class_pixels[c] == 0:
                class_pixels[c] = 1.0
        
        # Compute weights: total / class_count
        weights = total_pixels / class_pixels
        
        # Normalize weights to sum to num_classes
        weights = weights * (num_classes / weights.sum())
        
        return weights
    
    def forward(self, input, target):
        # Ensure input and target have the same spatial dimensions
        if input.shape[-2:] != target.shape[-2:]:
            input = F.interpolate(input, size=target.shape[-2:], 
                                mode='bilinear', align_corners=False)
        
        # Update CrossEntropyLoss with dynamic class weights if enabled
        if self.dynamic_weights and target.size(0) > 0:
            weights = self._compute_class_weights(target)
            self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=self.ignore_index)
        
        # Cross entropy loss
        ce_loss = self.ce(input, target)
        
        # Dice loss with ignore index handling
        dice_loss = self._dice_loss(input, target)
        
        # Combine losses
        return self.weight_ce * ce_loss + self.weight_dice * dice_loss
    
    def _dice_loss(self, input, target):
        # Create a mask for valid pixels (not ignore_index/border class)
        mask = (target != self.ignore_index).float()
        
        # Get the number of classes
        num_classes = input.shape[1]
        
        # Apply softmax to get class probabilities
        input_soft = F.softmax(input, dim=1)
        
        # Initialize dice loss
        dice_loss = 0
        
        # Calculate dice loss for each class (including background class 0)
        for c in range(num_classes):  # Include background class
            # Create binary target for current class
            target_c = (target == c).float()
            
            # Apply mask to both target and input
            target_c = target_c * mask
            input_c = input_soft[:, c, :, :] * mask
            
            # Flatten for easier calculation
            target_flat = target_c.reshape(target_c.size(0), -1)
            input_flat = input_c.reshape(input_c.size(0), -1)
            
            # Calculate intersection and union
            intersection = (input_flat * target_flat).sum(dim=1)
            union = input_flat.sum(dim=1) + target_flat.sum(dim=1)
            
            # Calculate dice
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            
            # Add to total dice loss (1 - dice for minimization)
            dice_loss += (1.0 - dice.mean())
        
        # Average over number of classes (including background)
        return dice_loss / num_classes