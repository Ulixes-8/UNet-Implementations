#!/usr/bin/env python
"""
Module: losses.py

This module implements loss functions for semantic segmentation, specifically designed
for the Oxford-IIIT Pet Dataset segmentation task. It includes:

1. SoftDiceLoss - A differentiable version of the Dice coefficient
2. CrossEntropyND - N-dimensional cross-entropy loss
3. DC_and_CE_loss - Combined Dice and Cross-Entropy loss
4. DeepSupervisionWrapper - Wrapper for handling deep supervision in UNet architectures

These loss functions are based on the nnUNet implementation but have been adapted and
reimplemented to be self-contained.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_helper(x: torch.Tensor) -> torch.Tensor:
    """
    Compute softmax along the channel dimension (dim=1).
    This implementation avoids numerical instability by subtracting the maximum value.
    
    Args:
        x: Input tensor of shape (B, C, ...)
        
    Returns:
        Softmax output tensor of same shape
    """
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


def sum_tensor(inp: torch.Tensor, axes: Union[List[int], Tuple[int, ...], np.ndarray], 
               keepdim: bool = False) -> torch.Tensor:
    """
    Sum a tensor along multiple axes.
    
    Args:
        inp: Input tensor
        axes: List/tuple of axes to sum over
        keepdim: Whether to keep the summed dimensions
        
    Returns:
        Summed tensor
    """
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn(net_output: torch.Tensor, 
                 gt: torch.Tensor, 
                 axes: Optional[Union[List[int], Tuple[int, ...], np.ndarray]] = None,
                 mask: Optional[torch.Tensor] = None, 
                 square: bool = False,
                 ignore_label: int = 255) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate true positives (TP), false positives (FP), and false negatives (FN).
    
    Args:
        net_output: Model output of shape (B, C, ...)
        gt: Ground truth of shape (B, 1, ...) or (B, C, ...)
        axes: Spatial dimensions to sum over. If None, sums over all spatial dimensions.
        mask: Optional binary mask for region of interest
        square: Whether to square TP, FP, FN values before summation
        ignore_label: Label to ignore in evaluation (e.g., border regions)
        
    Returns:
        Tuple of (TP, FP, FN) tensors
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    # Create an ignore mask for pixels with ignore_label
    ignore_mask = None
    if ignore_label is not None:
        ignore_mask = (gt != ignore_label)
        if len(ignore_mask.shape) > len(shp_x) - 1:
            ignore_mask = ignore_mask.squeeze(1)

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # If shapes match, ground truth is probably already one-hot encoded
            y_onehot = gt
        else:
            # Create a copy of gt that replaces ignore_label with 0 (to avoid index errors)
            gt_temp = gt.clone()
            if ignore_label is not None:
                gt_temp[gt == ignore_label] = 0
            
            gt_temp = gt_temp.long()
            y_onehot = torch.zeros(shp_x, device=net_output.device)
            y_onehot.scatter_(1, gt_temp, 1)
            
            # Mask out the ignore regions in the one-hot encoded gt
            if ignore_mask is not None:
                y_onehot = y_onehot * ignore_mask.unsqueeze(1)

    # Apply ignore mask to net_output if needed
    net_output_masked = net_output
    if ignore_mask is not None:
        net_output_masked = net_output * ignore_mask.unsqueeze(1)

    # Calculate TP, FP, FN using masked versions
    tp = net_output_masked * y_onehot
    fp = net_output_masked * (1 - y_onehot)
    fn = (1 - net_output_masked) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss for semantic segmentation.
    
    This is a differentiable version of the Dice coefficient, suitable for training.
    """

    def __init__(self, 
                 apply_nonlin: Optional[Callable] = None, 
                 batch_dice: bool = False, 
                 do_bg: bool = True, 
                 smooth: float = 1.0,
                 square: bool = False,
                 ignore_label: int = 255):
        """
        Initialize SoftDiceLoss.
        
        Args:
            apply_nonlin: Function to apply to network outputs (e.g., softmax)
            batch_dice: Whether to compute dice over the batch dimension
            do_bg: Whether to include background class in loss
            smooth: Smoothing factor to avoid division by zero
            square: Whether to square TP, FP, FN values before summation
            ignore_label: Label to ignore in evaluation (e.g., border regions)
        """
        super(SoftDiceLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ignore_label = ignore_label

    def forward(self, 
                x: torch.Tensor, 
                y: torch.Tensor, 
                loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of Dice loss.
        
        Args:
            x: Network output (B, C, ...)
            y: Ground truth labels
            loss_mask: Optional mask for regions to include in loss
            
        Returns:
            Dice loss value (negative to minimize)
        """
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(
            x, y, axes, loss_mask, self.square, ignore_label=self.ignore_label
        )

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class CrossEntropyND(nn.CrossEntropyLoss):
    """
    N-dimensional Cross Entropy Loss.
    
    This is an extension of nn.CrossEntropyLoss that handles N-dimensional inputs.
    Note: The network should have NO SOFTMAX/SIGMOID at the output.
    """

    def forward(self, 
                inp: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of N-dimensional Cross Entropy Loss.
        
        Args:
            inp: Network output (B, C, ...)
            target: Ground truth labels
            
        Returns:
            Cross entropy loss value
        """
        # Create a copy of target to avoid modifying the original
        target_temp = target.clone()
        
        # Replace ignore label (255) with 0 to avoid indexing errors
        # The ignore label will be handled by the loss function's ignore_index parameter
        target_temp[target == 255] = 0
        
        target_temp = target_temp.long()
        num_classes = inp.size()[1]
        
        # Reshape input for cross entropy: (B, C, D1, D2, ...) -> (B*D1*D2*..., C)
        i0 = 1
        i1 = 2
        
        while i1 < len(inp.shape):  # Transpose only works for two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1
            
        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)
        
        # Reshape target to match: (B, 1, D1, D2, ...) -> (B*D1*D2*...)
        target_temp = target_temp.view(-1)
        
        # Create a mask for non-ignored pixels
        mask = (target != 255).view(-1)
        
        # Apply the mask to both input and target
        masked_inp = inp[mask]
        masked_target = target_temp[mask]
        
        if masked_inp.shape[0] == 0:
            return torch.tensor(0.0, device=inp.device, requires_grad=True)
        
        # Compute CE loss only on non-ignored pixels
        return nn.functional.cross_entropy(masked_inp, masked_target)


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    Robust version of Cross Entropy Loss.
    
    This handles edge cases and potential numerical instabilities better
    than the standard CrossEntropyLoss.
    """

    def forward(self, 
                inp: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Robust Cross Entropy Loss.
        
        Args:
            inp: Network output (B, C, ...)
            target: Ground truth labels
            
        Returns:
            Cross entropy loss value
        """
        # Create a copy of target to avoid modifying the original
        target_temp = target.clone()
        
        # Replace ignore label (255) with 0 to avoid indexing errors
        # The ignore label will be handled separately with a mask
        target_temp[target == 255] = 0
        
        # Check if target is one-hot encoded
        if len(target.shape) > 1:
            # Convert one-hot encoding to class indices
            if target.shape[1] > 1:
                target_temp = target_temp.argmax(1)
            else:
                target_temp = target_temp[:, 0]
                
        # Create cross entropy input of shape (N, C)
        if len(inp.shape) > 2:
            # Convert 5D/4D/3D input to 2D
            inp = inp.transpose(1, -1)  # Convert to (B, ..., C)
            inp = inp.reshape(-1, inp.size(-1))  # Reshape to (B*..., C)
            
        target_temp = target_temp.reshape(-1)
        
        # Create a mask for non-ignored pixels
        mask = (target.reshape(-1) != 255)
        
        # Apply the mask to both input and target
        masked_inp = inp[mask]
        masked_target = target_temp[mask]
        
        # Handle empty mask case
        if masked_inp.shape[0] == 0:
            return torch.tensor(0.0, device=inp.device, requires_grad=True)
        
        # Compute CE loss only on non-ignored pixels
        return nn.functional.cross_entropy(masked_inp, masked_target)


class DC_and_CE_loss(nn.Module):
    """
    Combined loss of Dice coefficient and Cross Entropy.
    
    This combination provides both good optimization properties and
    meaningful gradient signals for imbalanced segmentation tasks.
    """

    def __init__(self, 
                 soft_dice_kwargs: Optional[Dict] = None, 
                 ce_kwargs: Optional[Dict] = None, 
                 aggregate: str = "sum",
                 ignore_label: int = 255):
        """
        Initialize combined Dice and Cross Entropy loss.
        
        Args:
            soft_dice_kwargs: Arguments for SoftDiceLoss
            ce_kwargs: Arguments for CrossEntropyND
            aggregate: How to aggregate the losses ('sum' for now)
            ignore_label: Label to ignore in evaluation (e.g., border regions)
        """
        super(DC_and_CE_loss, self).__init__()
        
        self.aggregate = aggregate
        self.ignore_label = ignore_label
        
        if soft_dice_kwargs is None:
            soft_dice_kwargs = {'batch_dice': True, 'do_bg': False, 'smooth': 1e-5, 'ignore_label': ignore_label}
        else:
            soft_dice_kwargs['ignore_label'] = ignore_label
            
        if ce_kwargs is None:
            ce_kwargs = {}
            
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, 
                net_output: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of combined Dice and Cross Entropy loss.
        
        Args:
            net_output: Network output (B, C, ...)
            target: Ground truth labels
            
        Returns:
            Combined loss value
        """
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            # Reserved for future implementations
            raise NotImplementedError("Only 'sum' aggregation is implemented")
            
        return result


class DeepSupervisionWrapper(nn.Module):
    """
    Wrapper module for applying a loss function to deep supervision outputs.
    
    This handles varying numbers of outputs from a deeply supervised network,
    applying appropriate weights to each level's loss.
    """

    def __init__(self, loss: nn.Module):
        """
        Initialize deep supervision wrapper.
        
        Args:
            loss: Base loss function to apply at each supervision level
        """
        super(DeepSupervisionWrapper, self).__init__()
        self.loss = loss

    def forward(self, 
                net_output: Union[torch.Tensor, List[torch.Tensor]], 
                target: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass of deep supervision wrapper.
        
        Args:
            net_output: Network output or list of outputs from different levels
            target: Ground truth or list of ground truths for different levels
            
        Returns:
            Weighted sum of losses from different levels
        """
        # If a single output is provided, just apply the loss directly
        if not isinstance(net_output, (list, tuple)):
            return self.loss(net_output, target)
            
        # Handle deep supervision outputs
        if isinstance(target, (list, tuple)):
            # If target is also a list, we assume it's already properly formatted for each level
            assert len(net_output) == len(target)
            losses = [self.loss(net_output[i], target[i]) for i in range(len(target))]
        else:
            # If target is a single tensor, use it for all outputs
            losses = [self.loss(net_output[i], target) for i in range(len(net_output))]
            
        # Weight the losses - give more weight to the final output
        # We use a simple linear weighting scheme
        weights = np.linspace(0.5, 1.0, len(losses))
        weights = weights / weights.sum()
        
        # Apply weights and sum
        weighted_losses = [weights[i] * losses[i] for i in range(len(losses))]
        
        return sum(weighted_losses)