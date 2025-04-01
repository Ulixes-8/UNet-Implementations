import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


class SegmentationMetrics:
    """
    Class for accumulating segmentation metrics across an entire dataset.
    This implementation handles all classes including background and allows 
    ignoring specific pixel values (like border pixels marked as 255).
    """
    
    def __init__(self, num_classes: int, ignore_index: int = 255):
        """
        Initialize the metrics calculator.
        
        Args:
            num_classes: Number of classes in the segmentation task
            ignore_index: Pixel value to ignore in calculations (default: 255)
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all accumulators to zero."""
        # Initialize accumulators for each class
        self.intersections = np.zeros(self.num_classes)
        self.unions = np.zeros(self.num_classes)
        self.true_positives = np.zeros(self.num_classes)
        self.false_positives = np.zeros(self.num_classes)
        self.false_negatives = np.zeros(self.num_classes)
        self.total_pixels = 0
        self.correct_pixels = 0
    
    def update(self, pred: Union[np.ndarray, torch.Tensor], 
               target: Union[np.ndarray, torch.Tensor]):
        """
        Update accumulators with a new batch of predictions and targets.
        
        Args:
            pred: Prediction array/tensor with shape (H, W) or (B, H, W)
            target: Ground truth array/tensor with shape (H, W) or (B, H, W)
        """
        # Convert tensors to NumPy if needed
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        # Handle batch input
        if pred.ndim == 3:
            for i in range(pred.shape[0]):
                self._update_single(pred[i], target[i])
        else:
            self._update_single(pred, target)
    
    def _update_single(self, pred: np.ndarray, target: np.ndarray):
        """
        Update metrics for a single prediction-target pair.
        
        Args:
            pred: Single prediction array with shape (H, W)
            target: Single ground truth array with shape (H, W)
        """
        # Create mask for valid pixels (not ignore_index)
        mask = (target != self.ignore_index)
        
        # Update pixel accuracy metrics
        self.total_pixels += mask.sum()
        self.correct_pixels += ((pred == target) & mask).sum()
        
        # Update class-wise metrics
        for cls in range(self.num_classes):
            # Create binary masks for current class
            pred_cls = (pred == cls) & mask
            target_cls = (target == cls) & mask
            
            # Calculate intersection and union
            intersection = (pred_cls & target_cls).sum()
            union = pred_cls.sum() + target_cls.sum() - intersection
            
            # Update accumulators
            self.intersections[cls] += intersection
            self.unions[cls] += union
            
            # For precision/recall/dice calculation
            self.true_positives[cls] += intersection
            self.false_positives[cls] += pred_cls.sum() - intersection
            self.false_negatives[cls] += target_cls.sum() - intersection
    
    def compute_pixel_accuracy(self) -> float:
        """
        Compute overall pixel accuracy using accumulated statistics.
        
        Returns:
            Pixel accuracy across the entire dataset
        """
        if self.total_pixels > 0:
            return float(self.correct_pixels / self.total_pixels)
        return float('nan')
    
    def compute_iou(self, cls: int) -> float:
        """
        Compute IoU (Intersection over Union) for a specific class.
        
        Args:
            cls: Class index
            
        Returns:
            IoU value for the specified class
        """
        if self.unions[cls] > 0:
            return float(self.intersections[cls] / self.unions[cls])
        return float('nan')
    
    def compute_mean_iou(self) -> float:
        """
        Compute mean IoU across all classes.
        
        Returns:
            Mean IoU value
        """
        valid_ious = []
        for cls in range(self.num_classes):
            iou = self.compute_iou(cls)
            if not np.isnan(iou):
                valid_ious.append(iou)
        
        if valid_ious:
            return float(sum(valid_ious) / len(valid_ious))
        return float('nan')
    
    def compute_dice(self, cls: int) -> float:
        """
        Compute Dice coefficient for a specific class.
        
        Args:
            cls: Class index
            
        Returns:
            Dice coefficient for the specified class
        """
        numerator = 2 * self.true_positives[cls]
        denominator = 2 * self.true_positives[cls] + self.false_positives[cls] + self.false_negatives[cls]
        
        if denominator > 0:
            return float(numerator / denominator)
        return float('nan')
    
    def compute_mean_dice(self) -> float:
        """
        Compute mean Dice coefficient across all classes.
        
        Returns:
            Mean Dice coefficient
        """
        valid_dices = []
        for cls in range(self.num_classes):
            dice = self.compute_dice(cls)
            if not np.isnan(dice):
                valid_dices.append(dice)
        
        if valid_dices:
            return float(sum(valid_dices) / len(valid_dices))
        return float('nan')
    
    def compute_precision(self, cls: int) -> float:
        """
        Compute precision for a specific class.
        
        Args:
            cls: Class index
            
        Returns:
            Precision value for the specified class
        """
        tp = self.true_positives[cls]
        fp = self.false_positives[cls]
        
        if (tp + fp) > 0:
            return float(tp / (tp + fp))
        return float('nan')
    
    def compute_recall(self, cls: int) -> float:
        """
        Compute recall for a specific class.
        
        Args:
            cls: Class index
            
        Returns:
            Recall value for the specified class
        """
        tp = self.true_positives[cls]
        fn = self.false_negatives[cls]
        
        if (tp + fn) > 0:
            return float(tp / (tp + fn))
        return float('nan')
    
    def compute_f1_score(self, cls: int) -> float:
        """
        Compute F1 score for a specific class.
        Note: F1 score is equivalent to Dice coefficient.
        
        Args:
            cls: Class index
            
        Returns:
            F1 score for the specified class
        """
        return self.compute_dice(cls)
    
    def get_all_metrics(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Get all metrics in a dictionary format.
        
        Returns:
            Dictionary containing all metrics
        """
        results = {
            "pixel_accuracy": self.compute_pixel_accuracy(),
            "mean_iou": self.compute_mean_iou(),
            "mean_dice": self.compute_mean_dice(),
            "class_metrics": {}
        }
        
        # Add class-specific metrics
        for cls in range(self.num_classes):
            results["class_metrics"][f"class_{cls}"] = {
                "iou": self.compute_iou(cls),
                "dice": self.compute_dice(cls),
                "precision": self.compute_precision(cls),
                "recall": self.compute_recall(cls),
                "f1_score": self.compute_f1_score(cls)
            }
        
        return results


# Standalone functions for single prediction-target evaluation
def compute_dice(pred: Union[np.ndarray, torch.Tensor], 
                 target: Union[np.ndarray, torch.Tensor], 
                 cls: int, 
                 ignore_index: int = 255) -> float:
    """
    Calculate Dice coefficient for a specific class in a single prediction-target pair.
    
    Args:
        pred: Prediction array/tensor
        target: Ground truth array/tensor
        cls: Class index to compute Dice for
        ignore_index: Pixel value to ignore in calculations
        
    Returns:
        Dice coefficient value
    """
    metrics = SegmentationMetrics(num_classes=max(cls+1, 3), ignore_index=ignore_index)
    metrics.update(pred, target)
    return metrics.compute_dice(cls)


def compute_iou(pred: Union[np.ndarray, torch.Tensor], 
                target: Union[np.ndarray, torch.Tensor], 
                cls: int, 
                ignore_index: int = 255) -> float:
    """
    Calculate IoU for a specific class in a single prediction-target pair.
    
    Args:
        pred: Prediction array/tensor
        target: Ground truth array/tensor
        cls: Class index to compute IoU for
        ignore_index: Pixel value to ignore in calculations
        
    Returns:
        IoU value
    """
    metrics = SegmentationMetrics(num_classes=max(cls+1, 3), ignore_index=ignore_index)
    metrics.update(pred, target)
    return metrics.compute_iou(cls)


def compute_pixel_accuracy(pred: Union[np.ndarray, torch.Tensor], 
                           target: Union[np.ndarray, torch.Tensor], 
                           ignore_index: int = 255) -> float:
    """
    Calculate pixel accuracy for a single prediction-target pair.
    
    Args:
        pred: Prediction array/tensor
        target: Ground truth array/tensor
        ignore_index: Pixel value to ignore in calculations
        
    Returns:
        Pixel accuracy value
    """
    metrics = SegmentationMetrics(num_classes=3, ignore_index=ignore_index)
    metrics.update(pred, target)
    return metrics.compute_pixel_accuracy()


def evaluate_model_metrics(model: torch.nn.Module, 
                          data_loader: torch.utils.data.DataLoader,
                          device: torch.device,
                          num_classes: int = 3,
                          ignore_index: int = 255) -> Dict:
    """
    Evaluate a segmentation model on a dataset and return comprehensive metrics.
    
    Args:
        model: The segmentation model to evaluate
        data_loader: DataLoader for the evaluation dataset
        device: Device to run the model on
        num_classes: Number of classes in the segmentation task
        ignore_index: Pixel value to ignore in calculations
        
    Returns:
        Dictionary containing all computed metrics
    """
    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            original_dims = batch["original_dims"]
            
            # Forward pass
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Process each prediction
            for i in range(preds.size(0)):
                orig_h, orig_w = original_dims[i]
                
                # Resize prediction to original dimensions
                pred_resized = F.interpolate(
                    preds[i][None, None].float(),
                    size=(orig_h, orig_w),
                    mode="nearest"
                ).squeeze().cpu().numpy().astype(np.uint8)
                
                # Resize mask to original dimensions
                mask_resized = F.interpolate(
                    masks[i][None, None].float(),
                    size=(orig_h, orig_w),
                    mode="nearest"
                ).squeeze().cpu().numpy().astype(np.uint8)
                
                # Update metrics
                metrics.update(pred_resized, mask_resized)
    
    # Gather all metrics
    return metrics.get_all_metrics()