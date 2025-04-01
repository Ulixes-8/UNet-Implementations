# Define SegmentationMetrics class based on the functions in metrics.py
import numpy as np

class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
        
    def reset(self):
        # Initialize accumulators for each class
        self.intersections = np.zeros(self.num_classes)
        self.unions = np.zeros(self.num_classes)
        self.true_positives = np.zeros(self.num_classes)
        self.false_positives = np.zeros(self.num_classes)
        self.false_negatives = np.zeros(self.num_classes)
        self.total_pixels = 0
        self.correct_pixels = 0
        
    def update(self, pred, target):
        """
        Update accumulators with a new batch of predictions and targets
        
        Args:
            pred: prediction tensor/array
            target: ground truth tensor/array
        """
        mask = (target != self.ignore_index)
        self.total_pixels += mask.sum()
        self.correct_pixels += ((pred == target) & mask).sum()
        
        # Update class-wise metrics
        for cls in range(self.num_classes):
            pred_cls = (pred == cls) & mask
            target_cls = (target == cls) & mask
            
            intersection = (pred_cls & target_cls).sum()
            union = (pred_cls | target_cls).sum()
            
            # Update accumulators
            self.intersections[cls] += intersection
            self.unions[cls] += union
            
            # For dice coefficient
            self.true_positives[cls] += intersection
            self.false_positives[cls] += pred_cls.sum() - intersection
            self.false_negatives[cls] += target_cls.sum() - intersection
            
    def compute_dice(self, cls):
        """
        Compute Dice coefficient for a specific class using accumulated statistics
        
        Args:
            cls: class index
            
        Returns:
            dice: Dice coefficient for the specified class
        """
        numerator = 2 * self.true_positives[cls]
        denominator = 2 * self.true_positives[cls] + self.false_positives[cls] + self.false_negatives[cls]
        
        if denominator > 0:
            return (numerator / denominator).item()
        return float('nan')
    
    def compute_pixel_accuracy(self):
        """
        Compute overall pixel accuracy using accumulated statistics
        
        Returns:
            accuracy: Pixel accuracy across the entire dataset
        """
        if self.total_pixels > 0:
            return (self.correct_pixels / self.total_pixels).item()
        return float('nan')
    
    def compute_iou(self, cls):
        """
        Compute IoU for a specific class using accumulated statistics
        
        Args:
            cls: class index
            
        Returns:
            iou: IoU for the specified class
        """
        if self.unions[cls] > 0:
            return (self.intersections[cls] / self.unions[cls]).item()
        return float('nan')
    
    def compute_mean_iou(self):
        """
        Compute mean IoU across all classes
        
        Returns:
            mean_iou: Mean IoU value
        """
        valid_ious = []
        for cls in range(self.num_classes):
            iou = self.compute_iou(cls)
            if not np.isnan(iou):
                valid_ious.append(iou)
        
        if valid_ious:
            return sum(valid_ious) / len(valid_ious)
        return float('nan')
    
    def compute_precision(self, cls):
        tp = self.true_positives[cls]
        fp = self.false_positives[cls]
        if (tp + fp) > 0:
            return (tp / (tp + fp)).item()
        return float('nan')

    def compute_recall(self, cls):
        tp = self.true_positives[cls]
        fn = self.false_negatives[cls]
        if (tp + fn) > 0:
            return (tp / (tp + fn)).item()
        return float('nan')
