import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert a segmentation mask with values {0,1,2} into an RGB color image.
    - 0 → black (0, 0, 0) for background
    - 1 → red (255, 0, 0) for cats
    - 2 → green (0, 255, 0) for dogs
    
    Args:
        mask: Segmentation mask with class indices
    
    Returns:
        RGB color image
    """
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb_mask[mask == 1] = [255, 0, 0]  # Red for class 1 (cat)
    rgb_mask[mask == 2] = [0, 255, 0]  # Green for class 2 (dog)
    return rgb_mask


def visualize_prediction_batch(model: torch.nn.Module, batch: Dict, device: torch.device) -> None:
    """
    Visualize a batch of predictions from the model.
    
    Args:
        model: The trained UNet model
        batch: A batch of data from the dataloader
        device: Device to run the model on
    """
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    filenames = batch.get("filename", [f"Sample_{i}" for i in range(images.size(0))])
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
    
    # Create a figure for the batch
    n_samples = images.size(0)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
    
    # Handle case where only one sample is present
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Denormalization parameters
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Process each sample in the batch
    for i in range(n_samples):
        # Get original image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # Get ground truth and prediction masks
        gt_mask = masks[i].cpu().numpy()
        pred_mask = preds[i].cpu().numpy()
        
        # Colorize masks
        gt_colored = colorize_mask(gt_mask)
        pred_colored = colorize_mask(pred_mask)
        
        # Create a mask for pixels to ignore (value = 255)
        ignore_mask = (gt_mask == 255)
        
        # Set ignored pixels to black in ground truth visualization
        if np.any(ignore_mask):
            gt_colored[ignore_mask] = [0, 0, 0]
        
        # Plot image and masks
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image: {filenames[i]}")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(gt_colored)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(pred_colored)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    plt.show()


def visualize_confidence_maps_batch(model: torch.nn.Module, batch: Dict, device: torch.device, 
                                    class_names: List[str] = None) -> None:
    """
    Visualize the softmax probability maps (confidence heatmaps) for each image in the batch.
    Overlays each class's probability heatmap on top of the (denormalized) original image,
    with a color bar to show the [0..1] range of model confidence.
    
    Args:
        model: The trained UNet model in eval mode
        batch: A dictionary from the PetSegmentationDataset/DataLoader
        device: The device (CPU or CUDA) on which the model is running
        class_names: Optional list of class names for labeling subplots
    """
    if class_names is None:
        class_names = ["Background", "Cat", "Dog"]
    
    # Extract images and filenames
    images = batch["image"].to(device)  # shape: [B, 3, H, W]
    filenames = batch.get("filename", [f"Sample_{i}" for i in range(images.size(0))])
    
    # Forward pass to get softmax probabilities
    with torch.no_grad():
        logits = model(images)  # [B, num_classes, H, W]
        prob_maps = F.softmax(logits, dim=1)  # [B, num_classes, H, W], values in [0..1]
    
    # Convert images to NumPy for plotting
    images_np = images.cpu().numpy()  # shape: [B, 3, H, W]
    num_classes = prob_maps.size(1)
    
    # Iterate over each image in the batch
    for i in range(images_np.shape[0]):
        fig, axes = plt.subplots(1, 1 + num_classes, figsize=((1 + num_classes) * 5, 5))
        
        # If only one class or image, ensure 'axes' is iterable
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        
        # Show Original Image (Denormalized)
        original_img = images_np[i].transpose(1, 2, 0)  # shape: [H, W, 3]
        
        # Denormalize if you used ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original_img = (original_img * std + mean)
        original_img = np.clip(original_img, 0, 1)
        
        axes[0].imshow(original_img)
        axes[0].set_title(f"Original\n{filenames[i]}")
        axes[0].axis("off")
        
        # Show Confidence Heatmaps for Each Class
        alpha = 0.5  # how much to blend heatmap with original image
        for c in range(num_classes):
            ax_map = axes[c + 1]
            
            # Probability map for this class
            prob_map_c = prob_maps[i, c].cpu().numpy()  # shape: [H, W]
            
            # First, plot the original image (so it sits "behind")
            ax_map.imshow(original_img)
            
            # Then overlay the probability heatmap in "jet" colors
            heatmap_img = ax_map.imshow(
                prob_map_c,
                cmap='jet',
                alpha=alpha,  # partially transparent so we can see the original
                vmin=0,
                vmax=1
            )
            
            class_label = class_names[c] if c < len(class_names) else f"Class {c}"
            ax_map.set_title(f"Confidence: {class_label}")
            ax_map.axis("off")
            
            # Add a color bar to show the 0..1 scale for the probability
            cbar = plt.colorbar(heatmap_img, ax=ax_map, fraction=0.046, pad=0.04)
            cbar.set_label("Probability")
        
        plt.tight_layout()
        plt.show()


def create_error_visualization(pred_mask: np.ndarray, gt_mask: np.ndarray, 
                              original_img: np.ndarray) -> np.ndarray:
    """
    Create an error visualization showing correct and incorrect predictions.
    
    Args:
        pred_mask: Predicted segmentation mask
        gt_mask: Ground truth segmentation mask
        original_img: Original image (denormalized)
        
    Returns:
        Blended error visualization
    """
    # Create error map with same dimensions as original image
    if len(original_img.shape) == 3:
        h, w, c = original_img.shape
        error_map = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        h, w = original_img.shape
        error_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Convert original_img to uint8 if needed
    if original_img.dtype != np.uint8 and np.max(original_img) <= 1.0:
        original_img_uint8 = (original_img * 255).astype(np.uint8)
    else:
        original_img_uint8 = original_img.astype(np.uint8)
    
    # Modify masks to treat 255 as background (0)
    pred_mask_processed = pred_mask.copy()
    gt_mask_processed = gt_mask.copy()
    pred_mask_processed[pred_mask_processed == 255] = 0
    gt_mask_processed[gt_mask_processed == 255] = 0
    
    # Compute error categories
    # True Positive: Pixels correctly predicted for non-background classes
    true_positive = (pred_mask_processed > 0) & (gt_mask_processed > 0) & (pred_mask_processed == gt_mask_processed)
    
    # False Positive: Pixels predicted as an object where no object exists
    false_positive = (pred_mask_processed > 0) & (gt_mask_processed == 0)
    
    # False Negative: Pixels that are objects in ground truth but not predicted
    false_negative = (pred_mask_processed == 0) & (gt_mask_processed > 0)
    
    # Wrong Class: Pixels where an object is predicted, but with the wrong class
    wrong_class = (pred_mask_processed > 0) & (gt_mask_processed > 0) & (pred_mask_processed != gt_mask_processed)
    
    # Color code error categories
    error_map[true_positive] = [0, 255, 0]    # Green for true positives
    error_map[false_positive] = [255, 0, 0]   # Red for false positives
    error_map[false_negative] = [0, 0, 255]   # Blue for false negatives
    error_map[wrong_class] = [255, 255, 0]    # Yellow for wrong class predictions
    
    # Blend error map with original image
    alpha = 0.5
    if len(original_img_uint8.shape) == 2:
        # Convert grayscale to RGB
        original_img_uint8 = np.stack([original_img_uint8] * 3, axis=2)
    
    blended = (original_img_uint8 * (1 - alpha) + error_map * alpha).astype(np.uint8)
    
    return blended


def visualize_error_analysis_batch(model: torch.nn.Module, batch: Dict, device: torch.device) -> None:
    """
    Visualize error analysis for a batch of images.
    
    Args:
        model: The trained UNet model
        batch: A batch of data from the dataloader
        device: Device to run the model on
    """
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    filenames = batch.get("filename", [f"Sample_{i}" for i in range(images.size(0))])
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
    
    # Create a figure for the batch
    n_samples = images.size(0)
    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 5 * n_samples))
    
    # Handle case where only one sample is present
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Denormalization parameters
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Process each sample in the batch
    for i in range(n_samples):
        # Get original image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * std + mean
        img = np.clip(img, 0, 1)
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Get ground truth and prediction masks
        gt_mask = masks[i].cpu().numpy()
        pred_mask = preds[i].cpu().numpy()
        
        # Colorize masks
        gt_colored = colorize_mask(gt_mask)
        pred_colored = colorize_mask(pred_mask)
        
        # Create error visualization
        error_viz = create_error_visualization(pred_mask, gt_mask, img_uint8)
        
        # Plot images
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image: {filenames[i]}")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(gt_colored)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(pred_colored)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")
        
        axes[i, 3].imshow(error_viz)
        axes[i, 3].set_title("Error Analysis")
        axes[i, 3].axis("off")
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color='g', label='Correct (TP)'),
        plt.Rectangle((0, 0), 1, 1, color='r', label='False Positive'),
        plt.Rectangle((0, 0), 1, 1, color='b', label='False Negative'),
        plt.Rectangle((0, 0), 1, 1, color='y', label='Wrong Class')
    ]
    
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 0),
        ncol=4,
        fontsize='large'
    )
    
    plt.tight_layout(rect=(0, 0.05, 1, 1))  # Adjust layout to make room for legend
    plt.show()


def plot_class_distributions(masks: List[np.ndarray], class_names: List[str] = None) -> None:
    """
    Plot the distribution of classes in a set of masks.
    
    Args:
        masks: List of segmentation masks
        class_names: Optional list of class names
    """
    if class_names is None:
        class_names = ["Background", "Cat", "Dog"]
    
    # Count pixels for each class
    class_counts = {}
    total_pixels = 0
    
    for mask in masks:
        # Exclude ignored pixels (255)
        valid_mask = (mask != 255)
        for cls_idx, cls_name in enumerate(class_names):
            if cls_name not in class_counts:
                class_counts[cls_name] = 0
            # Count pixels for this class
            cls_count = np.sum((mask == cls_idx) & valid_mask)
            class_counts[cls_name] += cls_count
            total_pixels += cls_count
    
    # Convert counts to percentages
    class_percentages = {cls_name: count / total_pixels * 100 
                        for cls_name, count in class_counts.items()}
    
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(class_percentages.keys(), class_percentages.values())
    plt.ylabel("Percentage of Pixels (%)")
    plt.title("Distribution of Classes")
    
    # Add percentage labels on the bars
    for i, (cls_name, percentage) in enumerate(class_percentages.items()):
        plt.text(i, percentage + 1, f"{percentage:.1f}%", 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def generate_gradcam_heatmap(model: torch.nn.Module, input_tensor: torch.Tensor, 
                             target_class: int, target_layer: torch.nn.Module,
                             device: torch.device) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for a given input image and target class.
    
    Args:
        model: Segmentation model
        input_tensor: Input image tensor (1, C, H, W)
        target_class: Class index for which to compute Grad-CAM
        target_layer: Model layer to target (e.g., a convolutional layer)
        device: Device to run on
    
    Returns:
        Grad-CAM heatmap
    """
    model.eval()
    
    # Hook for saving feature maps
    feature_maps = {}
    def forward_hook(module, input, output):
        feature_maps['value'] = output.detach()

    # Hook for saving gradients
    gradients = {}
    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    # Register hooks
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    input_tensor = input_tensor.to(device)
    output = model(input_tensor)
    
    # Target logits for the specified class
    target_score = output[0, target_class, :, :].mean()
    
    # Backward pass to get gradients
    model.zero_grad()
    target_score.backward(retain_graph=True)
    
    # Get feature maps and gradients
    grads_val = gradients['value']
    fmap = feature_maps['value']
    
    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()
    
    # Global average pooling of gradients
    weights = torch.mean(grads_val, dim=(2, 3), keepdim=True)
    
    # Weighted combination of feature maps
    cam = torch.sum(weights * fmap, dim=1)
    cam = F.relu(cam)
    
    # Normalize heatmap
    cam = cam - cam.min()
    if cam.max() != 0:
        cam = cam / cam.max()
    
    # Resize to input dimensions
    heatmap = F.interpolate(cam.unsqueeze(1), size=input_tensor.shape[2:], 
                            mode='bilinear', align_corners=False)
    
    return heatmap.squeeze().cpu().numpy()


def visualize_gradcam(model: torch.nn.Module, batch: Dict, device: torch.device,
                    target_class: int = 1, target_layer: torch.nn.Module = None) -> None:
    """
    Visualize Grad-CAM heatmaps for images in a batch.
    
    Args:
        model: Segmentation model
        batch: Batch of data
        device: Device to run on
        target_class: Class index for which to compute Grad-CAM
        target_layer: Model layer to target (defaults to first decoder stage)
    """
    # Set default target layer if not specified
    if target_layer is None:
        # Default to first decoder stage convolutional layer
        target_layer = model.decoder_stages[0].conv_block.block[0]
    
    class_names = ["Background", "Cat", "Dog"]
    target_class_name = class_names[target_class] if target_class < len(class_names) else f"Class {target_class}"
    
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    filenames = batch.get("filename", [f"Sample_{i}" for i in range(images.size(0))])
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
    
    # Visualize each image
    for i in range(images.size(0)):
        # Skip images that don't contain the target class
        mask = masks[i].cpu().numpy()
        if target_class not in np.unique(mask):
            continue
        
        # Generate Grad-CAM heatmap
        input_tensor = images[i:i+1]  # Add batch dimension
        heatmap = generate_gradcam_heatmap(model, input_tensor, target_class, target_layer, device)
        
        # Denormalize image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title(f"Original: {filenames[i]}")
        axes[0].axis("off")
        
        # Ground truth mask for the target class
        target_mask = np.zeros_like(mask)
        target_mask[mask == target_class] = 1
        axes[1].imshow(img)
        axes[1].imshow(target_mask, alpha=0.5, cmap='Reds')
        axes[1].set_title(f"Ground Truth: {target_class_name}")
        axes[1].axis("off")
        
        # Grad-CAM heatmap
        axes[2].imshow(img)
        cax = axes[2].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[2].set_title(f"Grad-CAM: {target_class_name}")
        axes[2].axis("off")
        
        # Add colorbar
        plt.colorbar(cax, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()


def plot_confusion_matrix(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray], 
                         class_names: List[str] = None) -> None:
    """
    Plot a confusion matrix for segmentation results.
    
    Args:
        pred_masks: List of predicted segmentation masks
        gt_masks: List of ground truth segmentation masks
        class_names: Optional list of class names
    """
    if class_names is None:
        class_names = ["Background", "Cat", "Dog"]
    
    num_classes = len(class_names)
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # Fill confusion matrix
    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        # Exclude ignored pixels (255)
        valid_mask = (gt_mask != 255)
        
        for gt_cls in range(num_classes):
            for pred_cls in range(num_classes):
                # Count pixels that are ground truth class 'gt_cls' and predicted as 'pred_cls'
                count = np.sum((gt_mask == gt_cls) & (pred_mask == pred_cls) & valid_mask)
                confusion_matrix[gt_cls, pred_cls] += count
    
    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set axis labels
    ax.set(xticks=np.arange(num_classes),
          yticks=np.arange(num_classes),
          xticklabels=class_names,
          yticklabels=class_names,
          xlabel="Predicted Label",
          ylabel="True Label",
          title="Confusion Matrix (Pixel Counts)")
    
    # Rotate the x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if confusion_matrix[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.show()
    
    # Also plot normalized confusion matrix (percentages)
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalized_cm = confusion_matrix / row_sums * 100
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(normalized_cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(num_classes),
          yticks=np.arange(num_classes),
          xticklabels=class_names,
          yticklabels=class_names,
          xlabel="Predicted Label",
          ylabel="True Label",
          title="Normalized Confusion Matrix (%)")
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(normalized_cm[i, j], '.1f') + '%',
                   ha="center", va="center",
                   color="white" if normalized_cm[i, j] > 50 else "black")
    
    fig.tight_layout()
    plt.show()