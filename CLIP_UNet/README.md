# CLIP_UNet: Vision-Language Enhanced Segmentation

## Overview

This repository contains a complete implementation of CLIP-enhanced UNet for pet image segmentation. The framework leverages the semantic understanding capabilities of OpenAI's CLIP (Contrastive Language-Image Pretraining) model to improve segmentation performance on the Oxford-IIIT Pet Dataset.

The approach integrates CLIP's visual token features at the UNet bottleneck, allowing the segmentation model to benefit from CLIP's powerful, pretrained visual representations without requiring additional labeled data. This fusion of vision-language representations with traditional segmentation architecture demonstrates significant improvements in segmentation accuracy.

## Project Structure

```
CLIP_UNet/
├── README.md                           # This file
├── config/                             # Configuration files
│   └── model_config.yaml               # Main model configuration
├── evaluation_results/                 # Results from model evaluation
│   └── evaluation_results.json         # Segmentation performance metrics
├── models/                             # Model definitions and checkpoints
│   ├── __init__.py                     # Model exports
│   ├── clip_unet_pet_segmentation/     # Trained model directory
│   │   ├── training_config.json        # Training configuration
│   │   └── training_log.csv            # Training metrics log
│   ├── losses.py                       # Loss function definitions
│   └── unet.py                         # CLIP-enhanced UNet implementation
├── scripts/                            # Utility scripts
│   ├── __init__.py
│   └── create_clip_resized_images.py   # Script to create 224x224 images for CLIP
├── src/                                # Source code
│   ├── __init__.py
│   ├── evaluate.py                     # Evaluation script
│   ├── main.py                         # Main entry point
│   └── train.py                        # Training script
└── utils/                              # Utility functions
    ├── __init__.py
    ├── metrics.py                      # Evaluation metrics
    └── visualize.py                    # Visualization tools
```

## Key Components

### 1. CLIP-Enhanced UNet Architecture

The core of this repository is a modified UNet architecture (`models/unet.py`) with CLIP integration:

- **Feature Fusion**: Incorporates CLIP's patch token features at the UNet bottleneck
- **ClipPatchExtractor**: Custom module to extract and reshape CLIP's visual features
- **Adaptive Fusion Layer**: Dynamically adapts to combine UNet and CLIP feature dimensions
- **Spatial Dropout**: Gradually increasing/decreasing dropout rates through encoder/decoder

### 2. Loss Functions

The segmentation training uses a combined loss approach (`models/losses.py`):

- **Dice Loss**: For better handling of class imbalance
- **Weighted Cross-Entropy Loss**: With dynamic class weighting based on pixel frequency
- **Ignore Index Handling**: Proper handling of boundary pixels (255) in the dataset

### 3. CLIP Integration

The framework handles CLIP feature extraction efficiently:

- **Pre-resizing Script**: Creates 224x224 images specifically for CLIP processing
- **Dynamic Feature Resolution**: Adapts to different CLIP model variants (ViT-B/16, ViT-B/32, ViT-L/14)
- **Gradient Isolation**: CLIP features are extracted without requiring gradients

### 4. Visualization Tools

Comprehensive visualization utilities (`utils/visualize.py`):

- **Prediction Visualization**: Side-by-side comparisons of ground truth and predictions
- **Confidence Maps**: Heatmaps showing model confidence for each class
- **Error Analysis**: Color-coded visualization of correct predictions, false positives, etc.
- **Grad-CAM Visualization**: Shows which image regions influence class predictions

## Performance Results

The CLIP-enhanced UNet achieves significant improvement over baseline models:

| Metric | Value |
|--------|-------|
| Pixel Accuracy | 0.818 |
| Mean IoU | 0.597 |
| Mean Foreground Dice | 0.657 |

Class-specific metrics:

| Class | Dice | IoU | Precision | Recall |
|-------|------|-----|-----------|--------|
| Background | 0.894 | 0.809 | 0.884 | 0.905 |
| Cat | 0.615 | 0.444 | 0.607 | 0.622 |
| Dog | 0.699 | 0.537 | 0.730 | 0.670 |

Training progression shows continual improvement over 54 epochs, reaching a foreground Dice score of 0.588 before early stopping.

## Usage

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CLIP (OpenAI)
- OpenCV
- NumPy
- Matplotlib
- tqdm

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/segmentation_cv.git
cd segmentation_cv

# Install dependencies
pip install -r requirements.txt

# Install CLIP
pip install git+https://github.com/openai/CLIP.git
```

### Data Preparation

The code expects the Oxford-IIIT Pet Dataset in the following structure:

```
data/processed/
├── Train/
│   ├── resized/         # Training images (512x512)
│   ├── resized_clip/    # Images resized for CLIP (224x224)
│   └── resized_label/   # Training masks (512x512)
├── Val/
│   ├── resized/         # Validation images
│   ├── resized_clip/    # CLIP-sized validation images 
│   └── processed_labels/ # Validation masks
└── Test/
    ├── resized/         # Test images
    ├── resized_clip/    # CLIP-sized test images
    └── masks/           # Test masks
```

To create the CLIP-resized images:

```bash
python CLIP_UNet/scripts/create_clip_resized_images.py --processed-dir data/processed
```

### Training the CLIP-Enhanced UNet

```bash
python CLIP_UNet/src/main.py \
    --data_dir data/processed \
    --output_dir CLIP_UNet/models/clip_unet_pet_segmentation \
    --batch_size 16 \
    --epochs 200 \
    --lr 0.005 \
    --weight_decay 0.0001 \
    --use_clip \
    --clip_model "ViT-B/16" \
    --amp
```

### Evaluating the Model

```bash
python CLIP_UNet/src/evaluate.py \
    --model_path CLIP_UNet/models/clip_unet_pet_segmentation/best_model.pth \
    --data_dir data/processed \
    --batch_size 4 \
    --visualize_samples 3
```

### Configuration

You can adjust model parameters through command-line arguments or by modifying:

- `CLIP_UNet/config/model_config.yaml`: Global configuration
- `CLIP_UNet/models/clip_unet_pet_segmentation/training_config.json`: Training configuration

## Implementation Details

### CLIP Feature Extraction

The model extracts CLIP features in a memory-efficient way:

```python
# Extract CLIP features with no gradients
with torch.no_grad():
    clip_features = clip_extractor(clip_images)

# Forward pass with CLIP features
outputs = model(images, clip_features)
```

### Feature Fusion

The UNet bottleneck fuses CLIP features with CNN features:

```python
# Concat and fuse features at the bottleneck
x = torch.cat([x, clip_features], dim=1)
x = self.clip_fusion_conv(x)
```

### Dynamic Class Weighting

Loss computation uses inverse frequency weighting to address class imbalance:

```python
# Compute weights based on class frequency in the batch
weights = total_pixels / class_pixels
```

### Training Optimizations

- **Mixed Precision**: Uses Automatic Mixed Precision (AMP) for faster training
- **Polynomial Learning Rate**: Decays learning rate using (1 - epoch/max_epochs)^0.9
- **Early Stopping**: Prevents overfitting with 15-epoch patience
- **Spatial Dropout**: Applies dropout to entire feature channels

## Results and Analysis

The model shows significant improvement in segmentation quality:

- **Background Class**: Very high accuracy (Dice 0.894) due to abundant examples
- **Cat Class**: More challenging (Dice 0.615) with greater appearance variation
- **Dog Class**: Better performance than cats (Dice 0.699)

The CLIP integration particularly helps with difficult examples where standard CNN features struggle to distinguish between similar appearances of cats and dogs.

## Visualization Examples

During evaluation, the framework can generate:

1. **Side-by-side comparisons** of original images, ground truth, and predictions
2. **Confidence maps** showing probability distributions for each class
3. **Error analysis** with color-coded regions (TP, FP, FN, wrong class)
4. **Grad-CAM visualizations** to understand model attention

## Acknowledgments

- The Oxford-IIIT Pet Dataset
- OpenAI's CLIP model
- PyTorch framework
- nnU-Net architecture for medical image segmentation