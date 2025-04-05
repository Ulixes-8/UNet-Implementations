# Data Augmentation Framework for Pet Segmentation

## Overview

This repository contains a comprehensive data augmentation framework for the Oxford-IIIT Pet Dataset, designed to address class imbalance and improve model generalization for image segmentation tasks. The framework implements class-specific augmentation strategies, with more aggressive transformations for minority classes (cats) and conservative ones for majority classes (dogs).

The data augmentation pipeline is highly configurable and includes:
- Geometric transformations (rotation, scaling, perspective)
- Pixel-level modifications (brightness, contrast, hue, saturation)
- Realistic noise simulation (motion blur, Gaussian noise, ISO noise)
- Occlusion handling (coarse dropout, random shadows)
- Elastic deformations (grid distortion, optical distortion)

## Project Structure

```
data_augmentation/
├── README.md                           # This file
├── config/                             # Configuration files
│   └── augmentation_config.yaml        # Detailed augmentation parameters
├── src/                                # Source code
│   ├── __init__.py                     # Package initialization
│   ├── augment_dataset.py              # Main augmentation script
│   ├── dataset_analyzer.py             # Dataset statistics and analysis
│   ├── debug_mask_values.py            # Debugging tools for mask values
│   ├── download_and_extract.py         # Dataset download utility
│   ├── preprocess_dataset.py           # Initial dataset preprocessing
│   ├── preprocess_test_val_labels.py   # Test/val mask processing
│   └── preprocess_training_labels.py   # Training mask processing
└── utils/                              # Utility functions
    ├── __init__.py                     # Package initialization
    └── helpers.py                      # Common helper functions
```

## Key Components

### 1. Dataset Preprocessing

Before augmentation, the framework ensures proper preprocessing of the Oxford-IIIT Pet Dataset through several steps:

- **Data Downloading** (`download_and_extract.py`): Downloads the dataset from Google Drive and extracts it.
- **Initial Processing** (`preprocess_dataset.py`): 
  - Detects and removes corrupt images
  - Performs stratified train/validation split
  - Standardizes image sizes with aspect ratio preservation
  - Creates properly structured directories
- **Mask Processing** (`preprocess_training_labels.py` and `preprocess_test_val_labels.py`):
  - Ensures proper class encoding (0=background, 1=cat, 2=dog, 255=border)
  - Handles resizing and padding of masks while preserving class values
  - Provides visualization tools for verification

### 2. Class-Specific Augmentation

The framework implements differential augmentation based on class to address imbalance:

- **Cat Augmentation** (Aggressive):
  - Higher transformation probabilities
  - Wider range of color/brightness variations
  - More aggressive elastic deformations
  - Multiple augmented versions per original image

- **Dog Augmentation** (Conservative):
  - Lower transformation probabilities
  - Narrower range of variations
  - Milder deformations
  - Fewer augmented versions per original image

### 3. Advanced Configuration

The `augmentation_config.yaml` file provides granular control over:

- Transform probabilities for each class
- Transformation parameters (intensity, limits, etc.)
- Number of augmentations per source image
- Visualization options

### 4. Visualization and Quality Control

The framework includes several tools for monitoring and verifying:

- **Dataset Analyzer** (`dataset_analyzer.py`): Provides comprehensive statistics about the dataset.
- **Mask Debugging** (`debug_mask_values.py`): Analyzes pixel values in masks to ensure proper encoding.
- **Augmentation Visualization**: Creates side-by-side comparisons of original and augmented images/masks.

## Usage

### Prerequisites

- Python 3.8+
- OpenCV
- NumPy
- Albumentations
- PIL (Pillow)
- tqdm
- PyYAML
- gdown (for downloading)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/segmentation_cv.git
cd segmentation_cv

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

The framework expects the Oxford-IIIT Pet Dataset to be organized as follows:

```
data/
├── raw/                # Original downloaded dataset
└── processed/          # Preprocessed dataset ready for training
    ├── Train/          # Training data
    │   ├── color/      # Original RGB images
    │   ├── label/      # Original segmentation masks
    │   ├── resized/    # Resized images (512×512)
    │   └── resized_label/ # Resized masks (512×512)
    ├── Val/            # Validation data with similar structure
    └── Test/           # Test data with similar structure
```

To download and preprocess the dataset:

```bash
# Download the dataset
python data_augmentation/src/download_and_extract.py

# Preprocess the dataset
python data_augmentation/src/preprocess_dataset.py --val-ratio 0.2 --size 512

# Process training masks
python data_augmentation/src/preprocess_training_labels.py --size 512

# Process validation and test masks
python data_augmentation/src/preprocess_test_val_labels.py --visualize 3
```

### Running Data Augmentation

To run the augmentation with default settings:

```bash
python data_augmentation/src/augment_dataset.py \
    --processed-dir data/processed \
    --config data_augmentation/config/augmentation_config.yaml \
    --cat-augmentations 5 \
    --dog-augmentations 2 \
    --visualize 5
```

### Analyzing the Dataset

To get comprehensive statistics about your dataset:

```bash
python data_augmentation/src/dataset_analyzer.py data/processed \
    --img-dir resized \
    --mask-dir resized_label
```

## Augmentation Pipeline Details

### Image Transformations

The augmentation pipeline applies a variety of transformations:

1. **Spatial Transformations**
   - Horizontal flipping
   - Rotation (±15° for cats, ±10° for dogs)
   - Scaling (±15% for cats, ±10% for dogs)
   - Shifting (±10% for cats, ±5% for dogs)
   - Perspective changes

2. **Pixel-level Transformations**
   - Brightness/contrast adjustments
   - Hue/saturation/value modifications
   - RGB channel shifts
   - CLAHE and equalization

3. **Noise and Blur**
   - Gaussian noise
   - Motion blur
   - Salt and pepper noise
   - ISO noise (camera sensor simulation)

4. **Occlusion Simulation**
   - Coarse dropout (random black squares)
   - Random shadows
   - Random sun flare

5. **Elastic Deformations**
   - Elastic transform
   - Grid distortion
   - Optical distortion

### Quality Preservation

Special care is taken to preserve mask quality during augmentation:
- Nearest-neighbor interpolation for masks
- Proper handling of class values (0, 1, 2, 255)
- Verification of augmented masks to ensure class integrity

## Implementation Notes

### Class Detection

The framework employs multiple strategies to determine if an image contains a cat or dog:
1. Analyze mask pixel values (most reliable method)
2. Check filename for breed name patterns
3. Fall back to manual annotation if necessary

### Augmentation Ratios

To address class imbalance, the default configuration generates:
- 5 augmented versions of each cat image
- 2 augmented versions of each dog image

This helps balance the training set while maintaining diversity.

## Results

The augmentation framework typically yields:
- A 3-5× increase in training data volume
- Better balance between cat and dog classes
- Improved model generalization and robustness
- Reduced overfitting in segmentation models

## Advanced Configuration

For fine-grained control, modify the `augmentation_config.yaml` file:

```yaml
cat:
  # Spatial transforms
  horizontal_flip_prob: 0.5
  scale_limit: 0.15
  rotate_limit: 15
  # More parameters...

dog:
  # More conservative parameters
  horizontal_flip_prob: 0.5
  scale_limit: 0.1
  rotate_limit: 10
  # More parameters...
```

## Acknowledgments

- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [Albumentations library](https://albumentations.ai/) for efficient augmentation implementation
- [PyTorch](https://pytorch.org/) ecosystem