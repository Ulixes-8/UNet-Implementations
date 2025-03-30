# Oxford-IIIT Pet Dataset Augmentation for Semantic Segmentation

This repository contains a comprehensive pipeline for preprocessing and augmenting the Oxford-IIIT Pet Dataset for semantic segmentation tasks. The implementation focuses on class-balanced data augmentation strategies to address the inherent dog-to-cat class imbalance in the original dataset.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Data Processing Pipeline](#data-processing-pipeline)
4. [Class Balancing Philosophy](#class-balancing-philosophy)
5. [Augmentation Strategy Design](#augmentation-strategy-design)
6. [Mask Value Encoding](#mask-value-encoding)
7. [Installation](#installation)
8. [Usage Guide](#usage-guide)
9. [Implementation Notes](#implementation-notes)
10. [Results and Statistics](#results-and-statistics)
11. [Acknowledgments](#acknowledgments)

## Project Overview

The Oxford-IIIT Pet Dataset is a widely used benchmark for semantic segmentation tasks, providing images of cats and dogs with pixel-level segmentation masks. However, the dataset has a significant class imbalance with approximately 2:1 dog-to-cat ratio. This project addresses this imbalance through targeted data augmentation while maintaining semantic integrity and preserving mask label values.

### Key Features

- **Robust Dataset Processing**: Handles corrupt images and standardizes data format
- **Targeted Class Balancing**: Applies more aggressive augmentation to cats (minority class)
- **Semantic Preservation**: Maintains critical mask value encoding (0: background, 1: cat, 2: dog, 255: border)
- **Configurable Parameters**: Allows fine-tuning of augmentation strategies via YAML configuration
- **Comprehensive Pipeline**: Covers dataset download, preprocessing, mask processing, and augmentation

## Repository Structure

```
.
├── config/
│   └── augmentation_config.yaml    # Configuration for augmentation parameters
├── data/
│   └── processed/                  # Processed dataset (auto-generated)
├── src/
│   ├── augment_dataset.py          # Class-specific augmentation implementation
│   ├── dataset_analyzer.py         # Dataset analysis utilities
│   ├── debug_mask_values.py        # Tool for analyzing mask encoding
│   ├── download_and_extract.py     # Download dataset from Google Drive
│   ├── preprocess_dataset.py       # Split dataset and resize images
│   ├── preprocess_test_val_labels.py # Process val/test masks
│   └── preprocess_training_labels.py # Resize training masks to match images
└── utils/
    └── helpers.py                  # Utility functions
```

## Data Directory Structure
```
data/
└── processed/
    ├── Train/
    │   ├── augmented/
    │   │   ├── images/         # Augmented .jpg images (resized)
    │   │   └── masks/          # Corresponding augmented .png masks (resized)
    │   ├── color/              # Original training .jpg images
    │   ├── label/              # Original training .png masks
    │   ├── resized/            # Resized .jpg images (512x512)
    │   └── resized_label/      # Resized .png masks (512x512)
    │
    ├── Val/
    │   ├── color/              # Original validation .jpg images
    │   ├── label/              # Original validation .png masks
    │   ├── resized/            # Resized .jpg validation images
    │   └── processed_labels/   # Processed .png masks
    │
    └── Test/
        ├── color/              # Original test .jpg images
        ├── label/              # Original test .png masks
        ├── resized/            # Resized .jpg test images
        └── processed_labels/   # Processed .png test masks
```

## Data Processing Pipeline

The preprocessing pipeline consists of several sequential stages:

### 1. Dataset Download and Extraction

The `download_and_extract.py` script:
- Downloads the Oxford-IIIT Pet Dataset from Google Drive
- Extracts it into the `data/raw/` directory
- Sets up the base directory structure

### 2. Dataset Preprocessing

The `preprocess_dataset.py` script:
- Detects and removes corrupt images
- Creates a stratified train-validation split (default 80%-20%)
- Standardizes image sizes to 512×512 while preserving aspect ratio
- Preserves original mask files for accurate evaluation

### 3. Mask Processing

Two specialized scripts handle mask processing:

The `preprocess_training_labels.py` script:
- Resizes training masks to match the dimensions of resized images
- Preserves exact pixel-level class values (0, 1, 2, 255)
- Uses nearest-neighbor interpolation to avoid value distortion

The `preprocess_test_val_labels.py` script:
- Processes validation and test masks without resizing
- Handles proper class value encoding
- Maintains original dimensions for accurate evaluation

### 4. Data Augmentation

The `augment_dataset.py` script:
- Applies class-specific augmentation strategies
- Generates more augmentations for cats (minority class)
- Applies different augmentation parameters based on class
- Ensures proper mask value preservation through all transformations

## Class Balancing Philosophy

The augmentation strategy addresses the 2:1 dog-to-cat imbalance through a targeted approach:

### Statistical Analysis

Initial analysis of the dataset revealed:
- ~948 cats and ~1,991 dogs (approximately 2:1 ratio)
- This imbalance would bias models toward the majority class (dogs)

### Balancing Strategy

Our approach uses two primary techniques:
1. **Quantity-based balancing**: Generate more augmented versions of cat images
2. **Quality-based balancing**: Apply more aggressive augmentations to cats

By default, the system generates:
- 5 augmented versions of each cat image
- 2 augmented versions of each dog image

This results in a final balanced dataset of approximately:
- ~5,688 cats (948 original + 4,740 augmented)
- ~5,973 dogs (1,991 original + 3,982 augmented)

## Augmentation Strategy Design

The augmentation strategy was carefully designed to:
1. Preserve semantic meaning of mask labels
2. Maintain domain-specific visual characteristics
3. Create useful variations while avoiding unrealistic distortions

### Cat Augmentation (Minority Class)

For cats, we apply more aggressive augmentations to maximize diversity:

- **Spatial Transforms**: More extreme rotation, scale, and perspective changes
- **Elastic Transforms**: Include elastic deformations, grid distortions, and optical distortions
- **Color Transforms**: More pronounced hue, saturation, and brightness variations
- **Noise Effects**: More varied noise patterns and blur effects

### Dog Augmentation (Majority Class)

For dogs, we apply more conservative augmentations:

- **Spatial Transforms**: More moderate rotation and scale parameters
- **Elastic Transforms**: Less extreme deformations
- **Color Transforms**: Subtler color variations
- **Noise Effects**: More limited noise and blur effects

### Transformation Categories

The augmentation pipeline includes diverse transforms organized by category:

1. **Spatial Transforms**
   - Horizontal flip, rotation, scaling, shifting
   - Random resized crop with aspect ratio preservation
   - Perspective transforms

2. **Pixel-level Transforms**
   - Brightness and contrast adjustments
   - HSV shifts (hue, saturation, value)
   - RGB channel shifts
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)

3. **Noise and Blur Effects**
   - Gaussian noise, blur
   - Motion blur
   - Salt and pepper noise
   - ISO noise (camera sensor simulation)

4. **Lighting Simulations**
   - Random shadows
   - Sun flares
   - Fog effects

5. **Occlusion Effects**
   - Coarse dropout (simulates partial occlusion)

## Mask Value Encoding

The segmentation masks use a specific value encoding that must be preserved through all transformations:

- **0**: Background (pixels not belonging to an animal)
- **1**: Cat (pixels belonging to a cat)
- **2**: Dog (pixels belonging to a dog)
- **255**: Border/Don't care (pixels that should be ignored during training)

Maintaining these exact values is critical for proper model training and evaluation. All processing steps use nearest-neighbor interpolation for masks to prevent value distortion.

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU recommended for faster processing

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd <repository-dir>

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage Guide

### Complete Pipeline

To run the complete pipeline from scratch:

```bash
# 1. Download and extract the dataset
python src/download_and_extract.py

# 2. Preprocess the dataset (split and resize)
python src/preprocess_dataset.py --val-ratio 0.2 --size 512

# 3. Process training masks
python src/preprocess_training_labels.py --size 512

# 4. Process validation and test masks
python src/preprocess_test_val_labels.py

# 5. Run augmentation
python src/augment_dataset.py --config config/augmentation_config.yaml
```

### Customizing Augmentation

To customize the augmentation process:

1. **Edit Configuration**: Modify parameters in `config/augmentation_config.yaml`
2. **Change Augmentation Count**: Adjust how many augmentations to generate per class
   ```bash
   python src/augment_dataset.py --cat-augmentations 3 --dog-augmentations 1
   ```
3. **Enable Visualization**: Generate sample visualizations of augmentations
   ```bash
   python src/augment_dataset.py --visualize 10
   ```

### Dataset Analysis

To analyze the processed dataset statistics:

```bash
python src/dataset_analyzer.py data/processed
```

### Debugging Mask Values

To debug and verify mask values:

```bash
python src/debug_mask_values.py
```

## Implementation Notes

### Critical Design Decisions

1. **Mask Value Preservation**
   - All mask operations use `cv2.INTER_NEAREST` interpolation
   - Pixel values are explicitly preserved through all transformations

2. **Aspect Ratio Handling**
   - Images maintain aspect ratio and are padded to square dimensions
   - This prevents distortion of pet shapes while providing uniform inputs to models

3. **Split Strategy**
   - The train/validation split is stratified by class to maintain class proportions
   - Random seed ensures reproducibility of splits

4. **Augmentation Implementation**
   - Uses Albumentations library for efficient and reliable transformations
   - Custom pipelines for each class with tailored parameters

### Performance Considerations

- **Preprocessing Efficiency**: The pipeline includes progress bars and optimized operations
- **Memory Management**: Images are processed in batches to manage memory usage
- **Error Handling**: Robust error detection for corrupt images and masks

## Results and Statistics

### Before Augmentation
- Training set: ~2,939 images (~948 cats, ~1,991 dogs)
- Dog-to-cat ratio: Approximately 2:1

### After Augmentation
- Training set: ~11,661 images (~5,688 cats, ~5,973 dogs)
- Dog-to-cat ratio: Approximately 1:1

### Data Storage Requirements
- Original dataset: ~700 MB
- After preprocessing and augmentation: ~2.5-3.0 GB

## Acknowledgments

The Oxford-IIIT Pet Dataset was created by Parkhi et al:
- O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, "Cats and Dogs," IEEE Conference on Computer Vision and Pattern Recognition, 2012.

This augmentation project builds upon their valuable contribution to the computer vision community.