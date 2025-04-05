# AE_pretrained: Transfer Learning for Image Segmentation

## Overview

This repository contains a complete implementation of a transfer learning approach for pet image segmentation. The framework uses a pre-trained autoencoder as a feature extractor for a segmentation model, following a two-phase approach:

1. **Phase 1: Autoencoder Pretraining** - Train a convolutional autoencoder for image reconstruction on the Oxford-IIIT Pet Dataset
2. **Phase 2: Transfer Learning** - Use the pre-trained encoder as a feature extractor for a UNet segmentation model

The repository demonstrates how unsupervised pretraining can improve segmentation performance, particularly when labeled data is limited.

## Project Structure

```
AE_pretrained/
├── README.md                           # This file
├── config/                             # Configuration files
│   └── model_config.yaml               # Main model configuration
├── evaluation_results/                 # Results from model evaluation
│   ├── evaluation_results.json         # Segmentation metrics
│   └── reconstruction_metrics.json     # Autoencoder reconstruction metrics
├── reconstruction/                     # Autoencoder reconstruction module
│   ├── models/                         # Model definitions
│   │   ├── __init__.py                 # Model exports
│   │   ├── ae_pet_reconstruction/      # Trained autoencoder model
│   │   │   ├── training_config.json    # Training configuration
│   │   │   └── training_log.csv        # Training metrics log
│   │   ├── autoencoder.py              # Autoencoder model definition
│   │   ├── losses.py                   # Loss functions
│   │   └── unet.py                     # UNet model definition
│   ├── src/                            # Source code
│   │   ├── __init__.py                 # Package initialization
│   │   ├── evaluate.py                 # Evaluation script
│   │   ├── main.py                     # Main entry point
│   │   └── train.py                    # Training script
│   └── utils/                          # Utility functions
│       ├── __init__.py                 # Utility exports
│       ├── metrics.py                  # Evaluation metrics
│       └── visualize.py                # Visualization tools
└── transfer_learning/                  # Transfer learning module
    ├── models/                         # Model definitions for transfer learning
    │   ├── __init__.py                 # Model exports
    │   ├── losses.py                   # Loss functions for segmentation
    │   ├── unet.py                     # UNet model with transfer learning support
    │   └── unet_pet_segmentation_transfer/ # Trained segmentation model
    │       ├── training_config.json    # Training configuration
    │       └── training_log.csv        # Training metrics log
    ├── src/                            # Source code
    │   ├── __init__.py                 # Package initialization
    │   ├── evaluate.py                 # Evaluation script
    │   ├── main.py                     # Main entry point
    │   └── train.py                    # Training script
    └── utils/                          # Utility functions
        ├── __init__.py                 # Utility exports
        ├── metrics.py                  # Segmentation metrics
        └── visualize.py                # Visualization tools
```

## Key Components

### 1. Autoencoder Architecture

The autoencoder (`reconstruction/models/autoencoder.py`) uses a UNet-style architecture with:

- Encoder path with 6 stages of increasing feature dimensions
- Skip connections to preserve spatial information
- Decoder path with upsampling and skip connection fusion
- Spatial dropout for regularization (increasing rates with depth)
- Instance normalization and LeakyReLU activations

### 2. Segmentation Model

The UNet segmentation model (`transfer_learning/models/unet.py`) features:

- Ability to load and freeze pre-trained encoder weights 
- 3-class segmentation: background, cat, and dog
- Enhanced dropout patterns to prevent overfitting
- Handling of boundary pixels (255) in dataset

### 3. Loss Functions

- **Reconstruction Loss** (`reconstruction/models/losses.py`): Combined MSE, perceptual, and SSIM loss for autoencoder training
- **Segmentation Loss** (`transfer_learning/models/losses.py`): Combined Dice and weighted Cross-Entropy loss with dynamic class weighting

### 4. Evaluation Metrics

- **Reconstruction Metrics**: MSE, PSNR, SSIM
- **Segmentation Metrics**: Pixel accuracy, IoU, Dice coefficient, precision, recall

## Performance Results

### Autoencoder Reconstruction

| Metric | Value |
|--------|-------|
| MSE    | 0.0023 |
| PSNR   | 28.23 dB |
| SSIM   | 0.876 |

### Segmentation with Transfer Learning

| Metric | Value |
|--------|-------|
| Pixel Accuracy | 0.657 |
| Mean IoU | 0.333 |
| Mean Foreground Dice | 0.253 |

Class-specific metrics:

| Class | Dice | IoU | Precision | Recall |
|-------|------|-----|-----------|--------|
| Background | 0.794 | 0.658 | 0.828 | 0.763 |
| Cat | 0.000 | 0.000 | 0.000 | 0.000 |
| Dog | 0.507 | 0.339 | 0.395 | 0.707 |

## Usage

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
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
```

### Data Preparation

The code expects data in the following structure:

```
data/processed/
├── Train/
│   ├── resized/         # Training images
│   └── resized_label/   # Training masks
├── Val/
│   ├── resized/         # Validation images
│   └── processed_labels/ # Validation masks
└── Test/
    ├── resized/         # Test images
    └── masks/           # Test masks
```

### Phase 1: Training the Autoencoder

```bash
# Train the autoencoder
python AE_pretrained/reconstruction/src/main.py \
    --data_dir data/processed \
    --output_dir AE_pretrained/reconstruction/models/ae_pet_reconstruction \
    --batch_size 32 \
    --epochs 200 \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --amp

# Evaluate the autoencoder
python AE_pretrained/reconstruction/src/evaluate.py \
    --model_path AE_pretrained/reconstruction/models/ae_pet_reconstruction/best_model.pth \
    --data_dir data/processed \
    --visualize_samples 16 \
    --analyze_latent_space
```

### Phase 2: Training the Segmentation Model with Transfer Learning

```bash
# Train the segmentation model with transfer learning
python AE_pretrained/transfer_learning/src/main.py \
    --pretrained_encoder AE_pretrained/reconstruction/models/ae_pet_reconstruction/best_model.pth \
    --data_dir data/processed \
    --output_dir AE_pretrained/transfer_learning/models/unet_pet_segmentation_transfer \
    --batch_size 32 \
    --epochs 200 \
    --lr 0.005 \
    --weight_decay 0.0001 \
    --dice_weight 1.0 \
    --ce_weight 1.0 \
    --weighted_ce \
    --amp

# Evaluate the segmentation model
python AE_pretrained/transfer_learning/src/evaluate.py \
    --model_path AE_pretrained/transfer_learning/models/unet_pet_segmentation_transfer/best_model.pth \
    --data_dir data/processed \
    --visualize_samples 3
```

### Configuration

You can adjust training parameters through command-line arguments or by modifying the configuration files:

- `AE_pretrained/config/model_config.yaml`: Global configuration
- `AE_pretrained/reconstruction/models/ae_pet_reconstruction/training_config.json`: Autoencoder training config
- `AE_pretrained/transfer_learning/models/unet_pet_segmentation_transfer/training_config.json`: Segmentation training config

## Visualization

The framework includes extensive visualization tools:

- Reconstruction comparisons: Original vs. reconstructed images
- Latent space analysis: PCA and t-SNE visualizations of latent representations
- Segmentation visualization: Side-by-side comparisons of ground truth and predictions
- Error analysis: Color-coded error maps for segmentation results
- Confidence maps: Visualization of model confidence for each class
- Grad-CAM: Class activation maps to understand model focus areas

## Implementation Details

### Autoencoder Training

- Learning rate: 0.001
- Batch size: 32
- Optimizer: Adam with weight decay 1e-5
- Loss: Combined MSE, perceptual, and SSIM loss
- Early stopping with patience of 15 epochs
- Mixed precision training for speed

### Segmentation Training

- Learning rate: 0.005 with polynomial decay
- Batch size: 32
- Optimizer: SGD with momentum 0.99 and weight decay 0.0001
- Loss: Combined Dice and weighted Cross-Entropy
- Dynamic class weighting based on pixel frequency
- Encoder weights frozen during initial training
- Early stopping with patience of 15 epochs

## License

[MIT License](LICENSE)

## Acknowledgments

- The Oxford-IIIT Pet Dataset
- PyTorch framework
- Inspiration from nnU-Net architecture