# UNet Implementation for Pet Segmentation

This repository contains a custom UNet implementation designed for the Oxford-IIIT Pet Dataset segmentation task. Our UNet model features a reduced complexity architecture with enhanced regularization techniques to prevent overfitting while maintaining high performance.

## Architecture Overview

Our UNet implementation follows the classic encoder-decoder structure with skip connections, but incorporates several modern improvements:

- **Reduced Complexity**: 6-stage architecture instead of the traditional 8-stage design
- **Spatial Dropout**: Implemented at various rates throughout the network
- **Instance Normalization**: Used instead of Batch Normalization for better performance with smaller batch sizes
- **Leaky ReLU**: Non-linearity for all layers
- **Dynamic Loss Function**: Combined Dice and weighted Cross-Entropy loss with proper handling of border pixels

## Model Configuration

The model uses the following configuration by default:

```yaml
{
    "data_dir": "data/processed/",
    "output_dir": "models/unet_pet_segmentation",
    "batch_size": 32,
    "epochs": 200,
    "lr": 0.005,
    "weight_decay": 0.0001,
    "momentum": 0.99,
    "num_workers": 4,
    "save_every": 10,
    "patience": 15,
    "reduced_complexity": true,
    "device": "",
    "resume": "",
    "amp": true
}
```

## Performance

Our UNet implementation achieves the following metrics on the test set:

- **Pixel Accuracy**: 87.57%
- **Mean IoU**: 68.90%
- **Mean Foreground Dice**: 75.11%

Class-specific metrics:
- **Background**: Dice 92.60%, IoU 86.22%, Precision 87.85%, Recall 97.89%
- **Cat**: Dice 72.07%, IoU 56.33%, Precision 89.17%, Recall 60.46%
- **Dog**: Dice 78.16%, IoU 64.14%, Precision 85.72%, Recall 71.81%

## Directory Structure

```
Our_UNet/
├── config/                 # Configuration files
│   └── model_config.yaml
├── evaluation_results/     # Evaluation metrics and visualizations
│   └── evaluation_results.json
├── models/                 # Model definitions and checkpoints
│   ├── __init__.py
│   ├── losses.py           # Custom loss functions
│   ├── unet.py             # UNet model implementation
│   └── unet_pet_segmentation/
│       ├── training_config.json
│       └── training_log.csv
├── src/                    # Training and evaluation scripts
│   ├── __init__.py
│   ├── evaluate.py         # Evaluation script
│   ├── main.py             # Main entry point
│   └── train.py            # Training script
└── utils/                  # Utility functions
    ├── __init__.py
    ├── metrics.py          # Metrics calculation
    └── visualize.py        # Visualization utilities
```

## Key Features

### UNet Architecture

The architecture is defined in `models/unet.py` with the following key components:

- **ConvBlock**: Basic building block for the encoder path with normalization, activation, and dropout
- **UpBlock**: Building block for the decoder path with upsampling and skip connections
- **SpatialDropout2d**: Custom dropout that drops entire feature maps for better regularization

### Loss Function

We implement a custom loss function in `models/losses.py` that combines:

- **Dice Loss**: For better handling of class imbalance
- **Cross-Entropy Loss**: With dynamic class weighting based on pixel frequency
- **Ignore Index**: Proper handling of border pixels (255) that should be ignored

### Training Pipeline

The training script in `src/train.py` includes:

- **Early Stopping**: Prevents overfitting by monitoring validation metrics
- **Learning Rate Scheduling**: Polynomial decay as in the nnUNet paper
- **Mixed Precision Training**: Accelerates training on compatible hardware
- **Checkpointing**: Regular model saving and best model tracking

### Evaluation Tools

The evaluation script in `src/evaluate.py` provides:

- **Comprehensive Metrics**: Pixel accuracy, IoU, Dice, precision, recall
- **Visualizations**: Prediction overlays, confidence maps, error analysis
- **Confusion Matrix**: For understanding model confusion between classes

## Usage

### Training

```bash
python src/main.py --data_dir data/processed --output_dir models/unet_pet_segmentation
```

Additional training options:
```bash
python src/train.py --batch_size 32 --epochs 200 --lr 0.005 --weight_decay 0.0001 --amp
```

### Evaluation

```bash
python src/evaluate.py --model_path models/unet_pet_segmentation/best_model.pth --data_dir data/processed
```

## References

- Original UNet Paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Oxford-IIIT Pet Dataset: [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)