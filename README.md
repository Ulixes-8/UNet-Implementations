# UNet Semantic Segmentation Project

This repository contains multiple implementations of semantic segmentation models for the Oxford-IIIT Pet Dataset. The project explores different architectures and approaches to perform accurate segmentation of cats and dogs in images.

## Overview

The goal of this project is to develop and compare different segmentation models for the pet segmentation task. We provide several implementations:

1. **Our_UNet**: A custom UNet implementation with reduced complexity and enhanced regularization techniques
2. **CLIP_UNet**: A UNet variant that incorporates CLIP embeddings for improved semantic understanding
3. **AE_pretrained**: An autoencoder-based approach with pretrained components
4. **data_augmentation**: Tools and scripts for augmenting the pet dataset

## Dataset

This project uses the Oxford-IIIT Pet Dataset, which consists of images of cats and dogs with pixel-wise segmentation masks. The segmentation task involves classifying each pixel into one of three categories: background, cat, or dog.

The dataset is automatically downloaded and processed by the training scripts. The processed data follows this structure:
```
data/
└── processed/
    ├── Train/
    │   ├── resized/         # Resized images (512x512)
    │   ├── resized_label/   # Processed labels for training
    │   └── augmented/       # Augmented training data
    ├── Val/
    │   ├── resized/         # Validation images
    │   └── processed_labels/ # Validation labels
    └── Test/
        ├── resized/         # Test images
        └── processed_labels/ # Test labels
```

## Model Implementations

### Our_UNet

A customized UNet implementation with 6-stage encoder-decoder architecture, featuring:
- Spatial dropout regularization
- Instance normalization
- Combined Dice and weighted Cross-Entropy loss
- Improved training techniques

[Read more about Our_UNet](./Our_UNet/README.md)

### CLIP_UNet

A UNet variant enhanced with CLIP embeddings for improved semantic understanding:
- Leverages pretrained CLIP model for rich feature extraction
- Combines visual and semantic information
- Performs well on challenging examples

[Read more about CLIP_UNet](./CLIP_UNet/README.md)

### AE_pretrained

An autoencoder-based approach that utilizes pretrained components:
- Encoder-decoder architecture for segmentation
- Transfer learning from pretrained models
- Efficient latent space representation

[Read more about AE_pretrained](./AE_pretrained/README.md)

### Data Augmentation

Tools and techniques for augmenting the pet dataset:
- Various transformations to improve model generalization
- Custom augmentation pipeline
- Class-balanced augmentation strategies

[Read more about data_augmentation](./data_augmentation/README.md)

## Performance Comparison

| Model | Pixel Accuracy | Mean IoU | Mean Foreground Dice | Cat Dice | Dog Dice |
|-------|---------------|----------|----------------------|----------|----------|
| Our_UNet | 87.57% | 68.90% | 75.11% | 72.07% | 78.16% |
| CLIP_UNet | 89.32% | 71.46% | 78.52% | 75.89% | 81.15% |
| AE_pretrained | 86.21% | 67.33% | 73.82% | 70.45% | 77.19% |

Class-specific metrics for Our_UNet:
- **Background**: Dice 92.60%, IoU 86.22%, Precision 87.85%, Recall 97.89%
- **Cat**: Dice 72.07%, IoU 56.33%, Precision 89.17%, Recall 60.46%
- **Dog**: Dice 78.16%, IoU 64.14%, Precision 85.72%, Recall 71.81%

## Installation

1. Clone the repository:
```bash
git clone https://github.com/username/unet.git
cd unet
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Training Models

Each sub-repository contains its own training scripts. For example, to train Our_UNet:

```bash
cd Our_UNet
python src/main.py --data_dir ../data/processed --output_dir models/unet_pet_segmentation
```

### Evaluation

To evaluate a trained model:

```bash
cd Our_UNet
python src/evaluate.py --model_path models/unet_pet_segmentation/best_model.pth --data_dir ../data/processed
```

## Project Structure

```
unet/
├── .gitignore
├── README.md
├── requirements.txt
├── Our_UNet/                # Custom UNet implementation
├── CLIP_UNet/               # CLIP-enhanced UNet
├── AE_pretrained/           # Autoencoder-based approach
└── data_augmentation/       # Data augmentation tools
```

## Dependencies

This project requires the following main dependencies:
- PyTorch (>= 2.6.0)
- torchvision
- OpenCV
- NumPy
- Matplotlib
- albumentations
- tqdm

See `requirements.txt` for the complete list of dependencies.

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

## License

This project is licensed under the MIT License - see the LICENSE file for details.