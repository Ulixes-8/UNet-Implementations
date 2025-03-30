# UNet Implementation for Pet Segmentation

This repository contains a professional implementation of UNet for semantic segmentation, specifically designed for the Oxford-IIIT Pet Dataset. The implementation combines the strengths of `segmentation-models-pytorch` with hyperparameters optimized through nnU-Net.

## Project Structure

```
UNet-Implementation/
├── config/
│   ├── model_config.yaml             # Model configuration generated from nnU-Net
│   └── model_config_template.yaml    # Template for configuration
├── data/                             # Dataset directory (processed by data_augmentation)
│   └── ...
├── data_augmentation/                # Data augmentation scripts (existing)
│   └── ...
├── models/                           # Directory for saved models
│   └── ...
├── scripts/
│   ├── extract_hyperparameters.py    # Extract nnU-Net hyperparameters
│   ├── train.py                      # Training script
│   └── predict.py                    # Prediction script
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── unet.py                   # UNet model implementation
│   └── utils/
│       └── __init__.py
├── requirements.txt                  # Project dependencies
└── README.md                         # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-enabled GPU (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/UNet-Implementation.git
   cd UNet-Implementation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install nnU-Net (optional, only required for hyperparameter optimization):
   ```bash
   pip install nnunetv2
   ```

## Workflow

### 1. Optimize Hyperparameters with nnU-Net (Optional)

This step uses nnU-Net to determine optimal hyperparameters for your dataset:

```bash
# Set environment variables
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

# Convert your data to nnU-Net format (example script needed for your specific data)
python scripts/convert_to_nnunet.py --data_dir data/processed --output_dir $nnUNet_raw/Dataset001_PetSegmentation

# Run nnU-Net planning and preprocessing
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

# Train nnU-Net models (this will take some time)
nnUNetv2_train 1 2d 0 --npz
nnUNetv2_train 1 2d 1 --npz
nnUNetv2_train 1 2d 2 --npz
nnUNetv2_train 1 2d 3 --npz
nnUNetv2_train 1 2d 4 --npz

# Find the best configuration
nnUNetv2_find_best_configuration 1 -c 2d
```

### 2. Extract Hyperparameters from nnU-Net

Extract the optimized hyperparameters:

```bash
python scripts/extract_hyperparameters.py \
    --plans_file $nnUNet_results/Dataset001_PetSegmentation/nnUNetTrainer__nnUNetPlans__2d/plans.json \
    --output_config config/model_config.yaml \
    --template_config config/model_config_template.yaml
```

### 3. Train the UNet Model

Train the model using the extracted hyperparameters:

```bash
python scripts/train.py \
    --config config/model_config.yaml \
    --data_dir data/processed \
    --output_dir models/unet_pet_segmentation \
    --num_workers 4
```

### 4. Run Inference

Make predictions on new images:

```bash
python scripts/predict.py \
    --config config/model_config.yaml \
    --checkpoint models/unet_pet_segmentation/best_model.pth \
    --input path/to/images \
    --output path/to/results \
    --visualize
```

## Model Architecture

The implementation uses the UNet architecture from `segmentation-models-pytorch` with:

- Encoder: ResNet34 (pretrained on ImageNet)
- Decoder: Standard UNet decoder with skip connections
- Loss Function: Combination of Dice Loss and Cross-Entropy
- Optimizer: Adam with learning rate from nnU-Net

## Data Preparation

This project assumes that your data has already been processed using the `data_augmentation` scripts, following the format specified in the Oxford-IIIT Pet Dataset.

## Customization

- **Different Encoder**: Change `encoder_name` in the configuration file
- **Number of Classes**: Adjust `classes` in the configuration file
- **Learning Rate**: Modify `learning_rate` in the configuration file
- **Data Augmentation**: Configure transforms in `src/models/unet.py`

## Acknowledgments

- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) for the UNet implementation
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) for the hyperparameter optimization
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) for the dataset