#!/usr/bin/env python
"""
Main script to run UNet training for pet segmentation.
This script ensures all modules are properly imported and training begins.

Usage:
    python main.py --data_dir data/processed --output_dir models/unet_pet_segmentation
"""

import os
import sys
from pathlib import Path

# Ensure the directory is in the Python path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Import the training module
from train import main as train_main

if __name__ == "__main__":
    # Run the training
    train_main()