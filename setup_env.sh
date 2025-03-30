#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Set nnU-Net environment variables to repository locations
export nnUNet_raw=$SCRIPT_DIR/nnUNet/nnUNet_raw
export nnUNet_preprocessed=$SCRIPT_DIR/nnUNet/nnUNet_preprocessed
export nnUNet_results=$SCRIPT_DIR/nnUNet/nnUNet_results

echo "nnU-Net environment variables set to repository locations:"
echo "nnUNet_raw = $nnUNet_raw"
echo "nnUNet_preprocessed = $nnUNet_preprocessed"
echo "nnUNet_results = $nnUNet_results"
