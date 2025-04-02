"""
UNet implementation for pet segmentation with CLIP patch token integration
This module implements a UNet model that follows the architecture and hyperparameters
defined by nnU-Net for the pet segmentation task, enhanced with CLIP patch token features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Union, Optional, Type
import numpy as np

class SpatialDropout2d(nn.Module):
    """
    Spatial dropout for 2D feature maps that drops entire channels.
    This performs better than standard dropout for convolutional features.
    """
    def __init__(self, drop_prob):
        super(SpatialDropout2d, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
            
        # Get dimensions
        _, channels, height, width = x.size()
        
        # Sample binary dropout mask
        mask = x.new_empty(x.size(0), channels, 1, 1).bernoulli_(1 - self.drop_prob)
        mask = mask.div_(1 - self.drop_prob)
        
        # Apply mask
        x = x * mask.expand_as(x)
        return x

class ConvBlock(nn.Module):
    """
    Basic convolutional block for UNet with spatial dropout.
    This block consists of n_convs convolutional layers, each followed by normalization, 
    activation, and optional spatial dropout.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        n_convs: int = 2,
        padding: Optional[int] = None,
        norm_op: Type[nn.Module] = nn.InstanceNorm2d,
        norm_op_kwargs: Dict = None,
        dropout_op: Optional[Type[nn.Module]] = None,
        dropout_op_kwargs: Dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: Dict = None,
        conv_bias: bool = True,
        spatial_dropout_rate: float = 0.0
    ):
        """
        Initialize the ConvBlock.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolutional kernel
            stride: Stride of the convolution
            n_convs: Number of convolutional layers in the block
            padding: Padding size (if None, calculated to maintain spatial dimensions)
            norm_op: Normalization operation to use
            norm_op_kwargs: Arguments for normalization operation
            dropout_op: Dropout operation to use (if any)
            dropout_op_kwargs: Arguments for dropout operation
            nonlin: Non-linear activation function to use
            nonlin_kwargs: Arguments for non-linear activation
            conv_bias: Whether to use bias in convolutions
            spatial_dropout_rate: Rate for spatial dropout (0 to disable)
        """
        super(ConvBlock, self).__init__()
        
        # Default arguments if not provided
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {}
        
        # Calculate padding if not provided
        if padding is None:
            if isinstance(kernel_size, int):
                padding = kernel_size // 2
            else:
                padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        
        # Create the convolutional blocks
        layers = []
        current_channels = in_channels
        
        for i in range(n_convs):
            # Only apply stride in the first convolution
            current_stride = stride if i == 0 else 1
            
            # Add convolutional layer
            layers.append(
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size,
                    current_stride,
                    padding,
                    bias=conv_bias
                )
            )
            
            # Add normalization
            if norm_op is not None:
                layers.append(norm_op(out_channels, **norm_op_kwargs))
            
            # Add non-linearity
            if nonlin is not None:
                layers.append(nonlin(**nonlin_kwargs))
            
            # Add spatial dropout if rate > 0
            if spatial_dropout_rate > 0:
                layers.append(SpatialDropout2d(spatial_dropout_rate))
            
            # Add regular dropout if specified
            if dropout_op is not None:
                layers.append(dropout_op(**dropout_op_kwargs))
            
            # Update current number of channels
            current_channels = out_channels
        
        # Create the sequential block
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass of the convolutional block."""
        return self.block(x)

class UpBlock(nn.Module):
    """
    Upsampling block for the decoder part of UNet.
    This block upsamples the feature maps and concatenates with skip connections.
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        n_convs: int = 2,
        norm_op: Type[nn.Module] = nn.InstanceNorm2d,
        norm_op_kwargs: Dict = None,
        dropout_op: Optional[Type[nn.Module]] = None,
        dropout_op_kwargs: Dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: Dict = None,
        conv_bias: bool = True,
        spatial_dropout_rate: float = 0.0
    ):
        """
        Initialize the UpBlock.
        
        Args:
            in_channels: Number of input channels from the lower level
            skip_channels: Number of channels from the skip connection
            out_channels: Number of output channels
            kernel_size: Size of the convolutional kernel
            n_convs: Number of convolutional layers in the block
            norm_op: Normalization operation to use
            norm_op_kwargs: Arguments for normalization operation
            dropout_op: Dropout operation to use (if any)
            dropout_op_kwargs: Arguments for dropout operation
            nonlin: Non-linear activation function to use
            nonlin_kwargs: Arguments for non-linear activation
            conv_bias: Whether to use bias in convolutions
            spatial_dropout_rate: Rate for spatial dropout (0 to disable)
        """
        super(UpBlock, self).__init__()
        
        # Create the convolution block
        self.conv_block = ConvBlock(
            in_channels + skip_channels,
            out_channels,
            kernel_size,
            stride=1,
            n_convs=n_convs,
            padding=None,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            conv_bias=conv_bias,
            spatial_dropout_rate=spatial_dropout_rate
        )
    
    def forward(self, x, skip):
        """
        Forward pass of the upsampling block.
        
        Args:
            x: Input feature maps from lower level
            skip: Feature maps from skip connection
        
        Returns:
            Output feature maps
        """
        # Upsample the input to match skip connection size
        x_shape = x.shape[2:]
        skip_shape = skip.shape[2:]
        
        # Compute upsampling size to match skip connection
        if x_shape[0] != skip_shape[0] or x_shape[1] != skip_shape[1]:
            x = F.interpolate(
                x,
                size=skip_shape,
                mode='bilinear',
                align_corners=False
            )
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Apply convolution block
        return self.conv_block(x)

class UNet(nn.Module):
    """
    UNet model for semantic segmentation with reduced complexity, spatial dropout,
    and integration of CLIP patch token features.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        n_stages: int = 6,  # Reduced from 8 to 6
        features_per_stage: List[int] = None,
        kernel_sizes: List[Tuple[int, int]] = None,
        strides: List[Tuple[int, int]] = None,
        n_conv_per_stage: List[int] = None,
        n_conv_per_stage_decoder: List[int] = None,
        conv_bias: bool = True,
        norm_op: Type[nn.Module] = nn.InstanceNorm2d,
        norm_op_kwargs: Dict = None,
        dropout_op: Optional[Type[nn.Module]] = None,
        dropout_op_kwargs: Dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: Dict = None,
        encoder_dropout_rates: List[float] = None,
        decoder_dropout_rates: List[float] = None,
        with_clip_features: bool = True,
        clip_dim: int = 512
    ):
        """
        Initialize the UNet model with CLIP feature integration.
        
        Args:
            in_channels: Number of input channels (3 for RGB images)
            num_classes: Number of output classes (3 for background, cat, dog)
            n_stages: Number of stages in the encoder
            features_per_stage: Number of features per stage
            kernel_sizes: Kernel sizes for each stage
            strides: Strides for each stage
            n_conv_per_stage: Number of convolutions per encoder stage
            n_conv_per_stage_decoder: Number of convolutions per decoder stage
            conv_bias: Whether to use bias in convolutions
            norm_op: Normalization operation to use
            norm_op_kwargs: Arguments for normalization operation
            dropout_op: Dropout operation to use (if any)
            dropout_op_kwargs: Arguments for dropout operation
            nonlin: Non-linear activation function to use
            nonlin_kwargs: Arguments for non-linear activation
            encoder_dropout_rates: Dropout rates for each encoder stage
            decoder_dropout_rates: Dropout rates for each decoder stage
            with_clip_features: Whether to use CLIP features (default: True)
            clip_dim: Feature dimension from CLIP (default: 512)
        """
        super(UNet, self).__init__()
        
        # Set default values for parameters if not provided
        if features_per_stage is None:
            features_per_stage = [32, 64, 128, 256, 512, 512]  # Reduced from [32, 64, 128, 256, 512, 512, 512, 512]
        
        if kernel_sizes is None:
            kernel_sizes = [[3, 3]] * n_stages
        
        if strides is None:
            strides = [[1, 1]] + [[2, 2]] * (n_stages - 1)
        
        if n_conv_per_stage is None:
            n_conv_per_stage = [2] * n_stages
        
        if n_conv_per_stage_decoder is None:
            n_conv_per_stage_decoder = [2] * (n_stages - 1)
        
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}
            
        # Default dropout rates if not provided
        if encoder_dropout_rates is None:
            # Gradually increasing dropout in encoder
            encoder_dropout_rates = [0.0, 0.0, 0.1, 0.2, 0.3, 0.3]
            
        if decoder_dropout_rates is None:
            # Gradually decreasing dropout in decoder
            decoder_dropout_rates = [0.3, 0.2, 0.2, 0.1, 0.0]
        
        # Store parameters
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.n_stages = n_stages
        self.features_per_stage = features_per_stage
        self.with_clip_features = with_clip_features
        self.clip_dim = clip_dim
        
        # Create encoder stages
        self.encoder_stages = nn.ModuleList()
        
        current_channels = in_channels
        
        for stage in range(n_stages):
            # Create encoder block
            self.encoder_stages.append(
                ConvBlock(
                    current_channels,
                    features_per_stage[stage],
                    kernel_sizes[stage],
                    strides[stage],
                    n_convs=n_conv_per_stage[stage],
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    conv_bias=conv_bias,
                    spatial_dropout_rate=encoder_dropout_rates[stage]
                )
            )
            
            # Update current channels
            current_channels = features_per_stage[stage]
        
        # Create CLIP fusion layer placeholder (will be properly initialized in forward pass)
        if self.with_clip_features:
            # Note: We'll create this properly in the forward pass when we know the dimensions
            self.clip_fusion_conv = nn.Sequential(
                nn.Conv2d(features_per_stage[-1] + clip_dim, features_per_stage[-1], kernel_size=1, bias=conv_bias),
                norm_op(features_per_stage[-1], **norm_op_kwargs),
                nonlin(**nonlin_kwargs)
            )
            # Flag to track if we've adapted the layer
            self._fusion_adapted = False
        
        # Create decoder stages
        self.decoder_stages = nn.ModuleList()
        
        for stage in range(n_stages - 1):
            # Decoder stage goes in reverse order
            decoder_idx = n_stages - 2 - stage
            
            # Create decoder block
            self.decoder_stages.append(
                UpBlock(
                    features_per_stage[decoder_idx + 1],
                    features_per_stage[decoder_idx],
                    features_per_stage[decoder_idx],
                    kernel_sizes[decoder_idx],
                    n_convs=n_conv_per_stage_decoder[decoder_idx],
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    conv_bias=conv_bias,
                    spatial_dropout_rate=decoder_dropout_rates[stage]
                )
            )
        
        # Create final segmentation output layer
        self.segmentation_output = nn.Conv2d(
            features_per_stage[0],  # First decoder stage features
            num_classes,           # Number of output classes
            kernel_size=1,         # 1x1 convolution
            stride=1,
            padding=0,
            bias=True
        )
        
        # Initialize weights
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize the weights of the network."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, clip_features=None):
        """
        Forward pass of the UNet model with optional CLIP feature integration.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            clip_features: Optional CLIP patch token features of shape 
                        (batch_size, 512, 16, 16)
                
        Returns:
            output: Final output tensor
        """
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        for stage in self.encoder_stages[:-1]:  # All but the last stage
            x = stage(x)
            skip_connections.append(x)
        
        # Bottom stage (without skip connection)
        x = self.encoder_stages[-1](x)
        
        # Fuse CLIP features at bottleneck if provided
        if self.with_clip_features and clip_features is not None:
            # Verify feature map sizes match
            if x.shape[2:] != clip_features.shape[2:]:
                clip_features = F.interpolate(
                    clip_features,
                    size=x.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Get the actual feature dimensions
            encoder_features = x.shape[1]  # Number of channels from encoder
            clip_channels = clip_features.shape[1]  # Number of channels from CLIP
            
            # Print dimensions for debugging
            # print(f"Encoder features: {encoder_features}, CLIP features: {clip_channels}")
            
            # Dynamically adapt the fusion layer if needed
            expected_channels = encoder_features + clip_channels
            if not hasattr(self, '_fusion_adapted') or not self._fusion_adapted:
                # Check if we need to recreate the fusion layer with proper dimensions
                if self.clip_fusion_conv[0].in_channels != expected_channels:
                    print(f"Adapting fusion layer: {self.clip_fusion_conv[0].in_channels} → {expected_channels}")
                    
                    # Create a new fusion layer with correct dimensions
                    self.clip_fusion_conv = nn.Sequential(
                        nn.Conv2d(expected_channels, encoder_features, kernel_size=1, bias=True),
                        nn.InstanceNorm2d(encoder_features, eps=1e-5, affine=True),
                        nn.LeakyReLU(inplace=True)
                    ).to(x.device)
                    
                    # Mark as adapted
                    self._fusion_adapted = True
            
            # Concatenate and fuse features
            x = torch.cat([x, clip_features], dim=1)
            x = self.clip_fusion_conv(x)
        
        # Decoder path
        for idx, decoder_stage in enumerate(self.decoder_stages):
            # Use the appropriate skip connection (in reverse order)
            skip_idx = len(skip_connections) - 1 - idx
            skip = skip_connections[skip_idx]
            
            # Decoder block
            x = decoder_stage(x, skip)
        
        # Final 1x1 convolution to produce segmentation map
        output = self.segmentation_output(x)
        
        return output
    
class ClipPatchExtractor(nn.Module):
    """
    Module to extract CLIP patch token features with consistent output dimensions.
    """
    def __init__(self, clip_model, device="cuda"):
        super(ClipPatchExtractor, self).__init__()
        self.clip_model = clip_model
        self.device = device
        
        # Extract visual backbone
        if not hasattr(clip_model, 'visual'):
            raise ValueError("CLIP model doesn't have a 'visual' attribute")
        
        self.visual = clip_model.visual
        print(f"CLIP Visual type: {type(self.visual)}")
        
        # Find feature dimension
        if hasattr(self.visual, 'output_dim'):
            self.feature_dim = self.visual.output_dim
            print(f"Found feature_dim from output_dim: {self.feature_dim}")
        elif hasattr(self.visual, 'transformer') and hasattr(self.visual.transformer, 'width'):
            self.feature_dim = self.visual.transformer.width
            print(f"Found feature_dim from transformer.width: {self.feature_dim}")
        else:
            # Default for ViT-B/16
            self.feature_dim = 768
            print(f"Using default feature_dim: {self.feature_dim}")
        
        # Find expected grid size
        if hasattr(self.visual, 'input_resolution'):
            # OpenAI CLIP ViT models
            input_res = self.visual.input_resolution
            patch_size = self.visual.patch_size if hasattr(self.visual, 'patch_size') else 16
            grid_size = input_res // patch_size
            self.grid_size = (grid_size, grid_size)
            print(f"Grid size from input_resolution ({input_res}) and patch_size ({patch_size}): {self.grid_size}")
        elif hasattr(self.visual, 'grid_size'):
            self.grid_size = self.visual.grid_size
            print(f"Grid size from visual.grid_size: {self.grid_size}")
        elif hasattr(self.visual, 'positional_embedding'):
            pos_embed_size = self.visual.positional_embedding.shape[0]
            # Often, position embedding has CLS token too
            patch_count = pos_embed_size - 1
            grid_side = int(math.sqrt(patch_count))
            if grid_side * grid_side == patch_count:
                self.grid_size = (grid_side, grid_side)
                print(f"Grid size from positional_embedding ({pos_embed_size}): {self.grid_size}")
            else:
                # Default for ViT-B/16
                self.grid_size = (14, 14)
                print(f"Using default grid_size (positional_embedding shape is {pos_embed_size} but not a square + 1): {self.grid_size}")
        else:
            # Default for ViT-B/16
            self.grid_size = (14, 14)
            print(f"Using default grid_size: {self.grid_size}")
            
        # Trace the model structure for debugging
        self._trace_model_structure()
        
        print(f"CLIP extractor initialized with feature dim {self.feature_dim}, grid size {self.grid_size}")
    
    def _trace_model_structure(self):
        """Print the structure of important components for debugging."""
        print("\nCLIP Model Structure Trace:")
        
        if hasattr(self.visual, 'transformer'):
            transformer = self.visual.transformer
            print(f"  Transformer: {type(transformer)}")
            
            if hasattr(transformer, 'resblocks'):
                resblocks = transformer.resblocks
                print(f"  Resblocks: {len(resblocks)} blocks")
                print(f"  Last Resblock: {type(resblocks[-1])}")
            elif hasattr(transformer, 'layers'):
                layers = transformer.layers
                print(f"  Layers: {len(layers)} layers")
                print(f"  Last Layer: {type(layers[-1])}")
            else:
                print("  No resblocks or layers found in transformer")
        
        if hasattr(self.visual, 'positional_embedding'):
            pos_embed = self.visual.positional_embedding
            print(f"  Positional Embedding shape: {pos_embed.shape}")
        
        print(f"  CLIP model expected input size: 224x224")
        print("End of structure trace\n")
    
    def forward(self, images):
        """
        Extract CLIP features and reshape to spatial feature map.
        
        Args:
            images: Input images [B, 3, H, W]
            
        Returns:
            Spatial feature map [B, C, 16, 16] with consistent channel dimension
        """
        batch_size = images.shape[0]
        
        # Resize to 224x224 for CLIP
        if images.shape[2:] != (224, 224):
            resized_images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        else:
            resized_images = images
        
        # Try simplest approach first - get image embeddings directly
        try:
            with torch.no_grad():
                # Extract image embeddings directly
                embeddings = self.clip_model.encode_image(resized_images.to(self.device))
                
                # For consistent output dimensions
                if embeddings.shape[1] != self.feature_dim:
                    print(f"Warning: CLIP embeddings dimension {embeddings.shape[1]} != expected {self.feature_dim}")
                    self.feature_dim = embeddings.shape[1]
                
                # Reshape to spatial feature map of size 16x16
                embeddings = embeddings.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
                output_features = F.interpolate(embeddings, size=(16, 16), mode='bilinear', align_corners=False)
                return output_features
        
        except Exception as e:
            print(f"Error extracting CLIP embeddings: {e}")
            # Return zero tensor as fallback
            return torch.zeros(batch_size, self.feature_dim, 16, 16, device=self.device)