"""
UNet implementation for pet segmentation
This module implements a UNet model that follows the architecture and hyperparameters
defined by nnU-Net for the pet segmentation task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Union, Optional, Type


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
    UNet model for semantic segmentation with reduced complexity and spatial dropout.
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
        decoder_dropout_rates: List[float] = None
    ):
        """
        Initialize the UNet model.
        
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
    
    def forward(self, x):
        """
        Forward pass of the UNet model.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
                
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