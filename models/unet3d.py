"""
3D U-Net Implementation for Vertebrae Segmentation.

Architecture: UnetClassicAvgLinear3d
- Downsampling: Average pooling (factor 2)
- Upsampling: Trilinear interpolation (factor 2)
- Skip connections: Channel concatenation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .layers import ConvBlock3D, DownsampleBlock, UpsampleBlock, CombineBlock, OutputHead


class UNet3D(nn.Module):
    """
    3D U-Net for medical image segmentation.
    
    Based on the original U-Net architecture adapted for 3D volumes.
    Uses average pooling for downsampling and trilinear interpolation for upsampling.
    
    Args:
        in_channels: Number of input channels (1 for CT, 2 if including heatmap prior)
        out_channels: Number of output channels (1 for heatmap, 2 for binary seg)
        num_filters_base: Base number of filters (doubled at each level)
        num_levels: Number of encoder/decoder levels
        kernel_size: Convolution kernel size
        repeats: Number of conv blocks per level
        dropout_ratio: Dropout probability
        activation: Activation function ('relu', 'leaky_relu', 'selu')
        use_batch_norm: Whether to use batch normalization
        output_activation: Final activation ('sigmoid', 'softmax', None)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_filters_base: int = 64,
        num_levels: int = 5,
        kernel_size: int = 3,
        repeats: int = 2,
        dropout_ratio: float = 0.5,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        output_activation: Optional[str] = None
    ):
        super().__init__()
        
        self.num_levels = num_levels
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Calculate filter sizes for each level
        # Level 0: 64, Level 1: 128, Level 2: 128, Level 3: 128, Level 4: 128
        # (matches original TensorFlow implementation)
        self.filters = []
        for level in range(num_levels):
            if level == 0:
                self.filters.append(num_filters_base)
            elif level == 1:
                self.filters.append(num_filters_base * 2)
            else:
                self.filters.append(num_filters_base * 2)  # Capped at 2x base
        
        # =====================================================================
        # ENCODER (Contracting Path)
        # =====================================================================
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        current_channels = in_channels
        for level in range(num_levels):
            # Convolution block
            self.encoders.append(
                ConvBlock3D(
                    in_channels=current_channels,
                    out_channels=self.filters[level],
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    repeats=repeats,
                    activation=activation,
                    dropout_ratio=dropout_ratio if level > 0 else 0.0,
                    use_batch_norm=use_batch_norm
                )
            )
            
            # Downsampling (except for last level)
            if level < num_levels - 1:
                self.downsamplers.append(DownsampleBlock(method='avg_pool'))
            
            current_channels = self.filters[level]
        
        # =====================================================================
        # BOTTLENECK
        # =====================================================================
        self.bottleneck = ConvBlock3D(
            in_channels=self.filters[-1],
            out_channels=self.filters[-1],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            repeats=repeats,
            activation=activation,
            dropout_ratio=dropout_ratio,
            use_batch_norm=use_batch_norm
        )
        
        # =====================================================================
        # DECODER (Expanding Path)
        # =====================================================================
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.combiners = nn.ModuleList()
        
        for level in range(num_levels - 2, -1, -1):  # Reverse order
            # Upsampling
            self.upsamplers.append(UpsampleBlock(method='trilinear'))
            
            # Combine skip + upsampled (concatenation doubles channels)
            self.combiners.append(CombineBlock(method='concat'))
            
            # Convolution block (input is concatenated channels)
            # Skip has filters[level] channels, upsampled has filters[level+1] channels
            combined_channels = self.filters[level] + self.filters[level + 1]
            
            self.decoders.append(
                ConvBlock3D(
                    in_channels=combined_channels,  # Skip + upsampled concatenated
                    out_channels=self.filters[level],
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    repeats=repeats,
                    activation=activation,
                    dropout_ratio=0.0,  # No dropout in decoder
                    use_batch_norm=use_batch_norm
                )
            )
        
        # =====================================================================
        # OUTPUT HEAD
        # =====================================================================
        self.output_head = OutputHead(
            in_channels=self.filters[0],
            out_channels=out_channels,
            activation=output_activation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor [B, C, D, H, W]
        
        Returns:
            Output tensor [B, out_channels, D, H, W]
        """
        # Store skip connections
        skip_features = []
        
        # =====================================================================
        # ENCODER
        # =====================================================================
        for level in range(self.num_levels - 1):
            x = self.encoders[level](x)
            skip_features.append(x)
            x = self.downsamplers[level](x)
        
        # Last encoder level (no downsampling after)
        x = self.encoders[-1](x)
        
        # =====================================================================
        # BOTTLENECK
        # =====================================================================
        x = self.bottleneck(x)
        
        # =====================================================================
        # DECODER
        # =====================================================================
        for level in range(self.num_levels - 1):
            # Upsample
            skip_idx = self.num_levels - 2 - level
            target_size = skip_features[skip_idx].shape[2:]
            x = self.upsamplers[level](x, target_size=target_size)
            
            # Combine with skip connection
            x = self.combiners[level](skip_features[skip_idx], x)
            
            # Decode
            x = self.decoders[level](x)
        
        # =====================================================================
        # OUTPUT
        # =====================================================================
        x = self.output_head(x)
        
        return x


class UNet3DSpineLocalization(UNet3D):
    """
    U-Net configured for spine localization (Stage 1).
    
    Input: CT volume [B, 1, 128, 64, 64]
    Output: Spine heatmap [B, 1, 128, 64, 64]
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            in_channels=1,
            out_channels=1,
            num_filters_base=kwargs.get('num_filters_base', 64),
            num_levels=kwargs.get('num_levels', 5),
            dropout_ratio=kwargs.get('dropout_ratio', 0.5),
            output_activation=None  # L2 loss, no activation
        )


class UNet3DVertebraeSegmentation(UNet3D):
    """
    U-Net configured for vertebrae segmentation (Stage 3).
    
    Input: CT volume + heatmap prior [B, 2, 96, 128, 128]
    Output: Binary segmentation [B, 1, 96, 128, 128]
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            in_channels=2,  # Image + heatmap prior
            out_channels=1,
            num_filters_base=kwargs.get('num_filters_base', 64),
            num_levels=kwargs.get('num_levels', 5),
            dropout_ratio=kwargs.get('dropout_ratio', 0.5),
            output_activation=None  # BCEWithLogitsLoss includes sigmoid
        )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test U-Net
    print("Testing UNet3D...")
    
    # Spine localization config
    model = UNet3DSpineLocalization(num_filters_base=64, num_levels=5)
    print(f"Spine Localization Model: {count_parameters(model):,} parameters")
    
    # Test forward pass
    x = torch.randn(1, 1, 128, 64, 64)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
    
    # Vertebrae segmentation config
    model = UNet3DVertebraeSegmentation(num_filters_base=64, num_levels=5)
    print(f"Vertebrae Segmentation Model: {count_parameters(model):,} parameters")
    
    # Test forward pass
    x = torch.randn(1, 2, 96, 128, 128)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
    
    print("All tests passed!")
