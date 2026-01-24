"""
SpatialConfiguration-Net (SCNet) for Vertebrae Localization.

Architecture combines:
1. Local appearance network (U-Net) - learns local features
2. Spatial configuration network - learns spatial relationships between vertebrae

Key innovation: Separates local appearance from global spatial configuration,
allowing the network to learn anatomical constraints (vertebrae order, spacing).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

from .layers import (
    ConvBlock3D,
    DownsampleBlock,
    UpsampleBlock,
    SpatialCoordinateGenerator,
    GaussianHeatmapLayer
)
from .unet3d import UNet3D


class LocalAppearanceNetwork(nn.Module):
    """
    Local appearance U-Net for SCNet.
    
    Produces intermediate heatmaps based on local image features.
    These are refined by the spatial configuration network.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 26,
        num_filters_base: int = 64,
        num_levels: int = 5,
        dropout_ratio: float = 0.5
    ):
        super().__init__()
        
        self.unet = UNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            num_filters_base=num_filters_base,
            num_levels=num_levels,
            dropout_ratio=dropout_ratio,
            output_activation=None
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input CT volume [B, 1, D, H, W]
        
        Returns:
            Local heatmaps [B, num_landmarks, D, H, W]
        """
        return self.unet(x)


class SpatialConfigurationNetwork(nn.Module):
    """
    Spatial configuration network for refining local heatmaps.
    
    Takes local heatmaps and spatial coordinates as input,
    learns spatial relationships between landmarks.
    """
    
    def __init__(
        self,
        num_landmarks: int = 26,
        num_filters_base: int = 64,
        spatial_downsample: int = 4
    ):
        super().__init__()
        
        self.num_landmarks = num_landmarks
        self.spatial_downsample = spatial_downsample
        
        # Spatial coordinate generator
        self.coord_generator = SpatialCoordinateGenerator()
        
        # Input: local heatmaps + spatial coordinates
        # num_landmarks channels + 3 coordinate channels
        in_channels = num_landmarks + 3
        
        # Downsampling path
        self.downsample_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        
        current_channels = in_channels
        num_downsample = int(math.log2(spatial_downsample))
        
        for i in range(num_downsample):
            out_channels = num_filters_base if i == 0 else num_filters_base * 2
            
            self.conv_blocks.append(
                ConvBlock3D(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    repeats=2,
                    activation='relu'
                )
            )
            self.downsample_blocks.append(DownsampleBlock(method='avg_pool'))
            current_channels = out_channels
        
        # Middle processing
        self.middle_conv = ConvBlock3D(
            in_channels=current_channels,
            out_channels=num_filters_base * 2,
            kernel_size=3,
            repeats=2,
            activation='relu'
        )
        
        # Upsampling path
        self.upsample_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        current_channels = num_filters_base * 2
        for i in range(num_downsample):
            self.upsample_blocks.append(UpsampleBlock(method='trilinear'))
            self.decoder_blocks.append(
                ConvBlock3D(
                    in_channels=current_channels,
                    out_channels=num_filters_base,
                    kernel_size=3,
                    repeats=2,
                    activation='relu'
                )
            )
            current_channels = num_filters_base
        
        # Output head
        self.output_conv = nn.Conv3d(num_filters_base, num_landmarks, kernel_size=1)
    
    def forward(
        self,
        local_heatmaps: torch.Tensor,
        image_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Args:
            local_heatmaps: Local appearance heatmaps [B, num_landmarks, D, H, W]
            image_size: Original image size for coordinate generation
        
        Returns:
            Spatial offsets [B, num_landmarks, D, H, W]
        """
        # Generate spatial coordinates
        coords = self.coord_generator(local_heatmaps)  # [B, 3, D, H, W]
        
        # Concatenate local heatmaps with coordinates
        x = torch.cat([local_heatmaps, coords], dim=1)  # [B, num_landmarks + 3, D, H, W]
        
        # Encode
        skip_features = []
        for conv, down in zip(self.conv_blocks, self.downsample_blocks):
            x = conv(x)
            skip_features.append(x)
            x = down(x)
        
        # Middle
        x = self.middle_conv(x)
        
        # Decode
        for up, dec in zip(self.upsample_blocks, self.decoder_blocks):
            x = up(x, target_size=skip_features[-1].shape[2:])
            skip_features.pop()
            x = dec(x)
        
        # Output spatial offsets
        offsets = self.output_conv(x)
        
        return offsets


class SpatialConfigurationNet(nn.Module):
    """
    Complete SpatialConfiguration-Net for vertebrae localization.
    
    Combines local appearance network with spatial configuration network.
    
    Output = sigmoid(local_heatmaps + spatial_offsets)
    
    Args:
        num_landmarks: Number of vertebrae classes (26 for full spine)
        num_filters_base: Base number of filters
        num_levels: Number of U-Net levels
        spatial_downsample: Downsampling factor for spatial network
        dropout_ratio: Dropout probability
        learnable_sigma: Whether to use learnable heatmap sigma
        initial_sigma: Initial sigma value if learnable
    """
    
    def __init__(
        self,
        num_landmarks: int = 26,
        num_filters_base: int = 64,
        num_levels: int = 5,
        spatial_downsample: int = 4,
        dropout_ratio: float = 0.5,
        learnable_sigma: bool = True,
        initial_sigma: float = 4.0
    ):
        super().__init__()
        
        self.num_landmarks = num_landmarks
        self.learnable_sigma = learnable_sigma
        
        # Local appearance network (U-Net)
        self.local_appearance = LocalAppearanceNetwork(
            in_channels=1,
            out_channels=num_landmarks,
            num_filters_base=num_filters_base,
            num_levels=num_levels,
            dropout_ratio=dropout_ratio
        )
        
        # Spatial configuration network
        self.spatial_config = SpatialConfigurationNetwork(
            num_landmarks=num_landmarks,
            num_filters_base=num_filters_base,
            spatial_downsample=spatial_downsample
        )
        
        # Learnable sigma parameters
        if learnable_sigma:
            # Initialize sigma for each landmark (3 values for x, y, z)
            self.sigma = nn.Parameter(
                torch.full((num_landmarks, 3), initial_sigma / 1000.0)
            )
        else:
            self.register_buffer(
                'sigma',
                torch.full((num_landmarks, 3), initial_sigma / 1000.0)
            )
    
    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through SCNet.
        
        Args:
            x: Input CT volume [B, 1, D, H, W]
            return_intermediate: Whether to return intermediate outputs
        
        Returns:
            If return_intermediate:
                (final_heatmaps, local_heatmaps, spatial_offsets, sigma)
            Else:
                final_heatmaps
        """
        # Get local appearance heatmaps
        local_heatmaps = self.local_appearance(x)  # [B, N, D, H, W]
        
        # Apply sigmoid to get probabilities
        local_probs = torch.sigmoid(local_heatmaps)
        
        # Get spatial configuration offsets
        spatial_offsets = self.spatial_config(local_probs, x.shape[2:])
        
        # Combine: add offsets to local heatmaps
        combined = local_heatmaps + spatial_offsets
        
        # Final activation
        final_heatmaps = torch.sigmoid(combined)
        
        if return_intermediate:
            return final_heatmaps, local_heatmaps, spatial_offsets, self.sigma
        
        return final_heatmaps
    
    def get_sigma(self) -> torch.Tensor:
        """Get current sigma values (scaled)."""
        return self.sigma * 1000.0


class SCNetWithLearnableSigma(SpatialConfigurationNet):
    """
    SCNet variant with learnable Gaussian sigma for heatmap generation.
    
    The sigma values are learned during training, allowing the network
    to adapt the heatmap spread for each vertebra class.
    """
    
    def __init__(self, **kwargs):
        kwargs['learnable_sigma'] = True
        super().__init__(**kwargs)
        
        self.heatmap_layer = GaussianHeatmapLayer(sigma_scale=1000.0)
        self.coord_generator = SpatialCoordinateGenerator()
    
    def generate_heatmaps(
        self,
        predictions: torch.Tensor,
        image_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Generate Gaussian heatmaps from predictions using learnable sigma.
        
        Args:
            predictions: Network predictions (logits) [B, N, D, H, W]
            image_size: Target image size
        
        Returns:
            Gaussian heatmaps [B, N, D, H, W]
        """
        batch_size = predictions.shape[0]
        
        # Find peak locations in predictions
        # This is used during inference to get landmark positions
        # During training, we use the predicted heatmaps directly
        
        return predictions  # For now, return raw predictions


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test SCNet
    print("Testing SpatialConfigurationNet...")
    
    model = SpatialConfigurationNet(
        num_landmarks=26,
        num_filters_base=64,
        num_levels=5,
        spatial_downsample=4,
        learnable_sigma=True,
        initial_sigma=4.0
    )
    
    print(f"SCNet Model: {count_parameters(model):,} parameters")
    
    # Test forward pass
    x = torch.randn(1, 1, 128, 96, 96)
    with torch.no_grad():
        heatmaps, local, offsets, sigma = model(x, return_intermediate=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Final heatmaps shape: {heatmaps.shape}")
    print(f"Local heatmaps shape: {local.shape}")
    print(f"Spatial offsets shape: {offsets.shape}")
    print(f"Sigma shape: {sigma.shape}")
    print(f"Sigma values (first 5): {model.get_sigma()[:5]}")
    
    print("All tests passed!")
