"""
Custom layers for 3D medical image segmentation networks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ConvBlock3D(nn.Module):
    """
    3D Convolutional block with BatchNorm and activation.
    
    Structure: [Conv3D -> BatchNorm -> Activation] x repeats
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        repeats: int = 2,
        activation: str = 'relu',
        dropout_ratio: float = 0.0,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        layers = []
        current_in = in_channels
        
        for i in range(repeats):
            # Convolution
            layers.append(
                nn.Conv3d(current_in, out_channels, kernel_size, padding=padding)
            )
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm3d(out_channels))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1, inplace=True))
            elif activation == 'selu':
                layers.append(nn.SELU(inplace=True))
            
            # Dropout (only after first conv in block)
            if dropout_ratio > 0 and i == 0:
                layers.append(nn.Dropout3d(dropout_ratio))
            
            current_in = out_channels
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownsampleBlock(nn.Module):
    """
    Downsampling block using average pooling.
    Factor of 2 in all dimensions.
    """
    
    def __init__(self, method: str = 'avg_pool'):
        super().__init__()
        
        if method == 'avg_pool':
            self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
        elif method == 'max_pool':
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        elif method == 'stride':
            # Use strided conv (defined in forward)
            self.pool = None
        else:
            raise ValueError(f"Unknown pooling method: {method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool is not None:
            return self.pool(x)
        else:
            # Strided convolution fallback
            return F.avg_pool3d(x, kernel_size=2, stride=2)


class UpsampleBlock(nn.Module):
    """
    Upsampling block using trilinear interpolation.
    Factor of 2 in all dimensions.
    """
    
    def __init__(
        self,
        in_channels: int = None,
        out_channels: int = None,
        method: str = 'trilinear',
        scale_factor: int = 2
    ):
        super().__init__()
        self.method = method
        self.scale_factor = scale_factor
        
        # Optional 1x1 conv to reduce channels after upsampling
        if in_channels is not None and out_channels is not None:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv = None
    
    def forward(self, x: torch.Tensor, target_size: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        if target_size is not None:
            # Upsample to exact target size
            x = F.interpolate(
                x, size=target_size, mode=self.method, align_corners=False
            )
        else:
            # Upsample by scale factor
            x = F.interpolate(
                x, scale_factor=self.scale_factor, mode=self.method, align_corners=False
            )
        
        if self.conv is not None:
            x = self.conv(x)
        
        return x


class SkipConnection(nn.Module):
    """
    Skip connection processing (parallel path in U-Net).
    Optionally applies convolution to skip features.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        use_conv: bool = False
    ):
        super().__init__()
        
        if use_conv and out_channels is not None:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is not None:
            return self.conv(x)
        return x


class CombineBlock(nn.Module):
    """
    Combine skip connection with upsampled features.
    Options: concatenation or addition.
    """
    
    def __init__(self, method: str = 'concat'):
        super().__init__()
        self.method = method
    
    def forward(
        self,
        skip_features: torch.Tensor,
        upsampled_features: torch.Tensor
    ) -> torch.Tensor:
        
        # Handle size mismatch by cropping or padding
        if skip_features.shape[2:] != upsampled_features.shape[2:]:
            # Interpolate upsampled to match skip size
            upsampled_features = F.interpolate(
                upsampled_features,
                size=skip_features.shape[2:],
                mode='trilinear',
                align_corners=False
            )
        
        if self.method == 'concat':
            return torch.cat([skip_features, upsampled_features], dim=1)
        elif self.method == 'add':
            return skip_features + upsampled_features
        else:
            raise ValueError(f"Unknown combine method: {self.method}")


class OutputHead(nn.Module):
    """
    Final output layer with optional activation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Optional[str] = None
    ):
        super().__init__()
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SpatialCoordinateGenerator(nn.Module):
    """
    Generate normalized spatial coordinate channels.
    Used in SpatialConfiguration-Net.
    
    Output: 3 channels with normalized X, Y, Z coordinates [-1, 1]
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate spatial coordinates matching input tensor size.
        
        Args:
            x: Input tensor [B, C, D, H, W]
        
        Returns:
            Coordinate tensor [B, 3, D, H, W] with values in [-1, 1]
        """
        batch_size = x.shape[0]
        d, h, w = x.shape[2:]
        device = x.device
        
        # Create normalized coordinate grids
        z_coords = torch.linspace(-1, 1, d, device=device)
        y_coords = torch.linspace(-1, 1, h, device=device)
        x_coords = torch.linspace(-1, 1, w, device=device)
        
        # Create meshgrid
        zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        
        # Stack and expand for batch
        coords = torch.stack([xx, yy, zz], dim=0)  # [3, D, H, W]
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        
        return coords


class GaussianHeatmapLayer(nn.Module):
    """
    Generate Gaussian heatmap from predicted mean and sigma.
    Used for learnable sigma in vertebrae localization.
    """
    
    def __init__(self, sigma_scale: float = 1000.0):
        super().__init__()
        self.sigma_scale = sigma_scale
    
    def forward(
        self,
        coords: torch.Tensor,
        mean: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate Gaussian heatmap.
        
        Args:
            coords: Spatial coordinates [B, 3, D, H, W]
            mean: Predicted mean location [B, N, 3] (N = num_landmarks)
            sigma: Predicted sigma [B, N, 3]
        
        Returns:
            Heatmaps [B, N, D, H, W]
        """
        batch_size = coords.shape[0]
        num_landmarks = mean.shape[1]
        d, h, w = coords.shape[2:]
        
        # Expand dimensions for broadcasting
        # coords: [B, 1, 3, D, H, W]
        coords = coords.unsqueeze(1)
        
        # mean: [B, N, 3, 1, 1, 1]
        mean = mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # sigma: [B, N, 3, 1, 1, 1]
        sigma = sigma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        sigma = sigma * self.sigma_scale + 1e-6  # Prevent division by zero
        
        # Compute squared distance
        diff = coords - mean  # [B, N, 3, D, H, W]
        sq_dist = (diff ** 2) / (2 * sigma ** 2)
        sq_dist = sq_dist.sum(dim=2)  # Sum over xyz -> [B, N, D, H, W]
        
        # Gaussian
        heatmap = torch.exp(-sq_dist)
        
        return heatmap
