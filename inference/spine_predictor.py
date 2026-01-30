"""
Spine Localization Predictor (Stage 1).
Predicts the spine centroid for cropping in Stage 2.
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
import SimpleITK as sitk
import torch.nn as nn

from config.spine_localization import SpineLocalizationConfig
from models.unet3d import UNet3DSpineLocalization
from .predictor import BasePredictor


class SpineLocalizationPredictor(BasePredictor):
    """
    Predictor for Stage 1: Spine Localization.
    
    Predicts a heatmap centered on the spine and extracts
    the centroid for cropping in subsequent stages.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        config: Optional[SpineLocalizationConfig] = None
    ):
        if config is None:
            config = SpineLocalizationConfig()
        
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            config=config
        )
    
    def create_model(self) -> nn.Module:
        """Create U-Net for spine localization."""
        return UNet3DSpineLocalization(
            in_channels=1,
            num_filters_base=self.config.num_filters_base,
            num_levels=self.config.num_levels
        )
    
    def predict(self, image: sitk.Image) -> Dict[str, Any]:
        """
        Predict spine centroid from CT image.
        
        Args:
            image: Input CT image (SimpleITK)
        
        Returns:
            Dictionary with:
                - 'centroid_voxel': Centroid in voxel coordinates of resampled image
                - 'centroid_physical': Centroid in physical (world) coordinates
                - 'heatmap': Predicted heatmap
                - 'metadata': Preprocessing metadata
        """
        import torch
        
        # Preprocess
        target_spacing = tuple(self.config.image_spacing)
        target_size = tuple(self.config.image_size)
        
        array, metadata = self.preprocess(image, target_spacing, target_size)
        
        # Center crop or pad to target size
        array = self._crop_or_pad(array, target_size)
        
        # Convert to tensor
        tensor = self.to_tensor(array)
        
        # Inference
        with torch.no_grad():
            pred = self.model(tensor)
            pred = pred.cpu().numpy()[0, 0]  # [D, H, W]
        
        # Find peak in heatmap
        centroid_voxel = self.find_heatmap_peak(pred)
        
        # Convert to physical coordinates
        centroid_physical = self._voxel_to_physical(
            centroid_voxel,
            target_spacing,
            metadata['original_origin']
        )
        
        return {
            'centroid_voxel': centroid_voxel,
            'centroid_physical': centroid_physical,
            'heatmap': pred,
            'metadata': metadata
        }
    
    def _crop_or_pad(
        self,
        array: np.ndarray,
        target_size: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Center crop or pad array to target size.
        
        Args:
            array: Input 3D array [D, H, W]
            target_size: Target size (D, H, W)
        
        Returns:
            Cropped/padded array
        """
        current_size = array.shape
        result = np.zeros(target_size, dtype=array.dtype)
        
        # Compute crop/pad for each dimension
        for i in range(3):
            if current_size[i] > target_size[i]:
                # Crop
                start = (current_size[i] - target_size[i]) // 2
                slc = slice(start, start + target_size[i])
                if i == 0:
                    array = array[slc, :, :]
                elif i == 1:
                    array = array[:, slc, :]
                else:
                    array = array[:, :, slc]
        
        # Pad if needed
        pad_before = [(max(0, t - c) // 2) for c, t in zip(array.shape, target_size)]
        
        result[
            pad_before[0]:pad_before[0] + array.shape[0],
            pad_before[1]:pad_before[1] + array.shape[1],
            pad_before[2]:pad_before[2] + array.shape[2]
        ] = array
        
        return result
    
    def _voxel_to_physical(
        self,
        voxel_coords: np.ndarray,
        spacing: Tuple[float, ...],
        origin: Tuple[float, ...]
    ) -> np.ndarray:
        """
        Convert voxel coordinates to physical coordinates.
        
        Note: SimpleITK uses (x, y, z) ordering for physical coordinates,
        while numpy uses (z, y, x) for array indexing.
        
        Args:
            voxel_coords: [z, y, x] voxel coordinates
            spacing: (x, y, z) spacing
            origin: (x, y, z) origin
        
        Returns:
            [x, y, z] physical coordinates
        """
        # Convert from numpy (z, y, x) to physical (x, y, z)
        physical = np.array([
            origin[0] + voxel_coords[2] * spacing[0],  # x
            origin[1] + voxel_coords[1] * spacing[1],  # y
            origin[2] + voxel_coords[0] * spacing[2],  # z
        ])
        
        return physical
