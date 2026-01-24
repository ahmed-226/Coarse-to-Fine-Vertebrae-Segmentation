"""
Base Predictor class for inference.
"""
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk


class BasePredictor(ABC):
    """
    Base class for inference predictors.
    
    Subclasses must implement:
        - create_model()
        - predict()
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        config: Any = None
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            config: Stage-specific configuration
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
    
    def _load_model(self) -> nn.Module:
        """Load model from checkpoint."""
        print(f"Loading model from {self.checkpoint_path}")
        
        model = self.create_model()
        model = model.to(self.device)
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded model from epoch {checkpoint.get('best_epoch', 'unknown')}")
        
        return model
    
    @abstractmethod
    def create_model(self) -> nn.Module:
        """Create the model architecture."""
        pass
    
    @abstractmethod
    def predict(self, image: sitk.Image) -> Dict[str, Any]:
        """
        Run inference on a single image.
        
        Args:
            image: Input SimpleITK image
        
        Returns:
            Dictionary with prediction results
        """
        pass
    
    def preprocess(
        self,
        image: sitk.Image,
        target_spacing: Tuple[float, ...],
        target_size: Tuple[int, ...]
    ) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess image for inference.
        
        Args:
            image: Input SimpleITK image
            target_spacing: Target voxel spacing
            target_size: Target volume size
        
        Returns:
            Tuple of (preprocessed_array, metadata_dict)
        """
        # Get original metadata
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        original_origin = image.GetOrigin()
        original_direction = image.GetDirection()
        
        # Resample to target spacing
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        
        # Compute new size
        new_size = [
            int(np.round(old_size * old_spacing / new_spacing))
            for old_size, old_spacing, new_spacing 
            in zip(original_size, original_spacing, target_spacing)
        ]
        resampler.SetSize(new_size)
        resampler.SetOutputOrigin(original_origin)
        resampler.SetOutputDirection(original_direction)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1024)  # CT background
        
        resampled = resampler.Execute(image)
        
        # Convert to numpy
        array = sitk.GetArrayFromImage(resampled)  # [D, H, W]
        
        # Normalize intensity
        array = self._normalize_intensity(array)
        
        # Store metadata for postprocessing
        metadata = {
            'original_spacing': original_spacing,
            'original_size': original_size,
            'original_origin': original_origin,
            'original_direction': original_direction,
            'resampled_spacing': target_spacing,
            'resampled_size': resampled.GetSize()
        }
        
        return array, metadata
    
    def _normalize_intensity(
        self,
        image: np.ndarray,
        window_center: float = 300.0,
        window_width: float = 1500.0
    ) -> np.ndarray:
        """
        Normalize CT intensity to [0, 1] range using windowing.
        
        Args:
            image: Input CT image in HU
            window_center: Window center in HU
            window_width: Window width in HU
        
        Returns:
            Normalized image in [0, 1]
        """
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        
        normalized = (image - min_val) / (max_val - min_val)
        normalized = np.clip(normalized, 0, 1)
        
        return normalized.astype(np.float32)
    
    def to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to batch tensor."""
        # Add batch and channel dimensions
        tensor = torch.from_numpy(array).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        return tensor.to(self.device)
    
    def find_heatmap_peak(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Find the peak location in a heatmap.
        
        Args:
            heatmap: 3D heatmap array
        
        Returns:
            Peak coordinates [z, y, x]
        """
        max_val = heatmap.max()
        if max_val < 1e-8:
            return np.array([heatmap.shape[0]/2, heatmap.shape[1]/2, heatmap.shape[2]/2])
        
        max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        # Weighted centroid for sub-voxel accuracy
        window_size = 3
        z_start = max(0, max_idx[0] - window_size)
        z_end = min(heatmap.shape[0], max_idx[0] + window_size + 1)
        y_start = max(0, max_idx[1] - window_size)
        y_end = min(heatmap.shape[1], max_idx[1] + window_size + 1)
        x_start = max(0, max_idx[2] - window_size)
        x_end = min(heatmap.shape[2], max_idx[2] + window_size + 1)
        
        local_window = heatmap[z_start:z_end, y_start:y_end, x_start:x_end]
        threshold = local_window.max() * 0.5
        local_window = np.maximum(local_window - threshold, 0)
        
        if local_window.sum() > 1e-8:
            z_coords, y_coords, x_coords = np.meshgrid(
                np.arange(z_start, z_end),
                np.arange(y_start, y_end),
                np.arange(x_start, x_end),
                indexing='ij'
            )
            
            total_weight = local_window.sum()
            z_center = (z_coords * local_window).sum() / total_weight
            y_center = (y_coords * local_window).sum() / total_weight
            x_center = (x_coords * local_window).sum() / total_weight
            
            return np.array([z_center, y_center, x_center])
        
        return np.array(max_idx, dtype=np.float64)
