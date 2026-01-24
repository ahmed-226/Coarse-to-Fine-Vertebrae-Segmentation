"""
Vertebrae Localization Predictor (Stage 2).
Predicts centroids for all vertebrae using SCNet.
"""
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn

from ..config.vertebrae_localization import VertebraeLocalizationConfig
from ..models.scnet import SpatialConfigurationNet
from .predictor import BasePredictor


class VertebraeLocalizationPredictor(BasePredictor):
    """
    Predictor for Stage 2: Vertebrae Localization.
    
    Predicts heatmaps for each vertebra centroid using SCNet.
    Uses sliding window approach for large volumes.
    """
    
    # Vertebra label mapping
    VERTEBRA_NAMES = (
        ['C' + str(i) for i in range(1, 8)] +
        ['T' + str(i) for i in range(1, 13)] +
        ['L' + str(i) for i in range(1, 7)] +
        ['S1', 'S2']
    )
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        config: Optional[VertebraeLocalizationConfig] = None
    ):
        if config is None:
            config = VertebraeLocalizationConfig()
        
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            config=config
        )
    
    def create_model(self) -> nn.Module:
        """Create SCNet for vertebrae localization."""
        return SpatialConfigurationNet(
            in_channels=1,
            num_landmarks=self.config.num_landmarks,
            num_filters_base=self.config.num_filters_base,
            num_levels=self.config.num_levels,
            initial_sigma=self.config.heatmap_sigma,
            learnable_sigma=self.config.learnable_sigma
        )
    
    def predict(
        self,
        image: sitk.Image,
        spine_centroid: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Predict vertebrae centroids from CT image.
        
        Args:
            image: Input CT image (SimpleITK)
            spine_centroid: Optional spine centroid from Stage 1 for cropping
        
        Returns:
            Dictionary with:
                - 'landmarks': Dict mapping vertebra name to physical coordinates
                - 'landmarks_voxel': Dict mapping vertebra name to voxel coordinates
                - 'heatmaps': Predicted heatmaps [N, D, H, W]
                - 'valid_mask': Boolean mask for valid predictions
                - 'metadata': Preprocessing metadata
        """
        # Preprocess
        target_spacing = tuple(self.config.image_spacing)
        target_size = tuple(self.config.image_size)
        
        array, metadata = self.preprocess(image, target_spacing, target_size)
        
        # Crop around spine centroid if provided
        if spine_centroid is not None:
            array, crop_offset = self._crop_around_centroid(
                array, spine_centroid, target_spacing, target_size, metadata
            )
            metadata['crop_offset'] = crop_offset
        else:
            array, crop_offset = self._center_crop(array, target_size)
            metadata['crop_offset'] = crop_offset
        
        # Use sliding window for large volumes
        if self.config.use_sliding_window:
            heatmaps = self._sliding_window_inference(array)
        else:
            tensor = self.to_tensor(array)
            with torch.no_grad():
                pred, _ = self.model(tensor)
                heatmaps = pred.cpu().numpy()[0]  # [N, D, H, W]
        
        # Find peaks in each heatmap
        landmarks_voxel = {}
        landmarks_physical = {}
        valid_mask = np.zeros(len(self.VERTEBRA_NAMES), dtype=bool)
        
        for i, name in enumerate(self.VERTEBRA_NAMES):
            peak = self.find_heatmap_peak(heatmaps[i])
            max_val = heatmaps[i].max()
            
            # Mark as valid if heatmap has significant activation
            if max_val > self.config.detection_threshold:
                valid_mask[i] = True
                
                # Adjust for crop offset
                peak_global = peak + np.array(crop_offset)
                
                landmarks_voxel[name] = peak_global
                landmarks_physical[name] = self._voxel_to_physical(
                    peak_global, target_spacing, metadata['original_origin']
                )
        
        return {
            'landmarks': landmarks_physical,
            'landmarks_voxel': landmarks_voxel,
            'heatmaps': heatmaps,
            'valid_mask': valid_mask,
            'metadata': metadata
        }
    
    def _crop_around_centroid(
        self,
        array: np.ndarray,
        centroid: np.ndarray,
        spacing: Tuple[float, ...],
        target_size: Tuple[int, ...],
        metadata: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crop volume around spine centroid.
        
        Args:
            array: Input volume
            centroid: Physical centroid from Stage 1
            spacing: Target spacing
            target_size: Target crop size
            metadata: Preprocessing metadata
        
        Returns:
            Tuple of (cropped_array, crop_offset)
        """
        # Convert centroid to voxel coordinates
        origin = metadata['original_origin']
        centroid_voxel = np.array([
            (centroid[2] - origin[2]) / spacing[2],  # z
            (centroid[1] - origin[1]) / spacing[1],  # y
            (centroid[0] - origin[0]) / spacing[0],  # x
        ])
        
        # Compute crop region
        start = np.maximum(0, centroid_voxel - np.array(target_size) // 2).astype(int)
        end = start + np.array(target_size)
        
        # Adjust if crop extends beyond volume
        for i in range(3):
            if end[i] > array.shape[i]:
                end[i] = array.shape[i]
                start[i] = max(0, end[i] - target_size[i])
        
        # Extract crop
        cropped = array[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        
        # Pad if needed
        if cropped.shape != target_size:
            padded = np.zeros(target_size, dtype=array.dtype)
            padded[:cropped.shape[0], :cropped.shape[1], :cropped.shape[2]] = cropped
            cropped = padded
        
        return cropped, start
    
    def _center_crop(
        self,
        array: np.ndarray,
        target_size: Tuple[int, ...]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Center crop array to target size.
        
        Returns:
            Tuple of (cropped_array, crop_offset)
        """
        start = [(s - t) // 2 for s, t in zip(array.shape, target_size)]
        start = [max(0, s) for s in start]
        
        cropped = array[
            start[0]:start[0] + target_size[0],
            start[1]:start[1] + target_size[1],
            start[2]:start[2] + target_size[2]
        ]
        
        # Pad if source is smaller than target
        if cropped.shape != target_size:
            padded = np.zeros(target_size, dtype=array.dtype)
            pad_start = [(t - c) // 2 for c, t in zip(cropped.shape, target_size)]
            padded[
                pad_start[0]:pad_start[0] + cropped.shape[0],
                pad_start[1]:pad_start[1] + cropped.shape[1],
                pad_start[2]:pad_start[2] + cropped.shape[2]
            ] = cropped
            return padded, np.array(start) - np.array(pad_start)
        
        return cropped, np.array(start)
    
    def _sliding_window_inference(
        self,
        array: np.ndarray,
        overlap: float = 0.5
    ) -> np.ndarray:
        """
        Sliding window inference for large volumes.
        
        Args:
            array: Input volume
            overlap: Overlap fraction between windows
        
        Returns:
            Aggregated heatmaps [N, D, H, W]
        """
        window_size = self.config.image_size
        stride = [int(w * (1 - overlap)) for w in window_size]
        
        # Initialize output
        num_landmarks = self.config.num_landmarks
        output = np.zeros((num_landmarks,) + array.shape, dtype=np.float32)
        count = np.zeros(array.shape, dtype=np.float32)
        
        # Sliding window
        for z in range(0, array.shape[0], stride[0]):
            for y in range(0, array.shape[1], stride[1]):
                for x in range(0, array.shape[2], stride[2]):
                    # Extract window
                    z_end = min(z + window_size[0], array.shape[0])
                    y_end = min(y + window_size[1], array.shape[1])
                    x_end = min(x + window_size[2], array.shape[2])
                    
                    window = np.zeros(window_size, dtype=array.dtype)
                    window[:z_end-z, :y_end-y, :x_end-x] = array[z:z_end, y:y_end, x:x_end]
                    
                    # Predict
                    tensor = self.to_tensor(window)
                    with torch.no_grad():
                        pred, _ = self.model(tensor)
                        pred = pred.cpu().numpy()[0]  # [N, D, H, W]
                    
                    # Accumulate
                    for i in range(num_landmarks):
                        output[i, z:z_end, y:y_end, x:x_end] += pred[i, :z_end-z, :y_end-y, :x_end-x]
                    count[z:z_end, y:y_end, x:x_end] += 1
        
        # Average
        count = np.maximum(count, 1)
        for i in range(num_landmarks):
            output[i] /= count
        
        return output
    
    def _voxel_to_physical(
        self,
        voxel_coords: np.ndarray,
        spacing: Tuple[float, ...],
        origin: Tuple[float, ...]
    ) -> np.ndarray:
        """Convert voxel to physical coordinates."""
        physical = np.array([
            origin[0] + voxel_coords[2] * spacing[0],  # x
            origin[1] + voxel_coords[1] * spacing[1],  # y
            origin[2] + voxel_coords[0] * spacing[2],  # z
        ])
        return physical
    
    def get_landmarks_as_json(self, result: Dict) -> List[Dict]:
        """
        Convert landmarks to JSON-compatible format.
        
        Args:
            result: Output from predict()
        
        Returns:
            List of landmark dictionaries
        """
        landmarks_json = []
        
        for name, coords in result['landmarks'].items():
            landmarks_json.append({
                'name': name,
                'x': float(coords[0]),
                'y': float(coords[1]),
                'z': float(coords[2])
            })
        
        return landmarks_json
