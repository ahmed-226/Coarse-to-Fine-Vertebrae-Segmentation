"""
Vertebrae Segmentation Predictor (Stage 3).
Segments individual vertebrae using their predicted centroids.
"""
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from scipy import ndimage

from ..config.vertebrae_segmentation import VertebraeSegmentationConfig
from ..models.unet3d import UNet3DVertebraeSegmentation
from .predictor import BasePredictor


class VertebraeSegmentationPredictor(BasePredictor):
    """
    Predictor for Stage 3: Vertebrae Segmentation.
    
    Segments each vertebra by cropping around its centroid
    and running a U-Net for binary segmentation.
    """
    
    # Vertebra label mapping (VerSe challenge format)
    VERTEBRA_LABELS = {
        'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'C7': 7,
        'T1': 8, 'T2': 9, 'T3': 10, 'T4': 11, 'T5': 12, 'T6': 13,
        'T7': 14, 'T8': 15, 'T9': 16, 'T10': 17, 'T11': 18, 'T12': 19,
        'L1': 20, 'L2': 21, 'L3': 22, 'L4': 23, 'L5': 24, 'L6': 25,
        'S1': 26, 'S2': 27
    }
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        config: Optional[VertebraeSegmentationConfig] = None
    ):
        if config is None:
            config = VertebraeSegmentationConfig()
        
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            config=config
        )
    
    def create_model(self) -> nn.Module:
        """Create U-Net for vertebrae segmentation."""
        return UNet3DVertebraeSegmentation(
            in_channels=1,
            num_filters_base=self.config.num_filters_base,
            num_levels=self.config.num_levels
        )
    
    def predict(
        self,
        image: sitk.Image,
        landmarks: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Segment all vertebrae using their centroids.
        
        Args:
            image: Input CT image (SimpleITK)
            landmarks: Dict mapping vertebra name to physical coordinates
        
        Returns:
            Dictionary with:
                - 'segmentation': Combined segmentation mask (SimpleITK)
                - 'per_vertebra': Dict of individual vertebra masks
                - 'metadata': Processing metadata
        """
        # Preprocess full image
        target_spacing = tuple(self.config.image_spacing)
        array, metadata = self.preprocess_full(image, target_spacing)
        
        # Initialize combined segmentation
        combined_mask = np.zeros(array.shape, dtype=np.uint8)
        per_vertebra_masks = {}
        
        # Segment each vertebra
        for name, centroid_physical in landmarks.items():
            if name not in self.VERTEBRA_LABELS:
                continue
            
            label = self.VERTEBRA_LABELS[name]
            
            # Convert centroid to voxel coordinates
            centroid_voxel = self._physical_to_voxel(
                centroid_physical,
                target_spacing,
                metadata['origin']
            )
            
            # Segment this vertebra
            mask = self._segment_single_vertebra(array, centroid_voxel)
            
            # Store individual mask
            per_vertebra_masks[name] = mask
            
            # Add to combined mask (handling overlaps by taking max label)
            combined_mask = np.where(mask > 0, label, combined_mask)
        
        # Convert to SimpleITK
        segmentation = sitk.GetImageFromArray(combined_mask)
        segmentation.SetSpacing(target_spacing)
        segmentation.SetOrigin(metadata['origin'])
        segmentation.SetDirection(metadata['direction'])
        
        return {
            'segmentation': segmentation,
            'per_vertebra': per_vertebra_masks,
            'metadata': metadata
        }
    
    def predict_single(
        self,
        image: sitk.Image,
        centroid: np.ndarray,
        vertebra_name: str
    ) -> Dict[str, Any]:
        """
        Segment a single vertebra.
        
        Args:
            image: Input CT image (SimpleITK)
            centroid: Physical coordinates of vertebra centroid
            vertebra_name: Name of the vertebra (e.g., 'L1')
        
        Returns:
            Dictionary with:
                - 'mask': Binary segmentation mask [D, H, W]
                - 'mask_sitk': SimpleITK mask
                - 'crop_info': Cropping metadata
        """
        target_spacing = tuple(self.config.image_spacing)
        array, metadata = self.preprocess_full(image, target_spacing)
        
        # Convert centroid to voxel
        centroid_voxel = self._physical_to_voxel(
            centroid, target_spacing, metadata['origin']
        )
        
        # Segment
        mask = self._segment_single_vertebra(array, centroid_voxel)
        
        # Convert to SimpleITK
        mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
        mask_sitk.SetSpacing(target_spacing)
        mask_sitk.SetOrigin(metadata['origin'])
        mask_sitk.SetDirection(metadata['direction'])
        
        return {
            'mask': mask,
            'mask_sitk': mask_sitk,
            'vertebra_name': vertebra_name,
            'metadata': metadata
        }
    
    def _segment_single_vertebra(
        self,
        array: np.ndarray,
        centroid_voxel: np.ndarray
    ) -> np.ndarray:
        """
        Segment a single vertebra by cropping and running U-Net.
        
        Args:
            array: Full preprocessed volume
            centroid_voxel: Voxel coordinates of vertebra centroid
        
        Returns:
            Binary mask in full volume coordinates
        """
        crop_size = self.config.image_size
        
        # Compute crop region
        start, end, pad_before, pad_after = self._compute_crop_region(
            centroid_voxel, crop_size, array.shape
        )
        
        # Extract crop
        crop = array[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        
        # Pad if needed
        if any(pad_before) or any(pad_after):
            crop = np.pad(
                crop,
                [(pad_before[i], pad_after[i]) for i in range(3)],
                mode='constant',
                constant_values=0
            )
        
        # Ensure correct size
        if crop.shape != tuple(crop_size):
            crop_correct = np.zeros(crop_size, dtype=crop.dtype)
            copy_size = [min(c, t) for c, t in zip(crop.shape, crop_size)]
            crop_correct[:copy_size[0], :copy_size[1], :copy_size[2]] = \
                crop[:copy_size[0], :copy_size[1], :copy_size[2]]
            crop = crop_correct
        
        # Run inference
        tensor = self.to_tensor(crop)
        with torch.no_grad():
            pred = self.model(tensor)
            pred = torch.sigmoid(pred).cpu().numpy()[0, 0]  # [D, H, W]
        
        # Threshold
        pred_binary = (pred > 0.5).astype(np.uint8)
        
        # Post-process: keep largest connected component
        pred_binary = self._keep_largest_component(pred_binary)
        
        # Map back to full volume
        full_mask = np.zeros(array.shape, dtype=np.uint8)
        
        # Remove padding from prediction
        pred_cropped = pred_binary[
            pad_before[0]:pred_binary.shape[0] - pad_after[0] if pad_after[0] > 0 else pred_binary.shape[0],
            pad_before[1]:pred_binary.shape[1] - pad_after[1] if pad_after[1] > 0 else pred_binary.shape[1],
            pad_before[2]:pred_binary.shape[2] - pad_after[2] if pad_after[2] > 0 else pred_binary.shape[2]
        ]
        
        # Place in full volume
        actual_size = [min(end[i] - start[i], pred_cropped.shape[i]) for i in range(3)]
        full_mask[start[0]:start[0] + actual_size[0],
                  start[1]:start[1] + actual_size[1],
                  start[2]:start[2] + actual_size[2]] = \
            pred_cropped[:actual_size[0], :actual_size[1], :actual_size[2]]
        
        return full_mask
    
    def _compute_crop_region(
        self,
        centroid: np.ndarray,
        crop_size: Tuple[int, ...],
        volume_shape: Tuple[int, ...]
    ) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
        """
        Compute crop region around centroid.
        
        Returns:
            Tuple of (start, end, pad_before, pad_after)
        """
        half_size = np.array(crop_size) // 2
        center = np.round(centroid).astype(int)
        
        start = center - half_size
        end = start + np.array(crop_size)
        
        # Compute padding needed
        pad_before = [max(0, -s) for s in start]
        pad_after = [max(0, e - vs) for e, vs in zip(end, volume_shape)]
        
        # Clamp to volume bounds
        start = np.maximum(start, 0)
        end = np.minimum(end, volume_shape)
        
        return start, end, pad_before, pad_after
    
    def _keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component."""
        if mask.sum() == 0:
            return mask
        
        labeled, num_features = ndimage.label(mask)
        
        if num_features <= 1:
            return mask
        
        # Find largest component
        component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        largest_label = np.argmax(component_sizes) + 1
        
        return (labeled == largest_label).astype(np.uint8)
    
    def _physical_to_voxel(
        self,
        physical_coords: np.ndarray,
        spacing: Tuple[float, ...],
        origin: Tuple[float, ...]
    ) -> np.ndarray:
        """Convert physical to voxel coordinates."""
        # Physical is (x, y, z), voxel is (z, y, x)
        voxel = np.array([
            (physical_coords[2] - origin[2]) / spacing[2],  # z
            (physical_coords[1] - origin[1]) / spacing[1],  # y
            (physical_coords[0] - origin[0]) / spacing[0],  # x
        ])
        return voxel
    
    def preprocess_full(
        self,
        image: sitk.Image,
        target_spacing: Tuple[float, ...]
    ) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess full image without size cropping.
        
        Args:
            image: Input CT image
            target_spacing: Target voxel spacing
        
        Returns:
            Tuple of (preprocessed_array, metadata)
        """
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        
        # Resample to target spacing
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        
        new_size = [
            int(np.round(old_size * old_spacing / new_spacing))
            for old_size, old_spacing, new_spacing 
            in zip(original_size, original_spacing, target_spacing)
        ]
        resampler.SetSize(new_size)
        resampler.SetOutputOrigin(origin)
        resampler.SetOutputDirection(direction)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1024)
        
        resampled = resampler.Execute(image)
        
        # Convert to numpy and normalize
        array = sitk.GetArrayFromImage(resampled)
        array = self._normalize_intensity(array)
        
        metadata = {
            'original_spacing': original_spacing,
            'original_size': original_size,
            'origin': origin,
            'direction': direction,
            'resampled_spacing': target_spacing,
            'resampled_size': resampled.GetSize()
        }
        
        return array, metadata
