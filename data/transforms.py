"""
Data augmentation transforms for vertebrae segmentation.
"""
import numpy as np
import torch
from scipy.ndimage import affine_transform, rotate, zoom
from typing import Tuple, Optional, Dict, Any
import random


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor:
    """Convert numpy arrays to PyTorch tensors."""
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for key in ['image', 'mask', 'heatmap', 'target']:
            if key in sample and isinstance(sample[key], np.ndarray):
                # Add channel dimension if needed
                arr = sample[key]
                if arr.ndim == 3:
                    arr = arr[np.newaxis, ...]  # [1, Z, Y, X]
                sample[key] = torch.from_numpy(arr.copy()).float()
        
        return sample


class NormalizeIntensity:
    """Normalize image intensity to [0, 1]."""
    
    def __init__(self, min_val: float = -1024.0, max_val: float = 4096.0):
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if 'image' in sample:
            image = sample['image']
            image = np.clip(image, self.min_val, self.max_val)
            image = (image - self.min_val) / (self.max_val - self.min_val)
            sample['image'] = image.astype(np.float32)
        return sample


class RandomFlip:
    """Random flip along each axis."""
    
    def __init__(self, prob: float = 0.5, axes: Tuple[int, ...] = (0, 1, 2)):
        self.prob = prob
        self.axes = axes
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for axis in self.axes:
            if random.random() < self.prob:
                # Flip all spatial arrays
                for key in ['image', 'mask', 'heatmap', 'target']:
                    if key in sample and sample[key] is not None:
                        sample[key] = np.flip(sample[key], axis=axis).copy()
                
                # Update landmarks if present
                if 'landmarks' in sample and sample['landmarks'] is not None:
                    landmarks = sample['landmarks'].copy()
                    size = sample.get('size', sample['image'].shape)
                    
                    # Flip coordinate along axis
                    # axis 0 = Z, axis 1 = Y, axis 2 = X in [Z, Y, X] array
                    coord_idx = 2 - axis  # Map array axis to XYZ index
                    for label in landmarks:
                        landmarks[label][coord_idx] = size[axis] - 1 - landmarks[label][coord_idx]
                    
                    sample['landmarks'] = landmarks
        
        return sample


class RandomRotation:
    """Random rotation around each axis."""
    
    def __init__(
        self,
        angle_range: float = 15.0,
        axes: Tuple[Tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2)),
        prob: float = 0.5
    ):
        self.angle_range = angle_range
        self.axes = axes
        self.prob = prob
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for rotation_axes in self.axes:
            if random.random() < self.prob:
                angle = random.uniform(-self.angle_range, self.angle_range)
                
                # Rotate all spatial arrays
                for key in ['image', 'mask', 'heatmap', 'target']:
                    if key in sample and sample[key] is not None:
                        order = 1 if key == 'image' else 0  # Linear for image, nearest for labels
                        sample[key] = rotate(
                            sample[key],
                            angle,
                            axes=rotation_axes,
                            reshape=False,
                            order=order,
                            mode='constant',
                            cval=0.0
                        )
        
        return sample


class RandomScale:
    """Random scaling."""
    
    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        prob: float = 0.5
    ):
        self.scale_range = scale_range
        self.prob = prob
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.prob:
            scale = random.uniform(*self.scale_range)
            
            # Store original shape
            original_shape = sample['image'].shape
            
            # Scale all spatial arrays
            for key in ['image', 'mask', 'heatmap', 'target']:
                if key in sample and sample[key] is not None:
                    order = 1 if key == 'image' else 0
                    scaled = zoom(sample[key], scale, order=order, mode='constant', cval=0.0)
                    
                    # Crop or pad to original shape
                    sample[key] = self._crop_or_pad(scaled, original_shape)
            
            # Update landmarks if present
            if 'landmarks' in sample and sample['landmarks'] is not None:
                landmarks = sample['landmarks'].copy()
                center = np.array(original_shape) / 2
                
                for label in landmarks:
                    # Scale relative to center
                    landmarks[label] = center + (landmarks[label] - center) * scale
                
                sample['landmarks'] = landmarks
        
        return sample
    
    def _crop_or_pad(self, array: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """Crop or pad array to target shape."""
        result = np.zeros(target_shape, dtype=array.dtype)
        
        # Calculate overlap
        src_start = [max(0, (s - t) // 2) for s, t in zip(array.shape, target_shape)]
        src_end = [min(s, src + t) for s, src, t in zip(array.shape, src_start, target_shape)]
        
        dst_start = [max(0, (t - s) // 2) for s, t in zip(array.shape, target_shape)]
        dst_end = [dst + (se - ss) for dst, ss, se in zip(dst_start, src_start, src_end)]
        
        # Copy data
        result[
            dst_start[0]:dst_end[0],
            dst_start[1]:dst_end[1],
            dst_start[2]:dst_end[2]
        ] = array[
            src_start[0]:src_end[0],
            src_start[1]:src_end[1],
            src_start[2]:src_end[2]
        ]
        
        return result


class RandomTranslation:
    """Random translation."""
    
    def __init__(
        self,
        translation_range: float = 0.2,  # Fraction of image size
        prob: float = 0.5
    ):
        self.translation_range = translation_range
        self.prob = prob
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.prob:
            shape = sample['image'].shape
            
            # Random translation in voxels
            translation = [
                random.uniform(-self.translation_range, self.translation_range) * s
                for s in shape
            ]
            translation = np.array(translation)
            
            # Translate all spatial arrays
            for key in ['image', 'mask', 'heatmap', 'target']:
                if key in sample and sample[key] is not None:
                    order = 1 if key == 'image' else 0
                    sample[key] = self._translate(sample[key], translation, order)
            
            # Update landmarks if present
            if 'landmarks' in sample and sample['landmarks'] is not None:
                landmarks = sample['landmarks'].copy()
                
                for label in landmarks:
                    # Translation in [Z, Y, X] order
                    landmarks[label] += translation[::-1]  # Convert to [X, Y, Z]
                
                sample['landmarks'] = landmarks
        
        return sample
    
    def _translate(self, array: np.ndarray, translation: np.ndarray, order: int) -> np.ndarray:
        """Translate array using affine transform."""
        # Create affine matrix for translation
        matrix = np.eye(3)
        offset = -translation
        
        return affine_transform(
            array,
            matrix,
            offset=offset,
            order=order,
            mode='constant',
            cval=0.0
        )


class RandomGaussianNoise:
    """Add random Gaussian noise to image."""
    
    def __init__(self, std_range: Tuple[float, float] = (0.0, 0.05), prob: float = 0.5):
        self.std_range = std_range
        self.prob = prob
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.prob and 'image' in sample:
            std = random.uniform(*self.std_range)
            noise = np.random.normal(0, std, sample['image'].shape).astype(np.float32)
            sample['image'] = sample['image'] + noise
            sample['image'] = np.clip(sample['image'], 0, 1)
        
        return sample


class RandomGaussianBlur:
    """Apply random Gaussian blur to image."""
    
    def __init__(self, sigma_range: Tuple[float, float] = (0.0, 1.0), prob: float = 0.3):
        self.sigma_range = sigma_range
        self.prob = prob
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.prob and 'image' in sample:
            from scipy.ndimage import gaussian_filter
            sigma = random.uniform(*self.sigma_range)
            sample['image'] = gaussian_filter(sample['image'], sigma=sigma)
        
        return sample


class RandomIntensityShift:
    """Random intensity shift and scale."""
    
    def __init__(
        self,
        shift_range: Tuple[float, float] = (-0.1, 0.1),
        scale_range: Tuple[float, float] = (0.9, 1.1),
        prob: float = 0.5
    ):
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.prob = prob
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.prob and 'image' in sample:
            shift = random.uniform(*self.shift_range)
            scale = random.uniform(*self.scale_range)
            
            sample['image'] = sample['image'] * scale + shift
            sample['image'] = np.clip(sample['image'], 0, 1)
        
        return sample


def get_train_transforms(
    stage: str = 'spine_loc',
    intensity_min: float = -1024.0,
    intensity_max: float = 4096.0
) -> Compose:
    """
    Get training transforms for a specific stage.
    
    Args:
        stage: 'spine_loc', 'vert_loc', or 'vert_seg'
        intensity_min: Minimum HU value
        intensity_max: Maximum HU value
    
    Returns:
        Compose transform
    """
    # Base transforms
    transforms = [
        NormalizeIntensity(min_val=intensity_min, max_val=intensity_max),
    ]
    
    # Augmentation (stage-specific parameters)
    if stage == 'spine_loc':
        transforms.extend([
            RandomFlip(prob=0.5, axes=(0, 1, 2)),
            RandomRotation(angle_range=15.0, prob=0.5),
            RandomScale(scale_range=(0.85, 1.15), prob=0.5),
            RandomTranslation(translation_range=0.2, prob=0.5),
        ])
    elif stage == 'vert_loc':
        transforms.extend([
            RandomFlip(prob=0.5, axes=(0, 1, 2)),
            RandomRotation(angle_range=15.0, prob=0.5),
            RandomScale(scale_range=(0.85, 1.15), prob=0.5),
            RandomTranslation(translation_range=0.2, prob=0.5),
        ])
    elif stage == 'vert_seg':
        transforms.extend([
            RandomFlip(prob=0.5, axes=(0, 1, 2)),
            RandomRotation(angle_range=30.0, prob=0.5),  # Larger rotation for seg
            RandomScale(scale_range=(0.85, 1.15), prob=0.5),
            RandomTranslation(translation_range=0.2, prob=0.5),
        ])
    
    # Intensity augmentation (all stages)
    transforms.extend([
        RandomGaussianNoise(std_range=(0.0, 0.03), prob=0.3),
        RandomIntensityShift(shift_range=(-0.05, 0.05), scale_range=(0.95, 1.05), prob=0.3),
    ])
    
    # Convert to tensor
    transforms.append(ToTensor())
    
    return Compose(transforms)


def get_val_transforms(
    intensity_min: float = -1024.0,
    intensity_max: float = 4096.0
) -> Compose:
    """
    Get validation transforms (no augmentation).
    
    Args:
        intensity_min: Minimum HU value
        intensity_max: Maximum HU value
    
    Returns:
        Compose transform
    """
    return Compose([
        NormalizeIntensity(min_val=intensity_min, max_val=intensity_max),
        ToTensor(),
    ])
