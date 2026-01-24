"""
PyTorch Dataset classes for vertebrae segmentation pipeline.
Loads data from CSV file with paths to images, masks, and landmarks.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import random

from .utils import (
    load_image,
    load_landmarks,
    resample_image,
    crop_image,
    generate_gaussian_heatmap,
    generate_multi_landmark_heatmap,
    normalize_intensity,
    compute_spine_center,
    physical_to_voxel,
    get_valid_vertebrae_mask
)


class VerSe19Dataset(Dataset):
    """
    Base dataset class for VerSe19 vertebrae segmentation.
    
    Loads data from a CSV file with columns:
    - name: Subject ID
    - type: 'train', 'val', or 'test'
    - image_path: Path to CT volume (.nii or .nii.gz)
    - mask_path: Path to segmentation mask
    - centroid_json_path: Path to vertebra centroids JSON
    
    Args:
        csv_path: Path to dataset CSV file
        split: Data split ('train', 'val', 'test', or None for all)
        fold: Cross-validation fold (0-4) or None for all
        transform: Optional data transforms
    """
    
    def __init__(
        self,
        csv_path: Union[str, Path],
        split: Optional[str] = None,
        fold: Optional[int] = None,
        transform: Optional[Any] = None
    ):
        self.csv_path = Path(csv_path)
        self.transform = transform
        
        # Load CSV
        self.df = pd.read_csv(self.csv_path)
        
        # Filter by split
        if split is not None:
            self.df = self.df[self.df['type'] == split].reset_index(drop=True)
        
        # Filter by fold (if fold column exists)
        if fold is not None and 'fold' in self.df.columns:
            if split == 'train':
                # Use all folds except the specified one for training
                self.df = self.df[self.df['fold'] != fold].reset_index(drop=True)
            else:
                # Use only the specified fold for validation
                self.df = self.df[self.df['fold'] == fold].reset_index(drop=True)
        
        self._build_sample_list()
    
    def _build_sample_list(self):
        """Build list of samples. Override in subclasses."""
        self.samples = []
        for idx, row in self.df.iterrows():
            self.samples.append({
                'name': row['name'],
                'image_path': row['image_path'],
                'mask_path': row['mask_path'],
                'centroid_json_path': row['centroid_json_path']
            })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement __getitem__")


class SpineLocalizationDataset(VerSe19Dataset):
    """
    Dataset for Stage 1: Spine Localization.
    
    Input: Full CT volume (downsampled)
    Target: Gaussian heatmap at spine center
    
    Args:
        csv_path: Path to dataset CSV
        split: Data split
        fold: Cross-validation fold
        transform: Data transforms
        image_size: Output image size [X, Y, Z]
        image_spacing: Target spacing in mm [X, Y, Z]
        heatmap_sigma: Gaussian sigma for spine heatmap (voxels)
    """
    
    def __init__(
        self,
        csv_path: Union[str, Path],
        split: Optional[str] = None,
        fold: Optional[int] = None,
        transform: Optional[Any] = None,
        image_size: Tuple[int, int, int] = (64, 64, 128),
        image_spacing: Tuple[float, float, float] = (8.0, 8.0, 8.0),
        heatmap_sigma: float = 2.0
    ):
        self.image_size = image_size
        self.image_spacing = np.array(image_spacing)
        self.heatmap_sigma = heatmap_sigma
        
        super().__init__(csv_path, split, fold, transform)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_info = self.samples[idx]
        
        # Load image
        image, metadata = load_image(sample_info['image_path'])
        original_spacing = metadata['spacing']
        
        # Load landmarks and compute spine center
        landmarks, _ = load_landmarks(sample_info['centroid_json_path'])
        spine_center_physical = compute_spine_center(landmarks)
        
        # Convert to voxel coordinates in original image
        spine_center_voxel = physical_to_voxel(
            spine_center_physical,
            original_spacing,
            metadata['origin'],
            metadata.get('direction')
        )
        
        # Resample image to target spacing
        image_resampled = resample_image(
            image,
            original_spacing,
            self.image_spacing,
            target_size=self.image_size,
            order=1
        )
        
        # Convert spine center to resampled coordinates
        scale_factor = original_spacing / self.image_spacing
        spine_center_resampled = spine_center_voxel * scale_factor
        
        # Clamp to valid range
        spine_center_resampled = np.clip(
            spine_center_resampled,
            [0, 0, 0],
            [s - 1 for s in self.image_size]
        )
        
        # Generate target heatmap
        target = generate_gaussian_heatmap(
            self.image_size,
            spine_center_resampled,
            self.heatmap_sigma
        )
        
        # Build sample dict
        sample = {
            'image': image_resampled,
            'target': target,
            'name': sample_info['name'],
            'spine_center': spine_center_resampled,
            'original_spacing': original_spacing,
            'size': self.image_size
        }
        
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class VertebraeLocalizationDataset(VerSe19Dataset):
    """
    Dataset for Stage 2: Vertebrae Localization.
    
    Input: Cropped CT volume centered on spine
    Target: Multi-channel heatmaps (one per vertebra)
    
    Args:
        csv_path: Path to dataset CSV
        split: Data split
        fold: Cross-validation fold
        transform: Data transforms
        image_size: Output image size [X, Y, Z]
        image_spacing: Target spacing in mm [X, Y, Z]
        num_landmarks: Number of vertebra classes
        heatmap_sigma: Gaussian sigma for vertebra heatmaps (voxels)
        spine_landmarks_path: Path to spine landmarks CSV (from Stage 1)
    """
    
    def __init__(
        self,
        csv_path: Union[str, Path],
        split: Optional[str] = None,
        fold: Optional[int] = None,
        transform: Optional[Any] = None,
        image_size: Tuple[int, int, int] = (96, 96, 128),
        image_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        num_landmarks: int = 26,
        heatmap_sigma: float = 4.0,
        spine_landmarks_path: Optional[Union[str, Path]] = None
    ):
        self.image_size = image_size
        self.image_spacing = np.array(image_spacing)
        self.num_landmarks = num_landmarks
        self.heatmap_sigma = heatmap_sigma
        self.spine_landmarks_path = spine_landmarks_path
        
        # Load spine landmarks if provided (for inference)
        self.spine_landmarks = None
        if spine_landmarks_path is not None:
            self.spine_landmarks = pd.read_csv(spine_landmarks_path)
        
        super().__init__(csv_path, split, fold, transform)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_info = self.samples[idx]
        
        # Load image
        image, metadata = load_image(sample_info['image_path'])
        original_spacing = metadata['spacing']
        
        # Load landmarks
        landmarks, _ = load_landmarks(sample_info['centroid_json_path'])
        
        # Get spine center (from Stage 1 output or compute from landmarks)
        if self.spine_landmarks is not None:
            row = self.spine_landmarks[self.spine_landmarks['name'] == sample_info['name']]
            spine_center_physical = np.array([row['x'].values[0], row['y'].values[0], row['z'].values[0]])
        else:
            spine_center_physical = compute_spine_center(landmarks)
        
        # Convert spine center to voxel coordinates
        spine_center_voxel = physical_to_voxel(
            spine_center_physical,
            original_spacing,
            metadata['origin']
        )
        
        # Calculate crop size in original spacing
        crop_size_physical = np.array(self.image_size) * self.image_spacing
        crop_size_voxel = (crop_size_physical / original_spacing).astype(int)
        
        # Crop around spine center
        image_cropped = crop_image(image, spine_center_voxel, tuple(crop_size_voxel))
        
        # Resample to target spacing/size
        image_resampled = resample_image(
            image_cropped,
            original_spacing,
            self.image_spacing,
            target_size=self.image_size,
            order=1
        )
        
        # Convert landmarks to cropped/resampled coordinates
        landmarks_resampled = {}
        crop_origin = spine_center_voxel - crop_size_voxel / 2
        
        for label, centroid in landmarks.items():
            # Convert to voxel in original
            centroid_voxel = physical_to_voxel(centroid, original_spacing, metadata['origin'])
            
            # Shift to crop origin
            centroid_cropped = centroid_voxel - crop_origin
            
            # Scale to resampled size
            scale = crop_size_voxel / np.array(self.image_size)
            centroid_resampled = centroid_cropped / scale
            
            landmarks_resampled[label] = centroid_resampled
        
        # Generate multi-channel target heatmaps
        target = generate_multi_landmark_heatmap(
            self.image_size,
            landmarks_resampled,
            self.heatmap_sigma,
            num_classes=self.num_landmarks
        )
        
        # Valid vertebrae mask
        valid_mask = get_valid_vertebrae_mask(landmarks, self.num_landmarks)
        
        # Build sample dict
        sample = {
            'image': image_resampled,
            'target': target,
            'valid_mask': valid_mask,
            'name': sample_info['name'],
            'landmarks': landmarks_resampled,
            'spine_center': spine_center_voxel,
            'original_spacing': original_spacing,
            'size': self.image_size
        }
        
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class VertebraeSegmentationDataset(VerSe19Dataset):
    """
    Dataset for Stage 3: Vertebrae Segmentation.
    
    Input: Cropped CT volume + heatmap prior centered on ONE vertebra
    Target: Binary segmentation mask
    
    This dataset expands to one sample per vertebra per image.
    
    Args:
        csv_path: Path to dataset CSV
        split: Data split
        fold: Cross-validation fold
        transform: Data transforms
        image_size: Output image size [X, Y, Z]
        image_spacing: Target spacing in mm [X, Y, Z]
        heatmap_sigma: Gaussian sigma for vertebra heatmap prior (voxels)
        vertebrae_landmarks_path: Path to vertebrae landmarks CSV (from Stage 2)
    """
    
    def __init__(
        self,
        csv_path: Union[str, Path],
        split: Optional[str] = None,
        fold: Optional[int] = None,
        transform: Optional[Any] = None,
        image_size: Tuple[int, int, int] = (128, 128, 96),
        image_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        heatmap_sigma: float = 3.0,
        vertebrae_landmarks_path: Optional[Union[str, Path]] = None
    ):
        self.image_size = image_size
        self.image_spacing = np.array(image_spacing)
        self.heatmap_sigma = heatmap_sigma
        self.vertebrae_landmarks_path = vertebrae_landmarks_path
        
        super().__init__(csv_path, split, fold, transform)
    
    def _build_sample_list(self):
        """Build sample list - one sample per vertebra per image."""
        self.samples = []
        
        for idx, row in self.df.iterrows():
            # Load landmarks to know which vertebrae exist
            landmarks, _ = load_landmarks(row['centroid_json_path'])
            
            for label, centroid in landmarks.items():
                self.samples.append({
                    'name': row['name'],
                    'image_path': row['image_path'],
                    'mask_path': row['mask_path'],
                    'centroid_json_path': row['centroid_json_path'],
                    'vertebra_label': label,
                    'vertebra_centroid': centroid
                })
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_info = self.samples[idx]
        
        # Load image
        image, metadata = load_image(sample_info['image_path'])
        original_spacing = metadata['spacing']
        
        # Load mask
        mask, _ = load_image(sample_info['mask_path'])
        
        # Get vertebra info
        vert_label = sample_info['vertebra_label']
        vert_centroid = sample_info['vertebra_centroid']
        
        # Convert centroid to voxel coordinates
        vert_center_voxel = physical_to_voxel(
            vert_centroid,
            original_spacing,
            metadata['origin']
        )
        
        # Calculate crop size in original spacing
        crop_size_physical = np.array(self.image_size) * self.image_spacing
        crop_size_voxel = (crop_size_physical / original_spacing).astype(int)
        
        # Crop image around vertebra center
        image_cropped = crop_image(image, vert_center_voxel, tuple(crop_size_voxel), pad_value=-1024)
        mask_cropped = crop_image(mask, vert_center_voxel, tuple(crop_size_voxel), pad_value=0)
        
        # Resample to target spacing/size
        image_resampled = resample_image(
            image_cropped,
            original_spacing,
            self.image_spacing,
            target_size=self.image_size,
            order=1
        )
        mask_resampled = resample_image(
            mask_cropped,
            original_spacing,
            self.image_spacing,
            target_size=self.image_size,
            order=0  # Nearest neighbor for labels
        )
        
        # Create binary mask for this vertebra
        target = (mask_resampled == vert_label).astype(np.float32)
        
        # Generate heatmap prior (centered on vertebra)
        center_resampled = np.array(self.image_size) / 2  # Center of crop
        heatmap = generate_gaussian_heatmap(
            self.image_size,
            center_resampled,
            self.heatmap_sigma
        )
        
        # Build sample dict
        sample = {
            'image': image_resampled,
            'heatmap': heatmap,
            'target': target,
            'name': sample_info['name'],
            'vertebra_label': vert_label,
            'original_spacing': original_spacing,
            'size': self.image_size
        }
        
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
        
        # Concatenate image and heatmap for model input
        if isinstance(sample['image'], torch.Tensor):
            sample['input'] = torch.cat([sample['image'], sample['heatmap']], dim=0)
        else:
            sample['input'] = np.concatenate([
                sample['image'][np.newaxis, ...],
                sample['heatmap'][np.newaxis, ...]
            ], axis=0)
        
        return sample


def create_fold_column(csv_path: Union[str, Path], n_folds: int = 5, random_state: int = 42):
    """
    Add fold column to dataset CSV for cross-validation.
    
    Args:
        csv_path: Path to dataset CSV
        n_folds: Number of folds
        random_state: Random seed
    """
    from sklearn.model_selection import KFold
    
    df = pd.read_csv(csv_path)
    
    # Only assign folds to training data
    train_mask = df['type'] == 'train'
    train_indices = df[train_mask].index.values
    
    # Initialize fold column
    df['fold'] = -1
    
    # Assign folds
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    for fold, (_, val_idx) in enumerate(kf.split(train_indices)):
        actual_indices = train_indices[val_idx]
        df.loc[actual_indices, 'fold'] = fold
    
    # Save updated CSV
    output_path = csv_path.replace('.csv', '_with_folds.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Saved CSV with folds to: {output_path}")
    print(f"Fold distribution:\n{df[df['fold'] >= 0]['fold'].value_counts().sort_index()}")
    
    return df


if __name__ == "__main__":
    # Test dataset
    print("Testing datasets...")
    
    # Example (would need actual CSV file)
    # dataset = SpineLocalizationDataset(
    #     csv_path="verse19_dataset.csv",
    #     split="train",
    #     image_size=(64, 64, 128),
    #     image_spacing=(8.0, 8.0, 8.0)
    # )
    # print(f"Number of samples: {len(dataset)}")
    
    print("Dataset tests passed!")
