"""
Training script for Stage 1: Spine Localization.

This stage trains a U-Net to predict a heatmap centered on the spine.
The predicted centroid is used to crop the volume for Stage 2.
"""
import argparse
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage

from config.spine_localization import SpineLocalizationConfig
from models.unet3d import UNet3DSpineLocalization
from data.dataset import SpineLocalizationDataset
from data.transforms import get_train_transforms, get_val_transforms
from .trainer import BaseTrainer
from .losses import HeatmapMSELoss


class SpineLocalizationTrainer(BaseTrainer):
    """
    Trainer for Stage 1: Spine Localization.
    
    Architecture: UNet3D with single output channel for spine heatmap
    Loss: Mean Squared Error on heatmaps
    Metric: Mean Localization Distance (MLD) in mm
    """
    
    def __init__(
        self,
        output_dir: str,
        csv_path: str,
        fold: int = 0,
        num_folds: int = 5,
        device: str = 'cuda',
        config: Optional[SpineLocalizationConfig] = None,
        multi_gpu: bool = False,
        **kwargs
    ):
        if config is None:
            config = SpineLocalizationConfig()
        
        super().__init__(
            config=config,
            output_dir=output_dir,
            csv_path=csv_path,
            fold=fold,
            num_folds=num_folds,
            device=device,
            multi_gpu=multi_gpu,
            **kwargs
        )
    
    def create_model(self) -> nn.Module:
        """Create U-Net for spine localization."""
        model = UNet3DSpineLocalization(
            in_channels=1,
            num_filters_base=self.config.num_filters_base,
            num_levels=self.config.num_levels
        )
        return model
    
    def create_datasets(self) -> Tuple[Any, Any]:
        """Create training and validation datasets."""
        train_transforms = get_train_transforms(stage='spine_localization')
        val_transforms = get_val_transforms()
        
        train_dataset = SpineLocalizationDataset(
            csv_path=self.csv_path,
            split='train',
            fold=self.fold,
            transform=train_transforms,
            image_size=self.config.image_size,
            image_spacing=self.config.image_spacing,
            heatmap_sigma=self.config.heatmap_sigma
        )
        
        val_dataset = SpineLocalizationDataset(
            csv_path=self.csv_path,
            split='val',
            fold=self.fold,
            transform=val_transforms,
            image_size=self.config.image_size,
            image_spacing=self.config.image_spacing,
            heatmap_sigma=self.config.heatmap_sigma
        )
        
        return train_dataset, val_dataset
    
    def create_loss_function(self):
        """Create MSE loss for heatmap regression."""
        return HeatmapMSELoss()
    
    def train_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Single training step.
        
        Args:
            batch: Dictionary with 'image' and 'heatmap' tensors
        
        Returns:
            Dictionary with loss values
        """
        self.optimizer.zero_grad()
        
        image = batch['image']  # [B, 1, D, H, W]
        target = batch['heatmap']  # [B, 1, D, H, W]
        
        # Forward pass
        pred = self.model(image)  # [B, 1, D, H, W]
        
        # Compute loss
        loss = self.loss_fn(pred, target)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if hasattr(self.config, 'grad_clip') and self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        
        self.optimizer.step()
        
        return {'total': loss}
    
    def val_step(self, batch: Dict) -> Dict[str, Any]:
        """
        Single validation step.
        
        Args:
            batch: Dictionary with 'image', 'heatmap', and 'centroid' tensors
        
        Returns:
            Dictionary with losses, predictions, and targets
        """
        image = batch['image']
        target = batch['heatmap']
        
        # Forward pass
        pred = self.model(image)
        
        # Compute loss
        loss = self.loss_fn(pred, target)
        
        # Extract predicted centroids
        predictions = []
        targets = []
        
        for i in range(pred.shape[0]):
            pred_heatmap = pred[i, 0].cpu().numpy()
            target_centroid = batch['centroid'][i].cpu().numpy() if 'centroid' in batch else None
            spacing = batch['spacing'][i].cpu().numpy() if 'spacing' in batch else self.config.image_spacing
            
            # Find peak in predicted heatmap
            pred_centroid = self._find_heatmap_peak(pred_heatmap)
            
            predictions.append({
                'centroid': pred_centroid,
                'spacing': spacing
            })
            
            if target_centroid is not None:
                targets.append({
                    'centroid': target_centroid,
                    'spacing': spacing
                })
        
        return {
            'losses': {'total': loss},
            'predictions': predictions,
            'targets': targets
        }
    
    def _find_heatmap_peak(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Find the peak location in a heatmap.
        Uses weighted centroid around the maximum value.
        
        Args:
            heatmap: 3D heatmap array
        
        Returns:
            Peak coordinates [z, y, x]
        """
        # Find maximum location
        max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        # Compute weighted centroid around max for sub-voxel accuracy
        window_size = 5
        z_start = max(0, max_idx[0] - window_size)
        z_end = min(heatmap.shape[0], max_idx[0] + window_size + 1)
        y_start = max(0, max_idx[1] - window_size)
        y_end = min(heatmap.shape[1], max_idx[1] + window_size + 1)
        x_start = max(0, max_idx[2] - window_size)
        x_end = min(heatmap.shape[2], max_idx[2] + window_size + 1)
        
        local_window = heatmap[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Threshold for weighted centroid
        threshold = local_window.max() * 0.5
        local_window = np.maximum(local_window - threshold, 0)
        
        # Compute weighted centroid
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
        else:
            return np.array(max_idx, dtype=np.float64)
    
    def compute_metrics(
        self,
        predictions: List[Dict],
        targets: List[Dict]
    ) -> Dict[str, float]:
        """
        Compute localization metrics.
        
        Args:
            predictions: List of prediction dicts with 'centroid' and 'spacing'
            targets: List of target dicts with 'centroid' and 'spacing'
        
        Returns:
            Dictionary with metric values
        """
        distances = []
        
        for pred, target in zip(predictions, targets):
            if target['centroid'] is None:
                continue
            
            pred_centroid = pred['centroid']
            target_centroid = target['centroid']
            spacing = pred['spacing']
            
            # Compute distance in mm
            diff = (pred_centroid - target_centroid) * spacing
            distance = np.sqrt(np.sum(diff ** 2))
            distances.append(distance)
        
        if len(distances) == 0:
            return {'mld_mm': 0.0}
        
        distances = np.array(distances)
        
        return {
            'mld_mm': float(np.mean(distances)),
            'mld_std_mm': float(np.std(distances)),
            'mld_median_mm': float(np.median(distances)),
            'success_rate_20mm': float(np.mean(distances < 20) * 100),
            'success_rate_10mm': float(np.mean(distances < 10) * 100)
        }


def train_spine_localization(
    csv_path: str,
    output_dir: str,
    fold: int = 0,
    num_folds: int = 5,
    device: str = 'cuda',
    config: Optional[SpineLocalizationConfig] = None,
    resume_from: Optional[str] = None,
    multi_gpu: bool = False
) -> float:
    """
    Train spine localization model for one fold.
    
    Args:
        csv_path: Path to dataset CSV file
        output_dir: Output directory for checkpoints and logs
        fold: Fold index (0 to num_folds-1)
        num_folds: Total number of folds
        device: Device to train on
        config: Optional configuration object
        resume_from: Optional checkpoint path to resume from
        multi_gpu: Whether to use multiple GPUs if available
    
    Returns:
        Best validation loss
    """
    trainer = SpineLocalizationTrainer(
        output_dir=output_dir,
        csv_path=csv_path,
        fold=fold,
        num_folds=num_folds,
        device=device,
        config=config,
        resume_from=resume_from,
        multi_gpu=multi_gpu
    )
    
    best_metric = trainer.train()
    
    return best_metric


def train_all_folds(
    csv_path: str,
    output_dir: str,
    num_folds: int = 5,
    device: str = 'cuda',
    config: Optional[SpineLocalizationConfig] = None,
    multi_gpu: bool = False
) -> Dict[int, float]:
    """
    Train spine localization model for all folds.
    
    Args:
        csv_path: Path to dataset CSV file
        output_dir: Output directory
        num_folds: Number of cross-validation folds
        device: Device to train on
        config: Optional configuration object
        multi_gpu: Whether to use multiple GPUs if available
    
    Returns:
        Dictionary mapping fold index to best validation loss
    """
    results = {}
    
    for fold in range(num_folds):
        print(f"\n{'#'*60}")
        print(f"# Training Fold {fold}/{num_folds-1}")
        print(f"{'#'*60}\n")
        
        best_metric = train_spine_localization(
            csv_path=csv_path,
            output_dir=output_dir,
            fold=fold,
            num_folds=num_folds,
            device=device,
            config=config,
            multi_gpu=multi_gpu
        )
        
        results[fold] = best_metric
    
    # Print summary
    print(f"\n{'='*60}")
    print("Cross-Validation Results")
    print(f"{'='*60}")
    for fold, metric in results.items():
        print(f"Fold {fold}: {metric:.4f}")
    print(f"\nMean: {np.mean(list(results.values())):.4f}")
    print(f"Std:  {np.std(list(results.values())):.4f}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Spine Localization Model')
    parser.add_argument('--csv', type=str, required=True, help='Path to dataset CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--fold', type=int, default=-1, help='Fold to train (-1 for all folds)')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    
    args = parser.parse_args()
    
    if args.fold < 0:
        train_all_folds(
            csv_path=args.csv,
            output_dir=args.output,
            num_folds=args.num_folds,
            device=args.device
        )
    else:
        train_spine_localization(
            csv_path=args.csv,
            output_dir=args.output,
            fold=args.fold,
            num_folds=args.num_folds,
            device=args.device,
            resume_from=args.resume
        )
