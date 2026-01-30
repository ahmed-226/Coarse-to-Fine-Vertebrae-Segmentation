"""
Training script for Stage 2: Vertebrae Localization.

This stage trains a SpatialConfiguration-Net (SCNet) to predict heatmaps
for each vertebra centroid. Uses learnable sigma for per-vertebra precision.
"""
import argparse
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage

from config.vertebrae_localization import VertebraeLocalizationConfig
from models.scnet import SpatialConfigurationNet
from data.dataset import VertebraeLocalizationDataset
from data.transforms import get_train_transforms, get_val_transforms
from .trainer import BaseTrainer
from .losses import CombinedLocalizationLoss


class VertebraeLocalizationTrainer(BaseTrainer):
    """
    Trainer for Stage 2: Vertebrae Localization.
    
    Architecture: SCNet (Local Appearance Net + Spatial Configuration Net)
    Loss: MSE on heatmaps + sigma regularization
    Metric: Mean Localization Distance (MLD) per vertebra
    """
    
    def __init__(
        self,
        output_dir: str,
        csv_path: str,
        fold: int = 0,
        num_folds: int = 5,
        device: str = 'cuda',
        config: Optional[VertebraeLocalizationConfig] = None,
        spine_model_path: Optional[str] = None,
        multi_gpu: bool = False,
        **kwargs
    ):
        """
        Args:
            spine_model_path: Path to trained spine localization model for cropping
            multi_gpu: Whether to use multiple GPUs if available
        """
        if config is None:
            config = VertebraeLocalizationConfig()
        
        self.spine_model_path = spine_model_path
        
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
        """Create SCNet for vertebrae localization."""
        model = SpatialConfigurationNet(
            num_landmarks=self.config.num_landmarks,
            num_filters_base=self.config.num_filters_base,
            num_levels=self.config.num_levels,
            initial_sigma=self.config.heatmap_sigma,
            learnable_sigma=self.config.learnable_sigma
        )
        return model
    
    def create_datasets(self) -> Tuple[Any, Any]:
        """Create training and validation datasets."""
        train_transforms = get_train_transforms(stage='vertebrae_localization')
        val_transforms = get_val_transforms()
        
        train_dataset = VertebraeLocalizationDataset(
            csv_path=self.csv_path,
            split='train',
            fold=self.fold,
            transform=train_transforms,
            image_size=self.config.image_size,
            image_spacing=self.config.image_spacing,
            num_landmarks=self.config.num_landmarks,
            heatmap_sigma=self.config.heatmap_sigma
        )
        
        val_dataset = VertebraeLocalizationDataset(
            csv_path=self.csv_path,
            split='val',
            fold=self.fold,
            transform=val_transforms,
            image_size=self.config.image_size,
            image_spacing=self.config.image_spacing,
            num_landmarks=self.config.num_landmarks,
            heatmap_sigma=self.config.heatmap_sigma
        )
        
        return train_dataset, val_dataset
    
    def create_loss_function(self):
        """Create combined localization loss."""
        return CombinedLocalizationLoss(
            heatmap_loss_type='mse',
            sigma_regularization=self.config.sigma_regularization,
            target_sigma=self.config.heatmap_sigma
        )
    
    def train_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Single training step.
        
        Args:
            batch: Dictionary with 'image', 'heatmaps', 'valid_mask'
        
        Returns:
            Dictionary with loss values
        """
        self.optimizer.zero_grad()
        
        image = batch['image']  # [B, 1, D, H, W]
        target = batch['heatmaps']  # [B, N, D, H, W]
        valid_mask = batch.get('valid_mask', None)  # [B, N]
        
        # Forward pass
        pred = self.model(image)  # [B, N, D, H, W]
        # Get learnable sigma from model [N, 3]
        sigma = self.model.sigma if hasattr(self.model, 'sigma') else None
        
        # Compute loss
        losses = self.loss_fn(pred, target, sigma, valid_mask)
        
        # Backward pass
        losses['total'].backward()
        
        # Gradient clipping
        if hasattr(self.config, 'grad_clip') and self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        
        self.optimizer.step()
        
        return losses
    
    def val_step(self, batch: Dict) -> Dict[str, Any]:
        """
        Single validation step.
        
        Args:
            batch: Dictionary with 'image', 'heatmaps', 'landmarks', 'valid_mask'
        
        Returns:
            Dictionary with losses, predictions, and targets
        """
        image = batch['image']
        target = batch['heatmaps']
        valid_mask = batch.get('valid_mask', None)
        
        # Forward pass
        pred = self.model(image)  # [B, N, D, H, W]
        # Get learnable sigma from model [N, 3]
        sigma = self.model.sigma if hasattr(self.model, 'sigma') else None
        
        # Compute loss
        losses = self.loss_fn(pred, target, sigma, valid_mask)
        
        # Extract predicted landmarks
        predictions = []
        targets = []
        
        for i in range(pred.shape[0]):
            pred_heatmaps = pred[i].cpu().numpy()  # [N, D, H, W]
            target_landmarks = batch['landmarks'][i].cpu().numpy() if 'landmarks' in batch else None
            mask = valid_mask[i].cpu().numpy() if valid_mask is not None else None
            spacing = batch.get('spacing', self.config.image_spacing)
            if isinstance(spacing, torch.Tensor):
                spacing = spacing.cpu().numpy()
            
            # Find peaks in predicted heatmaps
            pred_landmarks = self._find_all_peaks(pred_heatmaps, mask)
            
            predictions.append({
                'landmarks': pred_landmarks,
                'valid_mask': mask,
                'spacing': spacing
            })
            
            if target_landmarks is not None:
                targets.append({
                    'landmarks': target_landmarks,
                    'valid_mask': mask,
                    'spacing': spacing
                })
        
        return {
            'losses': {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()},
            'predictions': predictions,
            'targets': targets
        }
    
    def _find_all_peaks(
        self,
        heatmaps: np.ndarray,
        valid_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Find peak locations in all heatmaps.
        
        Args:
            heatmaps: [N, D, H, W] array of heatmaps
            valid_mask: [N] boolean mask for valid landmarks
        
        Returns:
            [N, 3] array of peak coordinates
        """
        num_landmarks = heatmaps.shape[0]
        peaks = np.zeros((num_landmarks, 3))
        
        for i in range(num_landmarks):
            if valid_mask is not None and not valid_mask[i]:
                peaks[i] = np.array([np.nan, np.nan, np.nan])
                continue
            
            peaks[i] = self._find_heatmap_peak(heatmaps[i])
        
        return peaks
    
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
        max_val = heatmap.max()
        if max_val < 1e-8:
            return np.array([heatmap.shape[0]/2, heatmap.shape[1]/2, heatmap.shape[2]/2])
        
        max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        # Compute weighted centroid around max for sub-voxel accuracy
        window_size = 3
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
        Compute localization metrics per vertebra.
        
        Args:
            predictions: List of prediction dicts with 'landmarks', 'valid_mask', 'spacing'
            targets: List of target dicts with same structure
        
        Returns:
            Dictionary with metric values
        """
        # Vertebra names for reporting
        vertebra_names = ['C' + str(i) for i in range(1, 8)] + \
                         ['T' + str(i) for i in range(1, 13)] + \
                         ['L' + str(i) for i in range(1, 7)] + \
                         ['S1', 'S2']
        
        all_distances = []
        per_vertebra_distances = {name: [] for name in vertebra_names}
        
        for pred, target in zip(predictions, targets):
            pred_landmarks = pred['landmarks']
            target_landmarks = target['landmarks']
            valid_mask = pred['valid_mask']
            spacing = pred['spacing']
            
            for i in range(len(pred_landmarks)):
                if valid_mask is not None and not valid_mask[i]:
                    continue
                
                if np.any(np.isnan(pred_landmarks[i])) or np.any(np.isnan(target_landmarks[i])):
                    continue
                
                # Compute distance in mm
                diff = (pred_landmarks[i] - target_landmarks[i]) * spacing
                distance = np.sqrt(np.sum(diff ** 2))
                
                all_distances.append(distance)
                per_vertebra_distances[vertebra_names[i]].append(distance)
        
        if len(all_distances) == 0:
            return {'mld_mm': 0.0}
        
        all_distances = np.array(all_distances)
        
        metrics = {
            'mld_mm': float(np.mean(all_distances)),
            'mld_std_mm': float(np.std(all_distances)),
            'mld_median_mm': float(np.median(all_distances)),
            'id_rate_4mm': float(np.mean(all_distances < 4) * 100),
            'id_rate_10mm': float(np.mean(all_distances < 10) * 100),
            'id_rate_20mm': float(np.mean(all_distances < 20) * 100)
        }
        
        # Per-region statistics
        cervical_dists = []
        thoracic_dists = []
        lumbar_dists = []
        
        for name, dists in per_vertebra_distances.items():
            if len(dists) == 0:
                continue
            if name.startswith('C'):
                cervical_dists.extend(dists)
            elif name.startswith('T'):
                thoracic_dists.extend(dists)
            elif name.startswith('L') or name.startswith('S'):
                lumbar_dists.extend(dists)
        
        if cervical_dists:
            metrics['mld_cervical_mm'] = float(np.mean(cervical_dists))
        if thoracic_dists:
            metrics['mld_thoracic_mm'] = float(np.mean(thoracic_dists))
        if lumbar_dists:
            metrics['mld_lumbar_mm'] = float(np.mean(lumbar_dists))
        
        return metrics


def train_vertebrae_localization(
    csv_path: str,
    output_dir: str,
    fold: int = 0,
    num_folds: int = 5,
    device: str = 'cuda',
    config: Optional[VertebraeLocalizationConfig] = None,
    spine_model_path: Optional[str] = None,
    resume_from: Optional[str] = None,
    multi_gpu: bool = False
) -> float:
    """
    Train vertebrae localization model for one fold.
    
    Args:
        csv_path: Path to dataset CSV file
        output_dir: Output directory for checkpoints and logs
        fold: Fold index (0 to num_folds-1)
        num_folds: Total number of folds
        device: Device to train on
        config: Optional configuration object
        spine_model_path: Path to trained spine localization model
        resume_from: Optional checkpoint path to resume from
    
    Returns:
        Best validation loss
    """
    trainer = VertebraeLocalizationTrainer(
        output_dir=output_dir,
        csv_path=csv_path,
        fold=fold,
        num_folds=num_folds,
        device=device,
        config=config,
        spine_model_path=spine_model_path,
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
    config: Optional[VertebraeLocalizationConfig] = None,
    spine_model_dir: Optional[str] = None
) -> Dict[int, float]:
    """
    Train vertebrae localization model for all folds.
    
    Args:
        csv_path: Path to dataset CSV file
        output_dir: Output directory
        num_folds: Number of cross-validation folds
        device: Device to train on
        config: Optional configuration object
        spine_model_dir: Directory containing spine models for each fold
    
    Returns:
        Dictionary mapping fold index to best validation loss
    """
    results = {}
    
    for fold in range(num_folds):
        print(f"\n{'#'*60}")
        print(f"# Training Fold {fold}/{num_folds-1}")
        print(f"{'#'*60}\n")
        
        spine_model_path = None
        if spine_model_dir:
            spine_model_path = str(Path(spine_model_dir) / f'fold_{fold}' / 'best.pth')
        
        best_metric = train_vertebrae_localization(
            csv_path=csv_path,
            output_dir=output_dir,
            fold=fold,
            num_folds=num_folds,
            device=device,
            config=config,
            spine_model_path=spine_model_path
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
    parser = argparse.ArgumentParser(description='Train Vertebrae Localization Model')
    parser.add_argument('--csv', type=str, required=True, help='Path to dataset CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--spine_model', type=str, default=None, help='Path to spine model directory')
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
            device=args.device,
            spine_model_dir=args.spine_model
        )
    else:
        train_vertebrae_localization(
            csv_path=args.csv,
            output_dir=args.output,
            fold=args.fold,
            num_folds=args.num_folds,
            device=args.device,
            spine_model_path=args.spine_model,
            resume_from=args.resume
        )
