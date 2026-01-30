"""
Training script for Stage 3: Vertebrae Segmentation.

This stage trains a U-Net to segment individual vertebrae.
Each vertebra is cropped around its centroid and segmented separately.
"""
import argparse
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage

from config.vertebrae_segmentation import VertebraeSegmentationConfig
from models.unet3d import UNet3DVertebraeSegmentation
from data.dataset import VertebraeSegmentationDataset
from data.transforms import get_train_transforms, get_val_transforms
from .trainer import BaseTrainer
from .losses import DiceBCELoss, DiceLoss


class VertebraeSegmentationTrainer(BaseTrainer):
    """
    Trainer for Stage 3: Vertebrae Segmentation.
    
    Architecture: UNet3D with single output channel for binary segmentation
    Loss: Dice + BCE combined loss
    Metric: Dice coefficient, Hausdorff Distance, IoU
    """
    
    def __init__(
        self,
        output_dir: str,
        csv_path: str,
        fold: int = 0,
        num_folds: int = 5,
        device: str = 'cuda',
        config: Optional[VertebraeSegmentationConfig] = None,
        localization_model_path: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            localization_model_path: Path to trained vertebrae localization model
        """
        if config is None:
            config = VertebraeSegmentationConfig()
        
        self.localization_model_path = localization_model_path
        
        super().__init__(
            config=config,
            output_dir=output_dir,
            csv_path=csv_path,
            fold=fold,
            num_folds=num_folds,
            device=device,
            **kwargs
        )
    
    def create_model(self) -> nn.Module:
        """Create U-Net for vertebrae segmentation."""
        model = UNet3DVertebraeSegmentation(
            in_channels=1,
            num_filters_base=self.config.num_filters_base,
            num_levels=self.config.num_levels
        )
        return model
    
    def create_datasets(self) -> Tuple[Any, Any]:
        """Create training and validation datasets."""
        train_transforms = get_train_transforms(stage='vertebrae_segmentation')
        val_transforms = get_val_transforms()
        
        train_dataset = VertebraeSegmentationDataset(
            csv_path=self.csv_path,
            split='train',
            fold=self.fold,
            transform=train_transforms,
            image_size=self.config.image_size,
            image_spacing=self.config.image_spacing,
            heatmap_sigma=self.config.heatmap_sigma
        )
        
        val_dataset = VertebraeSegmentationDataset(
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
        """Create Dice + BCE loss for segmentation."""
        return DiceBCELoss(
            dice_weight=self.config.dice_weight,
            bce_weight=self.config.bce_weight
        )
    
    def train_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Single training step.
        
        Args:
            batch: Dictionary with 'image' and 'mask' tensors
        
        Returns:
            Dictionary with loss values
        """
        self.optimizer.zero_grad()
        
        image = batch['image']  # [B, 1, D, H, W]
        target = batch['mask']  # [B, 1, D, H, W]
        
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
        
        # Compute Dice for logging
        with torch.no_grad():
            dice = self._compute_dice(pred, target)
        
        return {'total': loss, 'dice': dice}
    
    def val_step(self, batch: Dict) -> Dict[str, Any]:
        """
        Single validation step.
        
        Args:
            batch: Dictionary with 'image' and 'mask' tensors
        
        Returns:
            Dictionary with losses, predictions, and targets
        """
        image = batch['image']
        target = batch['mask']
        
        # Forward pass
        pred = self.model(image)
        
        # Compute loss
        loss = self.loss_fn(pred, target)
        
        # Compute Dice
        dice = self._compute_dice(pred, target)
        
        # Extract predictions for metrics
        predictions = []
        targets = []
        
        for i in range(pred.shape[0]):
            pred_mask = (torch.sigmoid(pred[i, 0]) > 0.5).cpu().numpy()
            target_mask = target[i, 0].cpu().numpy()
            spacing = batch['spacing'][i].cpu().numpy() if 'spacing' in batch else self.config.image_spacing
            vertebra_id = batch['vertebra_id'][i].item() if 'vertebra_id' in batch else None
            
            predictions.append({
                'mask': pred_mask,
                'spacing': spacing,
                'vertebra_id': vertebra_id
            })
            
            targets.append({
                'mask': target_mask.astype(bool),
                'spacing': spacing,
                'vertebra_id': vertebra_id
            })
        
        return {
            'losses': {'total': loss, 'dice': dice},
            'predictions': predictions,
            'targets': targets
        }
    
    def _compute_dice(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice coefficient."""
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum()
        
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        
        return dice
    
    def compute_metrics(
        self,
        predictions: List[Dict],
        targets: List[Dict]
    ) -> Dict[str, float]:
        """
        Compute segmentation metrics.
        
        Args:
            predictions: List of prediction dicts with 'mask', 'spacing', 'vertebra_id'
            targets: List of target dicts with same structure
        
        Returns:
            Dictionary with metric values
        """
        all_dice = []
        all_iou = []
        all_hausdorff = []
        all_assd = []
        per_vertebra_dice = {}
        
        vertebra_names = ['C' + str(i) for i in range(1, 8)] + \
                         ['T' + str(i) for i in range(1, 13)] + \
                         ['L' + str(i) for i in range(1, 7)] + \
                         ['S1', 'S2']
        
        for pred, target in zip(predictions, targets):
            pred_mask = pred['mask']
            target_mask = target['mask']
            spacing = pred['spacing']
            vertebra_id = pred['vertebra_id']
            
            # Dice coefficient
            dice = self._dice_coefficient(pred_mask, target_mask)
            all_dice.append(dice)
            
            # IoU
            iou = self._iou(pred_mask, target_mask)
            all_iou.append(iou)
            
            # Hausdorff distance (if both masks are non-empty)
            if pred_mask.sum() > 0 and target_mask.sum() > 0:
                hd = self._hausdorff_distance(pred_mask, target_mask, spacing)
                assd = self._average_surface_distance(pred_mask, target_mask, spacing)
                all_hausdorff.append(hd)
                all_assd.append(assd)
            
            # Per-vertebra stats
            if vertebra_id is not None and 0 <= vertebra_id < len(vertebra_names):
                name = vertebra_names[vertebra_id]
                if name not in per_vertebra_dice:
                    per_vertebra_dice[name] = []
                per_vertebra_dice[name].append(dice)
        
        metrics = {}
        
        if all_dice:
            metrics['dice_mean'] = float(np.mean(all_dice))
            metrics['dice_std'] = float(np.std(all_dice))
            metrics['dice_median'] = float(np.median(all_dice))
        
        if all_iou:
            metrics['iou_mean'] = float(np.mean(all_iou))
        
        if all_hausdorff:
            metrics['hd_mean_mm'] = float(np.mean(all_hausdorff))
            metrics['hd95_mm'] = float(np.percentile(all_hausdorff, 95))
        
        if all_assd:
            metrics['assd_mean_mm'] = float(np.mean(all_assd))
        
        # Per-region Dice
        cervical_dice = []
        thoracic_dice = []
        lumbar_dice = []
        
        for name, dices in per_vertebra_dice.items():
            if name.startswith('C'):
                cervical_dice.extend(dices)
            elif name.startswith('T'):
                thoracic_dice.extend(dices)
            elif name.startswith('L') or name.startswith('S'):
                lumbar_dice.extend(dices)
        
        if cervical_dice:
            metrics['dice_cervical'] = float(np.mean(cervical_dice))
        if thoracic_dice:
            metrics['dice_thoracic'] = float(np.mean(thoracic_dice))
        if lumbar_dice:
            metrics['dice_lumbar'] = float(np.mean(lumbar_dice))
        
        return metrics
    
    def _dice_coefficient(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute Dice coefficient between two binary masks."""
        intersection = np.logical_and(pred, target).sum()
        union = pred.sum() + target.sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return 2.0 * intersection / union
    
    def _iou(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute IoU between two binary masks."""
        intersection = np.logical_and(pred, target).sum()
        union = np.logical_or(pred, target).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def _hausdorff_distance(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        spacing: np.ndarray
    ) -> float:
        """Compute Hausdorff distance in mm."""
        from scipy.ndimage import distance_transform_edt
        
        # Get surface voxels
        pred_surface = self._get_surface(pred)
        target_surface = self._get_surface(target)
        
        if pred_surface.sum() == 0 or target_surface.sum() == 0:
            return np.inf
        
        # Distance transforms
        pred_coords = np.array(np.where(pred_surface)).T * spacing
        target_coords = np.array(np.where(target_surface)).T * spacing
        
        # Compute distances
        from scipy.spatial.distance import cdist
        distances = cdist(pred_coords, target_coords)
        
        hd = max(distances.min(axis=1).max(), distances.min(axis=0).max())
        
        return hd
    
    def _average_surface_distance(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        spacing: np.ndarray
    ) -> float:
        """Compute Average Surface Distance (ASSD) in mm."""
        pred_surface = self._get_surface(pred)
        target_surface = self._get_surface(target)
        
        if pred_surface.sum() == 0 or target_surface.sum() == 0:
            return np.inf
        
        pred_coords = np.array(np.where(pred_surface)).T * spacing
        target_coords = np.array(np.where(target_surface)).T * spacing
        
        from scipy.spatial.distance import cdist
        distances = cdist(pred_coords, target_coords)
        
        pred_to_target = distances.min(axis=1).mean()
        target_to_pred = distances.min(axis=0).mean()
        
        return (pred_to_target + target_to_pred) / 2
    
    def _get_surface(self, mask: np.ndarray) -> np.ndarray:
        """Extract surface voxels from a binary mask."""
        eroded = ndimage.binary_erosion(mask)
        surface = mask & ~eroded
        return surface


def train_vertebrae_segmentation(
    csv_path: str,
    output_dir: str,
    fold: int = 0,
    num_folds: int = 5,
    device: str = 'cuda',
    config: Optional[VertebraeSegmentationConfig] = None,
    localization_model_path: Optional[str] = None,
    resume_from: Optional[str] = None
) -> float:
    """
    Train vertebrae segmentation model for one fold.
    
    Args:
        csv_path: Path to dataset CSV file
        output_dir: Output directory for checkpoints and logs
        fold: Fold index (0 to num_folds-1)
        num_folds: Total number of folds
        device: Device to train on
        config: Optional configuration object
        localization_model_path: Path to trained localization model
        resume_from: Optional checkpoint path to resume from
    
    Returns:
        Best validation loss
    """
    trainer = VertebraeSegmentationTrainer(
        output_dir=output_dir,
        csv_path=csv_path,
        fold=fold,
        num_folds=num_folds,
        device=device,
        config=config,
        localization_model_path=localization_model_path,
        resume_from=resume_from
    )
    
    best_metric = trainer.train()
    
    return best_metric


def train_all_folds(
    csv_path: str,
    output_dir: str,
    num_folds: int = 5,
    device: str = 'cuda',
    config: Optional[VertebraeSegmentationConfig] = None,
    localization_model_dir: Optional[str] = None
) -> Dict[int, float]:
    """
    Train vertebrae segmentation model for all folds.
    
    Args:
        csv_path: Path to dataset CSV file
        output_dir: Output directory
        num_folds: Number of cross-validation folds
        device: Device to train on
        config: Optional configuration object
        localization_model_dir: Directory containing localization models for each fold
    
    Returns:
        Dictionary mapping fold index to best validation loss
    """
    results = {}
    
    for fold in range(num_folds):
        print(f"\n{'#'*60}")
        print(f"# Training Fold {fold}/{num_folds-1}")
        print(f"{'#'*60}\n")
        
        localization_model_path = None
        if localization_model_dir:
            localization_model_path = str(Path(localization_model_dir) / f'fold_{fold}' / 'best.pth')
        
        best_metric = train_vertebrae_segmentation(
            csv_path=csv_path,
            output_dir=output_dir,
            fold=fold,
            num_folds=num_folds,
            device=device,
            config=config,
            localization_model_path=localization_model_path
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
    parser = argparse.ArgumentParser(description='Train Vertebrae Segmentation Model')
    parser.add_argument('--csv', type=str, required=True, help='Path to dataset CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--localization_model', type=str, default=None, help='Path to localization model directory')
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
            localization_model_dir=args.localization_model
        )
    else:
        train_vertebrae_segmentation(
            csv_path=args.csv,
            output_dir=args.output,
            fold=args.fold,
            num_folds=args.num_folds,
            device=args.device,
            localization_model_path=args.localization_model,
            resume_from=args.resume
        )
