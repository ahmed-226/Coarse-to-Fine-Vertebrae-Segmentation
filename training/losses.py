"""
Loss functions for vertebrae segmentation pipeline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HeatmapMSELoss(nn.Module):
    """
    Mean Squared Error loss for heatmap regression.
    Used in Stage 1 (spine localization) and Stage 2 (vertebrae localization).
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted heatmaps [B, C, D, H, W]
            target: Target heatmaps [B, C, D, H, W]
            mask: Optional mask for valid landmarks [B, C]
        """
        loss = (pred - target) ** 2
        
        if mask is not None:
            # Expand mask to spatial dimensions
            while mask.dim() < loss.dim():
                mask = mask.unsqueeze(-1)
            loss = loss * mask
            
            # Average over valid elements only
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() * loss.shape[2] * loss.shape[3] * loss.shape[4] + 1e-8)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        
        return loss


class HeatmapBCELoss(nn.Module):
    """
    Binary Cross-Entropy loss for heatmap prediction.
    Used in Stage 2 with sigmoid cross-entropy.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits (before sigmoid) [B, C, D, H, W]
            target: Target heatmaps [B, C, D, H, W]
            mask: Optional mask for valid landmarks [B, C]
        """
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        if mask is not None:
            while mask.dim() < loss.dim():
                mask = mask.unsqueeze(-1)
            loss = loss * mask
            
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() * loss.shape[2] * loss.shape[3] * loss.shape[4] + 1e-8)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        
        return loss


class SigmaRegularizationLoss(nn.Module):
    """
    Regularization loss for learnable sigma in SCNet.
    Encourages sigma values to stay within reasonable range.
    """
    
    def __init__(self, target_sigma: float = 4.0, weight: float = 100.0):
        super().__init__()
        self.target_sigma = target_sigma
        self.weight = weight
    
    def forward(
        self,
        sigma: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            sigma: Learnable sigma values [N, 3] or [N]
            valid_mask: Mask for valid landmarks [N]
        """
        # Scale sigma to actual values
        sigma_scaled = sigma * 1000.0  # Assuming sigma stored as sigma/1000
        
        # Regularize towards target
        loss = (sigma_scaled - self.target_sigma) ** 2
        
        if valid_mask is not None:
            if valid_mask.dim() < loss.dim():
                valid_mask = valid_mask.unsqueeze(-1)
            loss = loss * valid_mask
            return self.weight * loss.sum() / (valid_mask.sum() + 1e-8)
        
        return self.weight * loss.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    Used in Stage 3 (vertebrae segmentation).
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities [B, 1, D, H, W] (after sigmoid)
            target: Binary target [B, 1, D, H, W]
        """
        pred = torch.sigmoid(pred) if pred.min() < 0 else pred
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE Loss.
    Often provides better gradient flow than Dice alone.
    """
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        return self.dice_weight * dice + self.bce_weight * bce


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Useful when vertebra occupies small fraction of volume.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits [B, 1, D, H, W]
            target: Binary target [B, 1, D, H, W]
        """
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        loss = focal_weight * bce
        
        return loss.mean()


class CombinedLocalizationLoss(nn.Module):
    """
    Combined loss for vertebrae localization (Stage 2).
    
    Combines:
    - Heatmap loss (MSE or BCE)
    - Sigma regularization loss (if learnable sigma)
    """
    
    def __init__(
        self,
        heatmap_loss_type: str = 'mse',
        sigma_regularization: float = 100.0,
        target_sigma: float = 4.0
    ):
        super().__init__()
        
        if heatmap_loss_type == 'mse':
            self.heatmap_loss = HeatmapMSELoss()
        else:
            self.heatmap_loss = HeatmapBCELoss()
        
        self.sigma_loss = SigmaRegularizationLoss(
            target_sigma=target_sigma,
            weight=sigma_regularization
        )
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            pred: Predicted heatmaps [B, N, D, H, W]
            target: Target heatmaps [B, N, D, H, W]
            sigma: Learnable sigma values [N, 3]
            valid_mask: Mask for valid landmarks [B, N]
        
        Returns:
            Dictionary with loss components
        """
        losses = {}
        
        # Heatmap loss
        losses['heatmap'] = self.heatmap_loss(pred, target, valid_mask)
        
        # Sigma regularization
        if sigma is not None:
            # Average valid_mask across batch: [B, N] -> [N]
            batch_valid_mask = valid_mask.mean(dim=0) if valid_mask is not None else None
            losses['sigma'] = self.sigma_loss(sigma, batch_valid_mask)
        else:
            losses['sigma'] = torch.tensor(0.0, device=pred.device)
        
        # Total loss
        losses['total'] = losses['heatmap'] + losses['sigma']
        
        return losses
