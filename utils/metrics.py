"""
Comprehensive metrics for vertebrae segmentation evaluation.

Includes metrics from VerSe Challenge:
- Dice coefficient
- Hausdorff Distance (HD)
- 95th percentile Hausdorff Distance (HD95)
- Average Symmetric Surface Distance (ASSD)
- Mean Localization Distance (MLD)
- Identification Rate
"""
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
from dataclasses import dataclass, field


@dataclass
class SegmentationResult:
    """Container for segmentation evaluation results."""
    dice: float = 0.0
    iou: float = 0.0
    hausdorff: float = float('inf')
    hd95: float = float('inf')
    assd: float = float('inf')
    volume_pred: float = 0.0
    volume_gt: float = 0.0
    volume_error: float = 0.0


@dataclass
class LocalizationResult:
    """Container for localization evaluation results."""
    mld: float = 0.0
    x_error: float = 0.0
    y_error: float = 0.0
    z_error: float = 0.0
    identified: bool = False


@dataclass
class PerVertebraResults:
    """Container for per-vertebra results."""
    name: str
    segmentation: Optional[SegmentationResult] = None
    localization: Optional[LocalizationResult] = None


def compute_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Dice coefficient between two binary masks.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
    
    Returns:
        Dice coefficient in [0, 1]
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    intersection = np.logical_and(pred, target).sum()
    total = pred.sum() + target.sum()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return 2.0 * intersection / total


def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU / Jaccard).
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
    
    Returns:
        IoU in [0, 1]
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def get_surface_points(mask: np.ndarray) -> np.ndarray:
    """
    Extract surface voxel coordinates from a binary mask.
    
    Args:
        mask: Binary 3D mask
    
    Returns:
        Array of surface point coordinates [N, 3]
    """
    mask = mask.astype(bool)
    
    if mask.sum() == 0:
        return np.zeros((0, 3))
    
    # Erode and find surface
    eroded = ndimage.binary_erosion(mask)
    surface = mask & ~eroded
    
    # Get coordinates
    coords = np.array(np.where(surface)).T
    
    return coords


def compute_hausdorff(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
) -> float:
    """
    Compute Hausdorff Distance between two binary masks.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        spacing: Voxel spacing (z, y, x) in mm
    
    Returns:
        Hausdorff distance in mm
    """
    pred_surface = get_surface_points(pred)
    target_surface = get_surface_points(target)
    
    if len(pred_surface) == 0 or len(target_surface) == 0:
        return float('inf')
    
    # Scale by spacing
    pred_surface = pred_surface * np.array(spacing)
    target_surface = target_surface * np.array(spacing)
    
    # Compute pairwise distances
    distances = cdist(pred_surface, target_surface)
    
    # Hausdorff is max of directed distances
    hd_pred_to_target = distances.min(axis=1).max()
    hd_target_to_pred = distances.min(axis=0).max()
    
    return max(hd_pred_to_target, hd_target_to_pred)


def compute_hd95(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
) -> float:
    """
    Compute 95th percentile Hausdorff Distance.
    
    More robust to outliers than full Hausdorff distance.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        spacing: Voxel spacing in mm
    
    Returns:
        HD95 in mm
    """
    pred_surface = get_surface_points(pred)
    target_surface = get_surface_points(target)
    
    if len(pred_surface) == 0 or len(target_surface) == 0:
        return float('inf')
    
    # Scale by spacing
    pred_surface = pred_surface * np.array(spacing)
    target_surface = target_surface * np.array(spacing)
    
    # Compute pairwise distances
    distances = cdist(pred_surface, target_surface)
    
    # Directed distances
    pred_to_target = distances.min(axis=1)
    target_to_pred = distances.min(axis=0)
    
    # Combine and take 95th percentile
    all_distances = np.concatenate([pred_to_target, target_to_pred])
    
    return np.percentile(all_distances, 95)


def compute_assd(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
) -> float:
    """
    Compute Average Symmetric Surface Distance (ASSD).
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        spacing: Voxel spacing in mm
    
    Returns:
        ASSD in mm
    """
    pred_surface = get_surface_points(pred)
    target_surface = get_surface_points(target)
    
    if len(pred_surface) == 0 or len(target_surface) == 0:
        return float('inf')
    
    # Scale by spacing
    pred_surface = pred_surface * np.array(spacing)
    target_surface = target_surface * np.array(spacing)
    
    # Compute pairwise distances
    distances = cdist(pred_surface, target_surface)
    
    # Mean of directed distances
    pred_to_target = distances.min(axis=1).mean()
    target_to_pred = distances.min(axis=0).mean()
    
    return (pred_to_target + target_to_pred) / 2


def compute_mld(
    pred_centroid: np.ndarray,
    target_centroid: np.ndarray,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
) -> float:
    """
    Compute Mean Localization Distance between centroids.
    
    Args:
        pred_centroid: Predicted centroid coordinates [z, y, x] or [x, y, z]
        target_centroid: Ground truth centroid coordinates
        spacing: Voxel spacing in mm
    
    Returns:
        Euclidean distance in mm
    """
    diff = (pred_centroid - target_centroid) * np.array(spacing)
    return np.sqrt(np.sum(diff ** 2))


class SegmentationMetrics:
    """
    Comprehensive segmentation metrics calculator.
    """
    
    def __init__(self, spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)):
        """
        Args:
            spacing: Voxel spacing (z, y, x) in mm
        """
        self.spacing = spacing
    
    def compute_all(
        self,
        pred: np.ndarray,
        target: np.ndarray
    ) -> SegmentationResult:
        """
        Compute all segmentation metrics.
        
        Args:
            pred: Predicted binary mask
            target: Ground truth binary mask
        
        Returns:
            SegmentationResult with all metrics
        """
        result = SegmentationResult()
        
        # Basic metrics
        result.dice = compute_dice(pred, target)
        result.iou = compute_iou(pred, target)
        
        # Distance-based metrics (only if both masks non-empty)
        if pred.sum() > 0 and target.sum() > 0:
            result.hausdorff = compute_hausdorff(pred, target, self.spacing)
            result.hd95 = compute_hd95(pred, target, self.spacing)
            result.assd = compute_assd(pred, target, self.spacing)
        
        # Volume metrics
        voxel_volume = np.prod(self.spacing)
        result.volume_pred = pred.sum() * voxel_volume
        result.volume_gt = target.sum() * voxel_volume
        
        if result.volume_gt > 0:
            result.volume_error = abs(result.volume_pred - result.volume_gt) / result.volume_gt * 100
        
        return result
    
    def compute_batch(
        self,
        predictions: List[np.ndarray],
        targets: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute aggregated metrics over a batch.
        
        Returns:
            Dictionary with mean and std of all metrics
        """
        results = [self.compute_all(p, t) for p, t in zip(predictions, targets)]
        
        dice_vals = [r.dice for r in results]
        iou_vals = [r.iou for r in results]
        hd_vals = [r.hausdorff for r in results if r.hausdorff != float('inf')]
        hd95_vals = [r.hd95 for r in results if r.hd95 != float('inf')]
        assd_vals = [r.assd for r in results if r.assd != float('inf')]
        
        return {
            'dice_mean': np.mean(dice_vals),
            'dice_std': np.std(dice_vals),
            'dice_median': np.median(dice_vals),
            'iou_mean': np.mean(iou_vals),
            'hd_mean': np.mean(hd_vals) if hd_vals else float('inf'),
            'hd95_mean': np.mean(hd95_vals) if hd95_vals else float('inf'),
            'assd_mean': np.mean(assd_vals) if assd_vals else float('inf'),
        }


class LocalizationMetrics:
    """
    Localization metrics calculator for vertebra centroid detection.
    """
    
    def __init__(
        self,
        spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
        identification_radius: float = 20.0
    ):
        """
        Args:
            spacing: Voxel spacing in mm
            identification_radius: Threshold for successful identification (mm)
        """
        self.spacing = spacing
        self.identification_radius = identification_radius
    
    def compute(
        self,
        pred_centroid: np.ndarray,
        target_centroid: np.ndarray
    ) -> LocalizationResult:
        """
        Compute localization metrics for a single vertebra.
        
        Args:
            pred_centroid: Predicted centroid [z, y, x]
            target_centroid: Ground truth centroid [z, y, x]
        
        Returns:
            LocalizationResult
        """
        result = LocalizationResult()
        
        # Per-axis errors
        diff = (pred_centroid - target_centroid) * np.array(self.spacing)
        result.z_error = abs(diff[0])
        result.y_error = abs(diff[1])
        result.x_error = abs(diff[2])
        
        # Total distance
        result.mld = np.sqrt(np.sum(diff ** 2))
        
        # Identification success
        result.identified = result.mld < self.identification_radius
        
        return result
    
    def compute_batch(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute aggregated localization metrics.
        
        Args:
            predictions: Dict mapping vertebra name to predicted centroid
            targets: Dict mapping vertebra name to ground truth centroid
        
        Returns:
            Dictionary with aggregated metrics
        """
        all_mld = []
        identified = 0
        total = 0
        
        per_vertebra = {}
        
        for name in targets:
            if name not in predictions:
                continue
            
            result = self.compute(predictions[name], targets[name])
            all_mld.append(result.mld)
            identified += int(result.identified)
            total += 1
            per_vertebra[name] = result.mld
        
        return {
            'mld_mean': np.mean(all_mld) if all_mld else 0.0,
            'mld_std': np.std(all_mld) if all_mld else 0.0,
            'mld_median': np.median(all_mld) if all_mld else 0.0,
            'identification_rate': identified / total * 100 if total > 0 else 0.0,
            'num_detected': len(predictions),
            'num_ground_truth': len(targets),
            'per_vertebra': per_vertebra
        }


class VerSeMetrics:
    """
    Comprehensive metrics following the VerSe Challenge evaluation protocol.
    
    Combines segmentation and localization metrics with proper weighting.
    """
    
    VERTEBRA_NAMES = (
        ['C' + str(i) for i in range(1, 8)] +
        ['T' + str(i) for i in range(1, 13)] +
        ['L' + str(i) for i in range(1, 7)] +
        ['S1', 'S2']
    )
    
    def __init__(self, spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)):
        self.spacing = spacing
        self.seg_metrics = SegmentationMetrics(spacing)
        self.loc_metrics = LocalizationMetrics(spacing)
    
    def evaluate_case(
        self,
        pred_seg: np.ndarray,
        gt_seg: np.ndarray,
        pred_landmarks: Optional[Dict[str, np.ndarray]] = None,
        gt_landmarks: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single case following VerSe protocol.
        
        Args:
            pred_seg: Predicted multi-label segmentation
            gt_seg: Ground truth multi-label segmentation
            pred_landmarks: Predicted centroids (optional)
            gt_landmarks: Ground truth centroids (optional)
        
        Returns:
            Dictionary with per-vertebra and aggregate metrics
        """
        results = {
            'per_vertebra': {},
            'aggregate': {}
        }
        
        # Get unique labels (excluding background)
        gt_labels = np.unique(gt_seg)
        gt_labels = gt_labels[gt_labels > 0]
        
        dice_vals = []
        hd95_vals = []
        assd_vals = []
        mld_vals = []
        
        for label in gt_labels:
            if label > len(self.VERTEBRA_NAMES):
                continue
            
            name = self.VERTEBRA_NAMES[int(label) - 1]
            
            # Extract binary masks
            pred_mask = (pred_seg == label)
            gt_mask = (gt_seg == label)
            
            # Segmentation metrics
            if gt_mask.sum() > 0:
                seg_result = self.seg_metrics.compute_all(pred_mask, gt_mask)
                dice_vals.append(seg_result.dice)
                
                if seg_result.hd95 != float('inf'):
                    hd95_vals.append(seg_result.hd95)
                if seg_result.assd != float('inf'):
                    assd_vals.append(seg_result.assd)
                
                results['per_vertebra'][name] = {
                    'dice': seg_result.dice,
                    'hd95': seg_result.hd95,
                    'assd': seg_result.assd
                }
            
            # Localization metrics
            if pred_landmarks and gt_landmarks and name in gt_landmarks:
                if name in pred_landmarks:
                    loc_result = self.loc_metrics.compute(
                        pred_landmarks[name],
                        gt_landmarks[name]
                    )
                    mld_vals.append(loc_result.mld)
                    results['per_vertebra'][name]['mld'] = loc_result.mld
                    results['per_vertebra'][name]['identified'] = loc_result.identified
        
        # Aggregate metrics
        results['aggregate'] = {
            'dice_mean': np.mean(dice_vals) if dice_vals else 0.0,
            'dice_std': np.std(dice_vals) if dice_vals else 0.0,
            'hd95_mean': np.mean(hd95_vals) if hd95_vals else float('inf'),
            'assd_mean': np.mean(assd_vals) if assd_vals else float('inf'),
            'num_vertebrae': len(gt_labels)
        }
        
        if mld_vals:
            results['aggregate']['mld_mean'] = np.mean(mld_vals)
            results['aggregate']['id_rate'] = np.mean([m < 20 for m in mld_vals]) * 100
        
        return results
    
    def evaluate_dataset(
        self,
        cases: List[Dict]
    ) -> Dict[str, Any]:
        """
        Evaluate entire dataset.
        
        Args:
            cases: List of case dictionaries with predictions and ground truth
        
        Returns:
            Aggregated dataset metrics
        """
        all_dice = []
        all_hd95 = []
        all_mld = []
        
        per_vertebra_dice = {name: [] for name in self.VERTEBRA_NAMES}
        per_vertebra_mld = {name: [] for name in self.VERTEBRA_NAMES}
        
        for case in cases:
            result = self.evaluate_case(
                case['pred_seg'],
                case['gt_seg'],
                case.get('pred_landmarks'),
                case.get('gt_landmarks')
            )
            
            for name, metrics in result['per_vertebra'].items():
                if 'dice' in metrics:
                    all_dice.append(metrics['dice'])
                    per_vertebra_dice[name].append(metrics['dice'])
                if 'hd95' in metrics and metrics['hd95'] != float('inf'):
                    all_hd95.append(metrics['hd95'])
                if 'mld' in metrics:
                    all_mld.append(metrics['mld'])
                    per_vertebra_mld[name].append(metrics['mld'])
        
        return {
            'overall': {
                'dice_mean': np.mean(all_dice) if all_dice else 0.0,
                'dice_std': np.std(all_dice) if all_dice else 0.0,
                'hd95_mean': np.mean(all_hd95) if all_hd95 else float('inf'),
                'mld_mean': np.mean(all_mld) if all_mld else 0.0,
                'id_rate_20mm': np.mean([m < 20 for m in all_mld]) * 100 if all_mld else 0.0,
                'id_rate_4mm': np.mean([m < 4 for m in all_mld]) * 100 if all_mld else 0.0,
                'num_vertebrae': len(all_dice)
            },
            'per_vertebra_dice': {k: np.mean(v) for k, v in per_vertebra_dice.items() if v},
            'per_vertebra_mld': {k: np.mean(v) for k, v in per_vertebra_mld.items() if v}
        }
