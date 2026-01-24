"""
Visualization utilities for vertebrae segmentation results.

Includes:
- Learning curves
- Cross-validation fold comparison
- Per-vertebra performance boxplots
- Confusion matrices
- Slice overlays
"""
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


# Vertebra region colors
REGION_COLORS = {
    'cervical': '#3498db',  # Blue
    'thoracic': '#2ecc71',  # Green
    'lumbar': '#e74c3c',    # Red
    'sacral': '#9b59b6'     # Purple
}

VERTEBRA_NAMES = (
    ['C' + str(i) for i in range(1, 8)] +
    ['T' + str(i) for i in range(1, 13)] +
    ['L' + str(i) for i in range(1, 7)] +
    ['S1', 'S2']
)


def get_vertebra_region(name: str) -> str:
    """Get the anatomical region of a vertebra."""
    if name.startswith('C'):
        return 'cervical'
    elif name.startswith('T'):
        return 'thoracic'
    elif name.startswith('L'):
        return 'lumbar'
    else:
        return 'sacral'


def plot_learning_curves(
    history_path: str,
    output_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot training and validation learning curves.
    
    Args:
        history_path: Path to training history JSON file
        output_path: Path to save figure
        show: Whether to display the figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for plotting")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    train_history = history['train']
    val_history = history['val']
    
    epochs = [h['epoch'] for h in train_history]
    train_loss = [h['total'] for h in train_history]
    val_loss = [h['total'] for h in val_history]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax = axes[0]
    ax.plot(epochs, train_loss, 'b-', label='Train', linewidth=2)
    ax.plot(epochs, val_loss, 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Check for additional metrics
    if 'dice' in val_history[0] or 'mld_mm' in val_history[0]:
        ax = axes[1]
        
        if 'dice' in val_history[0]:
            val_metric = [h.get('dice', h.get('dice_mean', 0)) for h in val_history]
            ax.plot(epochs, val_metric, 'g-', label='Val Dice', linewidth=2)
            ax.set_ylabel('Dice Coefficient', fontsize=12)
        elif 'mld_mm' in val_history[0]:
            val_metric = [h['mld_mm'] for h in val_history]
            ax.plot(epochs, val_metric, 'g-', label='Val MLD (mm)', linewidth=2)
            ax.set_ylabel('MLD (mm)', fontsize=12)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_title('Validation Metrics', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved learning curves to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_fold_comparison(
    results: Dict[int, Dict],
    metric_name: str = 'dice_mean',
    output_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot comparison of metrics across cross-validation folds.
    
    Args:
        results: Dict mapping fold index to metrics dictionary
        metric_name: Name of the metric to plot
        output_path: Path to save figure
        show: Whether to display the figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for plotting")
        return
    
    folds = sorted(results.keys())
    values = [results[f].get(metric_name, 0) for f in folds]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(folds, values, color='steelblue', edgecolor='navy', linewidth=1.5)
    
    # Add mean line
    mean_val = np.mean(values)
    ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11)
    
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Cross-Validation Results: {metric_name}', fontsize=14)
    ax.set_xticks(folds)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add std annotation
    std_val = np.std(values)
    ax.text(0.98, 0.02, f'Std: {std_val:.3f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved fold comparison to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_per_vertebra_boxplot(
    per_vertebra_metrics: Dict[str, List[float]],
    metric_name: str = 'Dice',
    output_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Create boxplot of per-vertebra performance.
    
    Args:
        per_vertebra_metrics: Dict mapping vertebra name to list of metric values
        metric_name: Name of the metric for labels
        output_path: Path to save figure
        show: Whether to display the figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for plotting")
        return
    
    # Order vertebrae
    ordered_names = [n for n in VERTEBRA_NAMES if n in per_vertebra_metrics]
    data = [per_vertebra_metrics[n] for n in ordered_names]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create boxplot
    bp = ax.boxplot(data, patch_artist=True, labels=ordered_names)
    
    # Color by region
    colors = [REGION_COLORS[get_vertebra_region(n)] for n in ordered_names]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Style
    ax.set_xlabel('Vertebra', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'Per-Vertebra {metric_name} Distribution', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    legend_patches = [
        mpatches.Patch(color=REGION_COLORS['cervical'], label='Cervical'),
        mpatches.Patch(color=REGION_COLORS['thoracic'], label='Thoracic'),
        mpatches.Patch(color=REGION_COLORS['lumbar'], label='Lumbar'),
        mpatches.Patch(color=REGION_COLORS['sacral'], label='Sacral')
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=10)
    
    # Add mean line
    overall_mean = np.mean([v for vals in data for v in vals])
    ax.axhline(y=overall_mean, color='red', linestyle='--', 
               linewidth=2, label=f'Overall Mean: {overall_mean:.3f}')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved boxplot to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_per_region_summary(
    per_vertebra_metrics: Dict[str, float],
    metric_name: str = 'Dice',
    output_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot summary by anatomical region.
    
    Args:
        per_vertebra_metrics: Dict mapping vertebra name to metric value
        metric_name: Name of the metric
        output_path: Path to save figure
        show: Whether to display
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # Group by region
    region_values = {'cervical': [], 'thoracic': [], 'lumbar': [], 'sacral': []}
    
    for name, value in per_vertebra_metrics.items():
        region = get_vertebra_region(name)
        region_values[region].append(value)
    
    # Compute statistics
    regions = ['cervical', 'thoracic', 'lumbar', 'sacral']
    means = [np.mean(region_values[r]) if region_values[r] else 0 for r in regions]
    stds = [np.std(region_values[r]) if region_values[r] else 0 for r in regions]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(regions))
    bars = ax.bar(x, means, yerr=stds, capsize=5, 
                  color=[REGION_COLORS[r] for r in regions],
                  edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Region', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} by Anatomical Region', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in regions])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add count annotations
    for i, (bar, region) in enumerate(zip(bars, regions)):
        count = len(region_values[region])
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.02,
                f'n={count}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    predictions: List[int],
    targets: List[int],
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot confusion matrix for vertebra identification.
    
    Args:
        predictions: Predicted vertebra labels
        targets: Ground truth vertebra labels
        class_names: Names for each class
        output_path: Path to save figure
        show: Whether to display
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    from sklearn.metrics import confusion_matrix
    
    if class_names is None:
        class_names = VERTEBRA_NAMES
    
    cm = confusion_matrix(targets, predictions)
    
    fig, ax = plt.subplots(figsize=(20, 18))
    
    if SEABORN_AVAILABLE:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
    else:
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        plt.colorbar(im)
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Ground Truth', fontsize=12)
    ax.set_title('Vertebra Identification Confusion Matrix', fontsize=14)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def save_slice_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: str,
    slice_axis: int = 0,
    slice_index: Optional[int] = None,
    alpha: float = 0.5,
    title: Optional[str] = None
) -> None:
    """
    Save overlay of segmentation mask on image slice.
    
    Args:
        image: 3D image array
        mask: 3D mask array (multi-label)
        output_path: Path to save figure
        slice_axis: Axis to slice along (0=axial, 1=coronal, 2=sagittal)
        slice_index: Index of slice (default: middle)
        alpha: Transparency of overlay
        title: Figure title
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    if slice_index is None:
        slice_index = image.shape[slice_axis] // 2
    
    # Extract slice
    if slice_axis == 0:
        img_slice = image[slice_index, :, :]
        mask_slice = mask[slice_index, :, :]
    elif slice_axis == 1:
        img_slice = image[:, slice_index, :]
        mask_slice = mask[:, slice_index, :]
    else:
        img_slice = image[:, :, slice_index]
        mask_slice = mask[:, :, slice_index]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title('CT Image')
    axes[0].axis('off')
    
    # Mask only
    cmap = plt.cm.get_cmap('tab20', len(np.unique(mask_slice)))
    axes[1].imshow(mask_slice, cmap=cmap)
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(img_slice, cmap='gray')
    masked = np.ma.masked_where(mask_slice == 0, mask_slice)
    axes[2].imshow(masked, cmap=cmap, alpha=alpha)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_evaluation_report(
    results: Dict[str, Any],
    output_dir: str,
    experiment_name: str = 'evaluation'
) -> None:
    """
    Create comprehensive evaluation report with all visualizations.
    
    Args:
        results: Complete evaluation results dictionary
        output_dir: Directory to save report
        experiment_name: Name prefix for saved files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Overall metrics table
    if 'overall' in results:
        metrics_path = output_path / f'{experiment_name}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(results['overall'], f, indent=2)
    
    # Per-vertebra boxplot
    if 'per_vertebra_dice' in results:
        # Convert means to lists for boxplot (if needed)
        per_vert = results['per_vertebra_dice']
        if isinstance(list(per_vert.values())[0], (int, float)):
            per_vert = {k: [v] for k, v in per_vert.items()}
        
        plot_per_vertebra_boxplot(
            per_vert,
            metric_name='Dice',
            output_path=str(output_path / f'{experiment_name}_dice_boxplot.png'),
            show=False
        )
    
    if 'per_vertebra_mld' in results:
        per_vert = results['per_vertebra_mld']
        if isinstance(list(per_vert.values())[0], (int, float)):
            per_vert = {k: [v] for k, v in per_vert.items()}
        
        plot_per_vertebra_boxplot(
            per_vert,
            metric_name='MLD (mm)',
            output_path=str(output_path / f'{experiment_name}_mld_boxplot.png'),
            show=False
        )
    
    # Region summary
    if 'per_vertebra_dice' in results:
        plot_per_region_summary(
            results['per_vertebra_dice'],
            metric_name='Dice',
            output_path=str(output_path / f'{experiment_name}_region_dice.png'),
            show=False
        )
    
    print(f"Evaluation report saved to {output_path}")
