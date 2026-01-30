#!/usr/bin/env python
"""
Visualize training metrics from history.json file.
Automatically plots loss curves and metrics after training.

Usage:
    python visualize_metrics.py --history /path/to/history.json
    python visualize_metrics.py --history outputs/spine_localization/logs/fold_0/history.json
"""

import json
import argparse
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


def load_history(history_path):
    """Load training history from JSON file."""
    with open(history_path, 'r') as f:
        return json.load(f)


def plot_learning_curves(history, output_dir=None):
    """Plot training and validation learning curves."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping plot: matplotlib not available")
        return
    
    train_data = history['train']
    val_data = history['val']
    
    epochs = [h['epoch'] for h in train_data]
    
    # Extract all metrics
    train_metrics = {}
    val_metrics = {}
    
    for h in train_data:
        for key, value in h.items():
            if key != 'epoch' and isinstance(value, (int, float)):
                if key not in train_metrics:
                    train_metrics[key] = []
                train_metrics[key].append(value)
    
    for h in val_data:
        for key, value in h.items():
            if key != 'epoch' and isinstance(value, (int, float)):
                if key not in val_metrics:
                    val_metrics[key] = []
                val_metrics[key].append(value)
    
    # Plot loss curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
    
    # Loss
    ax = axes[0, 0]
    if 'total' in train_metrics and 'total' in val_metrics:
        ax.plot(epochs, train_metrics['total'], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, val_metrics['total'], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # MLD (Mean Localization Distance)
    ax = axes[0, 1]
    if 'mld_mm' in val_metrics:
        ax.plot(epochs, val_metrics['mld_mm'], 'g-', linewidth=2, marker='o', markersize=4)
        ax.fill_between(epochs, 
                        np.array(val_metrics['mld_mm']) - np.array(val_metrics.get('mld_std_mm', [0]*len(epochs))),
                        np.array(val_metrics['mld_mm']) + np.array(val_metrics.get('mld_std_mm', [0]*len(epochs))),
                        alpha=0.2, color='g')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MLD (mm)')
        ax.set_title('Mean Localization Distance')
        ax.grid(True, alpha=0.3)
    
    # Success Rate 20mm
    ax = axes[1, 0]
    if 'success_rate_20mm' in val_metrics:
        ax.plot(epochs, val_metrics['success_rate_20mm'], 'purple', linewidth=2, marker='s', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate @ 20mm')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)
    
    # Success Rate 10mm
    ax = axes[1, 1]
    if 'success_rate_10mm' in val_metrics:
        ax.plot(epochs, val_metrics['success_rate_10mm'], 'orange', linewidth=2, marker='^', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate @ 10mm')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'learning_curves.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot: {output_path}")
    
    plt.show()


def print_summary(history):
    """Print summary of training results."""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    print(f"\nBest Epoch: {history['best_epoch']}")
    print(f"Best Metric: {history['best_metric']:.6f}")
    
    print(f"\nMetrics tracked:")
    if history['val']:
        first_val = history['val'][0]
        for key in sorted(first_val.keys()):
            if key != 'epoch':
                print(f"  ✓ {key}")
    
    print(f"\nFinal validation metrics:")
    if history['val']:
        last_val = history['val'][-1]
        for key, value in sorted(last_val.items()):
            if key != 'epoch' and isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize training metrics')
    parser.add_argument('--history', type=str, required=True, 
                        help='Path to history.json file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for plots (defaults to history directory)')
    
    args = parser.parse_args()
    
    history_path = Path(args.history)
    if not history_path.exists():
        print(f"Error: {history_path} not found")
        return
    
    # Load history
    history = load_history(history_path)
    
    # Print summary
    print_summary(history)
    
    # Plot
    output_dir = args.output or history_path.parent
    plot_learning_curves(history, output_dir)


if __name__ == '__main__':
    main()
