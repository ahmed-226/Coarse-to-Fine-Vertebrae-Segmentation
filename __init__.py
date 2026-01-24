"""
PyTorch Pipeline for Vertebrae Segmentation

A complete reimplementation of the TensorFlow vertebrae segmentation pipeline
using PyTorch, following the 3-stage approach:

1. Spine Localization: U-Net predicts spine centroid for cropping
2. Vertebrae Localization: SCNet predicts individual vertebra centroids
3. Vertebrae Segmentation: U-Net segments each vertebra

This implementation avoids MedicalDataAugmentationTool dependencies and
uses pure PyTorch with SimpleITK for image I/O.

Usage:
    # Training
    python -m pytorch_pipeline.train --stage spine --csv data.csv --output outputs/

    # Inference
    python -m pytorch_pipeline.infer --image ct.nii.gz --model_dir outputs/ --output results/

Author: Reimplemented from TensorFlow 1.x version
"""

__version__ = "1.0.0"

# Core components
from .config import (
    SpineLocalizationConfig,
    VertebraeLocalizationConfig,
    VertebraeSegmentationConfig
)

from .models import (
    UNet3D,
    UNet3DSpineLocalization,
    UNet3DVertebraeSegmentation,
    SpatialConfigurationNet
)

from .data import (
    SpineLocalizationDataset,
    VertebraeLocalizationDataset,
    VertebraeSegmentationDataset,
    get_train_transforms,
    get_val_transforms
)

from .training import (
    SpineLocalizationTrainer,
    VertebraeLocalizationTrainer,
    VertebraeSegmentationTrainer
)

from .inference import (
    SpineLocalizationPredictor,
    VertebraeLocalizationPredictor,
    VertebraeSegmentationPredictor,
    FullPipelinePredictor
)

from .utils import (
    SegmentationMetrics,
    LocalizationMetrics,
    VerSeMetrics,
    plot_learning_curves,
    plot_fold_comparison,
    plot_per_vertebra_boxplot
)

__all__ = [
    # Config
    'SpineLocalizationConfig',
    'VertebraeLocalizationConfig', 
    'VertebraeSegmentationConfig',
    
    # Models
    'UNet3D',
    'UNet3DSpineLocalization',
    'UNet3DVertebraeSegmentation',
    'SpatialConfigurationNet',
    
    # Data
    'SpineLocalizationDataset',
    'VertebraeLocalizationDataset',
    'VertebraeSegmentationDataset',
    'get_train_transforms',
    'get_val_transforms',
    
    # Training
    'SpineLocalizationTrainer',
    'VertebraeLocalizationTrainer',
    'VertebraeSegmentationTrainer',
    
    # Inference
    'SpineLocalizationPredictor',
    'VertebraeLocalizationPredictor',
    'VertebraeSegmentationPredictor',
    'FullPipelinePredictor',
    
    # Utils
    'SegmentationMetrics',
    'LocalizationMetrics',
    'VerSeMetrics',
    'plot_learning_curves',
    'plot_fold_comparison',
    'plot_per_vertebra_boxplot'
]
