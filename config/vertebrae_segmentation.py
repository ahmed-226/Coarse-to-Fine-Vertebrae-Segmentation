"""
Stage 3: Vertebrae Segmentation Configuration
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class VertebraeSegmentationConfig:
    """Configuration for vertebrae segmentation (Stage 3)."""
    
    # Training parameters
    batch_size: int = 1
    max_epochs: int = 300
    max_iter: int = 50000
    learning_rate: float = 1e-4
    learning_rate_milestones: List[int] = field(default_factory=lambda: [20000, 30000])
    learning_rate_gamma: float = 0.5
    weight_decay: float = 1e-6
    clip_gradient_norm: float = 1.0
    
    # Image parameters
    image_size: Tuple[int, int, int] = (128, 128, 96)  # [X, Y, Z]
    image_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # mm
    
    # Network parameters
    num_labels: int = 1  # Binary segmentation per vertebra
    num_labels_all: int = 26  # Total vertebra classes
    num_filters_base: int = 64
    num_levels: int = 5
    dropout_ratio: float = 0.5
    
    # Input channels: image + heatmap prior
    input_channels: int = 2
    
    # Heatmap parameters
    input_gaussian_sigma: float = 0.75
    label_gaussian_sigma: float = 1.0
    heatmap_sigma: float = 3.0
    
    # Data augmentation
    augmentation_flip_prob: float = 0.5
    augmentation_rotation_range: float = 30.0  # degrees (larger than localization)
    augmentation_scale_range: Tuple[float, float] = (0.85, 1.15)
    augmentation_translation_range: float = 0.2
    
    # Intensity preprocessing
    intensity_min: float = -1024.0
    intensity_max: float = 4096.0
    
    # Output
    output_folder: str = "output/vertebrae_segmentation"
    
    # Validation
    test_iter: int = 5000
    snapshot_iter: int = 5000
    
    # Cross-validation
    n_folds: int = 5
    
    # Post-processing
    min_component_size: int = 1000  # Minimum connected component size
