"""
Stage 1: Spine Localization Configuration
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SpineLocalizationConfig:
    """Configuration for spine localization (Stage 1)."""
    
    # Training parameters
    batch_size: int = 1
    max_epochs: int = 200
    max_iter: int = 20000
    learning_rate: float = 1e-4
    learning_rate_milestones: List[int] = field(default_factory=lambda: [10000, 15000])
    learning_rate_gamma: float = 0.5
    weight_decay: float = 5e-4
    
    # Image parameters
    image_size: Tuple[int, int, int] = (64, 64, 128)  # [X, Y, Z]
    image_spacing: Tuple[float, float, float] = (8.0, 8.0, 8.0)  # mm
    
    # Network parameters
    num_labels: int = 1  # Single heatmap output
    num_filters_base: int = 64
    num_levels: int = 5
    dropout_ratio: float = 0.5
    
    # Heatmap parameters
    input_gaussian_sigma: float = 3.0
    heatmap_sigma: float = 2.0
    
    # Data augmentation
    augmentation_flip_prob: float = 0.5
    augmentation_rotation_range: float = 15.0  # degrees
    augmentation_scale_range: Tuple[float, float] = (0.85, 1.15)
    augmentation_translation_range: float = 0.2  # fraction of image size
    
    # Intensity preprocessing
    intensity_min: float = -1024.0  # HU
    intensity_max: float = 4096.0   # HU
    
    # Output
    output_folder: str = "output/spine_localization"
    
    # Validation
    test_iter: int = 5000
    snapshot_iter: int = 5000
    
    # Cross-validation
    n_folds: int = 5
