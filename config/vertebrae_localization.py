"""
Stage 2: Vertebrae Localization Configuration
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class VertebraeLocalizationConfig:
    """Configuration for vertebrae localization (Stage 2)."""
    
    # Training parameters
    batch_size: int = 1
    max_epochs: int = 500
    max_iter: int = 100000
    learning_rate: float = 1e-8
    learning_rate_milestones: List[int] = field(default_factory=lambda: [50000, 75000])
    learning_rate_gamma: float = 0.5
    weight_decay: float = 5e-4
    clip_gradient_norm: float = 100000.0
    
    # Image parameters
    image_size: Tuple[int, int, int] = (96, 96, 128)  # [X, Y, Z]
    image_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # mm
    
    # Network parameters
    num_landmarks: int = 26  # C1-L5 + S1 (or 25 for VerSe2019)
    num_filters_base: int = 64
    num_levels: int = 5
    dropout_ratio: float = 0.5
    spatial_downsample: int = 4
    
    # Heatmap parameters
    input_gaussian_sigma: float = 0.75
    heatmap_sigma: float = 4.0
    learnable_sigma: bool = True
    sigma_regularization: float = 100.0
    sigma_scale: float = 1000.0
    
    # Data augmentation
    augmentation_flip_prob: float = 0.5
    augmentation_rotation_range: float = 15.0
    augmentation_scale_range: Tuple[float, float] = (0.85, 1.15)
    augmentation_translation_range: float = 0.2
    
    # Intensity preprocessing
    intensity_min: float = -1024.0
    intensity_max: float = 4096.0
    
    # Output
    output_folder: str = "output/vertebrae_localization"
    
    # Validation
    test_iter: int = 10000
    snapshot_iter: int = 5000
    
    # Cross-validation
    n_folds: int = 5
    
    # Sliding window for testing
    cropped_inc: Tuple[int, int, int, int] = (0, 96, 0, 0)
