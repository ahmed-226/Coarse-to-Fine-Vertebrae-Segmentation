# Training modules
from .trainer import BaseTrainer
from .train_spine_localization import (
    SpineLocalizationTrainer,
    train_spine_localization,
    train_all_folds as train_all_folds_spine
)
from .train_vertebrae_localization import (
    VertebraeLocalizationTrainer,
    train_vertebrae_localization,
    train_all_folds as train_all_folds_vertebrae
)
from .train_vertebrae_segmentation import (
    VertebraeSegmentationTrainer,
    train_vertebrae_segmentation,
    train_all_folds as train_all_folds_segmentation
)
from .losses import HeatmapMSELoss, DiceLoss, DiceBCELoss
