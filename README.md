# PyTorch Vertebrae Segmentation Pipeline

A complete PyTorch reimplementation of the 3-stage vertebrae segmentation pipeline, 
originally implemented in TensorFlow 1.x with MedicalDataAugmentationTool.

## Overview

This pipeline segments individual vertebrae from CT images using a 3-stage approach:

1. **Spine Localization**: A 3D U-Net predicts a heatmap centered on the spine, 
   providing a centroid for cropping in subsequent stages.

2. **Vertebrae Localization**: A SpatialConfiguration-Net (SCNet) predicts heatmaps 
   for each individual vertebra centroid, with learnable sigma parameters.

3. **Vertebrae Segmentation**: A 3D U-Net segments each vertebra by cropping 
   around its predicted centroid.

## Installation

### Requirements

```bash
pip install torch torchvision
pip install numpy scipy pandas
pip install SimpleITK
pip install tqdm tensorboard
pip install matplotlib seaborn  # For visualization
pip install scikit-learn  # For metrics
```

### Minimum Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU training)
- 16GB RAM minimum, 32GB recommended
- 8GB VRAM minimum for training

## Project Structure

```
pytorch_pipeline/
├── __init__.py              # Main package exports
├── train.py                 # Training entry point
├── infer.py                 # Inference entry point
├── sample_dataset.csv       # Sample dataset CSV format
│
├── config/                  # Configuration dataclasses
│   ├── spine_localization.py
│   ├── vertebrae_localization.py
│   └── vertebrae_segmentation.py
│
├── models/                  # Neural network architectures
│   ├── layers.py            # Reusable building blocks
│   ├── unet3d.py            # 3D U-Net implementations
│   └── scnet.py             # SpatialConfiguration-Net
│
├── data/                    # Data loading and augmentation
│   ├── utils.py             # Image I/O utilities
│   ├── transforms.py        # Data augmentation
│   └── dataset.py           # PyTorch Dataset classes
│
├── training/                # Training scripts
│   ├── trainer.py           # Base trainer class
│   ├── losses.py            # Loss functions
│   ├── train_spine_localization.py
│   ├── train_vertebrae_localization.py
│   └── train_vertebrae_segmentation.py
│
├── inference/               # Inference scripts
│   ├── predictor.py         # Base predictor class
│   ├── spine_predictor.py
│   ├── vertebrae_predictor.py
│   ├── segmentation_predictor.py
│   └── full_pipeline.py     # End-to-end pipeline
│
└── utils/                   # Utilities
    ├── metrics.py           # Evaluation metrics
    └── visualization.py     # Plotting utilities
```

## Dataset Preparation

### CSV Format

Create a CSV file with the following columns:

```csv
name,type,image_path,mask_path,centroid_json_path
verse001,train,path/to/ct.nii.gz,path/to/mask.nii.gz,path/to/centroids.json
```

- `name`: Unique sample identifier
- `type`: 'train' or 'val' (optional, for fixed splits)
- `image_path`: Path to CT image (NIfTI format)
- `mask_path`: Path to multi-label segmentation mask
- `centroid_json_path`: Path to JSON with vertebra centroids

### Centroid JSON Format

```json
[
  {"name": "C1", "x": 100.5, "y": 200.3, "z": 50.2},
  {"name": "C2", "x": 102.1, "y": 201.5, "z": 65.3},
  ...
]
```

### Segmentation Mask Labels

Labels follow VerSe challenge format:
- 1-7: C1-C7 (Cervical)
- 8-19: T1-T12 (Thoracic)
- 20-25: L1-L6 (Lumbar)
- 26-27: S1-S2 (Sacral)

## Training

### Train All Stages Sequentially

```bash
# Stage 1: Spine Localization
python -m pytorch_pipeline.train \
    --stage spine \
    --csv data/dataset.csv \
    --output outputs/

# Stage 2: Vertebrae Localization (after Stage 1 completes)
python -m pytorch_pipeline.train \
    --stage vertebrae \
    --csv data/dataset.csv \
    --output outputs/

# Stage 3: Vertebrae Segmentation (after Stage 2 completes)
python -m pytorch_pipeline.train \
    --stage segmentation \
    --csv data/dataset.csv \
    --output outputs/
```

### Train Single Fold

```bash
python -m pytorch_pipeline.train \
    --stage spine \
    --csv data/dataset.csv \
    --output outputs/ \
    --fold 0 \
    --num_folds 5
```

### Custom Training Parameters

```bash
python -m pytorch_pipeline.train \
    --stage segmentation \
    --csv data/dataset.csv \
    --output outputs/ \
    --epochs 200 \
    --batch_size 4 \
    --lr 0.0001 \
    --device cuda
```

### Resume Training

```bash
python -m pytorch_pipeline.train \
    --stage spine \
    --csv data/dataset.csv \
    --output outputs/ \
    --resume outputs/spine_localization/checkpoints/fold_0/latest.pth
```

## Inference

### Single Image

```bash
python -m pytorch_pipeline.infer \
    --image ct_scan.nii.gz \
    --model_dir outputs/ \
    --output results/ \
    --device cuda
```

### Batch Processing

```bash
python -m pytorch_pipeline.infer \
    --image images_directory/ \
    --model_dir outputs/ \
    --output results/ \
    --save_intermediates
```

### With Precomputed Landmarks

```bash
python -m pytorch_pipeline.infer \
    --image ct_scan.nii.gz \
    --landmarks landmarks.json \
    --model_dir outputs/ \
    --output results/
```

## Python API

### Training

```python
from pytorch_pipeline import SpineLocalizationTrainer, SpineLocalizationConfig

config = SpineLocalizationConfig(
    num_epochs=100,
    batch_size=2,
    learning_rate=0.0001
)

trainer = SpineLocalizationTrainer(
    output_dir='outputs/',
    csv_path='data/dataset.csv',
    fold=0,
    device='cuda',
    config=config
)

best_loss = trainer.train()
```

### Inference

```python
from pytorch_pipeline import FullPipelinePredictor

pipeline = FullPipelinePredictor(
    spine_checkpoint='outputs/spine_localization/checkpoints/fold_0/best.pth',
    vertebrae_checkpoint='outputs/vertebrae_localization/checkpoints/fold_0/best.pth',
    segmentation_checkpoint='outputs/vertebrae_segmentation/checkpoints/fold_0/best.pth',
    device='cuda'
)

result = pipeline.predict(
    'ct_scan.nii.gz',
    output_dir='results/',
    save_intermediates=True
)

# Access results
segmentation = result['segmentation']  # SimpleITK image
landmarks = result['landmarks']  # Dict of vertebra centroids
```

### Evaluation

```python
from pytorch_pipeline import VerSeMetrics
import SimpleITK as sitk

metrics = VerSeMetrics(spacing=(1.0, 1.0, 1.0))

pred_seg = sitk.GetArrayFromImage(sitk.ReadImage('pred.nii.gz'))
gt_seg = sitk.GetArrayFromImage(sitk.ReadImage('gt.nii.gz'))

results = metrics.evaluate_case(pred_seg, gt_seg)
print(f"Dice: {results['aggregate']['dice_mean']:.3f}")
print(f"HD95: {results['aggregate']['hd95_mean']:.1f} mm")
```

## Configuration

### Stage 1: Spine Localization

| Parameter | Default | Description |
|-----------|---------|-------------|
| image_size | (64, 64, 128) | Input volume size (D, H, W) |
| image_spacing | (8.0, 8.0, 8.0) | Voxel spacing in mm |
| num_filters_base | 64 | Base number of filters |
| heatmap_sigma | 4.0 | Gaussian sigma for heatmap |
| num_epochs | 100 | Training epochs |
| batch_size | 2 | Batch size |
| learning_rate | 1e-4 | Initial learning rate |

### Stage 2: Vertebrae Localization

| Parameter | Default | Description |
|-----------|---------|-------------|
| image_size | (96, 96, 128) | Input volume size |
| image_spacing | (2.0, 2.0, 2.0) | Voxel spacing in mm |
| num_landmarks | 26 | Number of vertebrae to detect |
| learnable_sigma | True | Learn per-vertebra sigma |
| num_epochs | 150 | Training epochs |

### Stage 3: Vertebrae Segmentation

| Parameter | Default | Description |
|-----------|---------|-------------|
| image_size | (128, 128, 96) | Crop size per vertebra |
| image_spacing | (1.0, 1.0, 1.0) | Voxel spacing in mm |
| dice_weight | 0.5 | Weight for Dice loss |
| bce_weight | 0.5 | Weight for BCE loss |
| num_epochs | 100 | Training epochs |

## Metrics

### Segmentation Metrics
- **Dice Coefficient**: Overlap between prediction and ground truth
- **IoU / Jaccard**: Intersection over union
- **Hausdorff Distance (HD)**: Maximum surface distance
- **HD95**: 95th percentile Hausdorff distance
- **ASSD**: Average symmetric surface distance

### Localization Metrics
- **MLD**: Mean localization distance in mm
- **Identification Rate**: Percentage within threshold (4mm, 20mm)

## Outputs

### Training Outputs

```
outputs/
├── spine_localization/
│   ├── checkpoints/
│   │   ├── fold_0/
│   │   │   ├── best.pth
│   │   │   ├── latest.pth
│   │   │   └── epoch_0050.pth
│   │   └── fold_1/
│   │       └── ...
│   └── logs/
│       └── fold_0/
│           ├── events.out.tfevents...
│           └── history.json
```

### Inference Outputs

```
results/
├── case_001_segmentation.nii.gz
├── case_001_landmarks.json
├── case_001_spine_centroid.json
└── intermediates/
    └── case_001/
        ├── spine_heatmap.nii.gz
        ├── L1_heatmap.nii.gz
        └── ...
```

## TensorBoard

Monitor training progress:

```bash
tensorboard --logdir outputs/spine_localization/logs/
```

## Citation

If you use this pipeline, please cite:

```bibtex
@article{payer2020integrating,
  title={Integrating spatial configuration into heatmap regression based CNNs for landmark localization},
  author={Payer, Christian and {\v{S}}tern, Darko and Bischof, Horst and Urschler, Martin},
  journal={Medical Image Analysis},
  volume={54},
  pages={207--219},
  year={2019}
}
```

## License

This reimplementation follows the original project's license terms.

## Acknowledgments

- Original TensorFlow implementation
- VerSe Challenge organizers
- SimpleITK community
