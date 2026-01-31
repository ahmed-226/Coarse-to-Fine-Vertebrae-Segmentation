# PyTorch Vertebrae Segmentation Pipeline
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg?logo=nvidia)
![Segmentation](https://img.shields.io/badge/Task-Segmentation-blue)
![Medical](https://img.shields.io/badge/Domain-Medical-red)
![U-Net](https://img.shields.io/badge/Architecture-U--Net-success)
![License](https://img.shields.io/badge/license-MIT-important.svg)

## Overview

This pipeline segments individual vertebrae from CT images using a 3-stage approach:

1. **Spine Localization**: A 3D U-Net predicts a heatmap centered on the spine, 
   providing a centroid for cropping in subsequent stages.

2. **Vertebrae Localization**: A SpatialConfiguration-Net (SCNet) predicts heatmaps 
   for each individual vertebra centroid, with learnable sigma parameters.

3. **Vertebrae Segmentation**: A 3D U-Net segments each vertebra by cropping 
   around its predicted centroid.

## Installation
```bash
git clone https://github.com/ahmed-226/Coarse-to-Fine-Vertebrae-Segmentation.git
```
### Requirements

```bash
pip install -r Coarse-to-Fine-Vertebrae-Segmentation/requirements.txt
```

### Minimum Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU training)
- 16GB RAM minimum, 32GB recommended
- 8GB VRAM minimum for training

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

## Training

### Train All Stages Sequentially

```bash
# Stage 1: Spine Localization
python train.py \
    --stage spine \
    --csv verse19_dataset.csv \
    --output experiments/baseline \
    --epochs <epoch number> \
    --batch_size <batch number> \
    --lr 0.0001 \
    --device cuda \
    --multi_gpu

    # To use fold-based validation 
    # --fold -1
    # --num_folds 5

# Stage 2: Vertebrae Localization (after Stage 1 completes)
python /train.py \
    --stage vertebrae \
    --csv "/kaggle/input/verse19-csv/verse19_dataset.csv" \
    --spine_model_path <spine localization checkpoint> \
    --output outputs \
    --epochs <epoch number> \
    --batch_size <batch number> \
    --multi_gpu

# Stage 3: Vertebrae Segmentation (after Stage 2 completes)
python -m pytorch_pipeline.train \
    --stage segmentation \
    --csv data/dataset.csv \
    --output outputs/ \
    --epochs <epoch number> \
    --batch_size <batch number> 
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

## Citation

If you use this pipeline, please cite:

```bibtex
@inproceedings{Payer2020,
  title     = {Coarse to Fine Vertebrae Localization and Segmentation with SpatialConfiguration-Net and U-Net},
  author    = {Payer, Christian and {\v{S}}tern, Darko and Bischof, Horst and Urschler, Martin},
  booktitle = {Proceedings of the 15th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 5: VISAPP},
  doi       = {10.5220/0008975201240133},
  pages     = {124--133},
  volume    = {5},
  year      = {2020}
}
```
