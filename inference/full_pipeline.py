"""
Full Pipeline Predictor combining all 3 stages.
Provides end-to-end inference from CT image to segmentation.
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import numpy as np
import SimpleITK as sitk

from config.spine_localization import SpineLocalizationConfig
from config.vertebrae_localization import VertebraeLocalizationConfig
from config.vertebrae_segmentation import VertebraeSegmentationConfig
from .spine_predictor import SpineLocalizationPredictor
from .vertebrae_predictor import VertebraeLocalizationPredictor
from .segmentation_predictor import VertebraeSegmentationPredictor


class FullPipelinePredictor:
    """
    Full pipeline predictor for vertebrae segmentation.
    
    Combines all three stages:
    1. Spine Localization: Find spine centroid for cropping
    2. Vertebrae Localization: Find individual vertebra centroids
    3. Vertebrae Segmentation: Segment each vertebra
    
    Can also perform inference with precomputed landmarks.
    """
    
    def __init__(
        self,
        spine_checkpoint: str,
        vertebrae_checkpoint: str,
        segmentation_checkpoint: str,
        device: str = 'cuda',
        spine_config: Optional[SpineLocalizationConfig] = None,
        vertebrae_config: Optional[VertebraeLocalizationConfig] = None,
        segmentation_config: Optional[VertebraeSegmentationConfig] = None
    ):
        """
        Initialize all three stage predictors.
        
        Args:
            spine_checkpoint: Path to Stage 1 model checkpoint
            vertebrae_checkpoint: Path to Stage 2 model checkpoint
            segmentation_checkpoint: Path to Stage 3 model checkpoint
            device: Device to run inference on
            spine_config: Optional Stage 1 configuration
            vertebrae_config: Optional Stage 2 configuration
            segmentation_config: Optional Stage 3 configuration
        """
        print("Initializing Full Pipeline Predictor...")
        
        # Initialize Stage 1: Spine Localization
        print("  Loading Stage 1: Spine Localization...")
        self.spine_predictor = SpineLocalizationPredictor(
            checkpoint_path=spine_checkpoint,
            device=device,
            config=spine_config
        )
        
        # Initialize Stage 2: Vertebrae Localization
        print("  Loading Stage 2: Vertebrae Localization...")
        self.vertebrae_predictor = VertebraeLocalizationPredictor(
            checkpoint_path=vertebrae_checkpoint,
            device=device,
            config=vertebrae_config
        )
        
        # Initialize Stage 3: Vertebrae Segmentation
        print("  Loading Stage 3: Vertebrae Segmentation...")
        self.segmentation_predictor = VertebraeSegmentationPredictor(
            checkpoint_path=segmentation_checkpoint,
            device=device,
            config=segmentation_config
        )
        
        print("Pipeline initialized successfully!")
    
    def predict(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        save_intermediates: bool = False
    ) -> Dict[str, Any]:
        """
        Run full pipeline on a single CT image.
        
        Args:
            image_path: Path to input CT image (NIfTI format)
            output_dir: Optional directory to save outputs
            save_intermediates: Whether to save intermediate results
        
        Returns:
            Dictionary with:
                - 'segmentation': Final segmentation (SimpleITK)
                - 'landmarks': Detected vertebra centroids
                - 'spine_centroid': Spine centroid from Stage 1
                - 'per_vertebra_masks': Individual vertebra masks
        """
        print(f"\nProcessing: {image_path}")
        
        # Load image
        image = sitk.ReadImage(image_path)
        print(f"  Image size: {image.GetSize()}")
        print(f"  Image spacing: {image.GetSpacing()}")
        
        # Stage 1: Spine Localization
        print("\n  Stage 1: Spine Localization...")
        spine_result = self.spine_predictor.predict(image)
        spine_centroid = spine_result['centroid_physical']
        print(f"    Spine centroid: [{spine_centroid[0]:.1f}, {spine_centroid[1]:.1f}, {spine_centroid[2]:.1f}]")
        
        # Stage 2: Vertebrae Localization
        print("\n  Stage 2: Vertebrae Localization...")
        vertebrae_result = self.vertebrae_predictor.predict(image, spine_centroid)
        landmarks = vertebrae_result['landmarks']
        print(f"    Detected {len(landmarks)} vertebrae")
        
        for name, coords in landmarks.items():
            print(f"      {name}: [{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}]")
        
        # Stage 3: Vertebrae Segmentation
        print("\n  Stage 3: Vertebrae Segmentation...")
        segmentation_result = self.segmentation_predictor.predict(image, landmarks)
        
        # Save outputs if requested
        if output_dir:
            self._save_outputs(
                output_dir,
                image_path,
                segmentation_result,
                landmarks,
                spine_centroid,
                save_intermediates,
                spine_result if save_intermediates else None,
                vertebrae_result if save_intermediates else None
            )
        
        return {
            'segmentation': segmentation_result['segmentation'],
            'landmarks': landmarks,
            'spine_centroid': spine_centroid,
            'per_vertebra_masks': segmentation_result['per_vertebra']
        }
    
    def predict_with_landmarks(
        self,
        image_path: str,
        landmarks_path: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run Stage 3 only using precomputed landmarks.
        
        Args:
            image_path: Path to input CT image
            landmarks_path: Path to JSON file with landmarks
            output_dir: Optional output directory
        
        Returns:
            Dictionary with segmentation results
        """
        print(f"\nProcessing with precomputed landmarks: {image_path}")
        
        # Load image
        image = sitk.ReadImage(image_path)
        
        # Load landmarks
        with open(landmarks_path, 'r') as f:
            landmarks_data = json.load(f)
        
        # Convert to expected format
        landmarks = {}
        for item in landmarks_data:
            name = item['name']
            coords = np.array([item['x'], item['y'], item['z']])
            landmarks[name] = coords
        
        print(f"  Loaded {len(landmarks)} landmarks")
        
        # Run segmentation
        segmentation_result = self.segmentation_predictor.predict(image, landmarks)
        
        if output_dir:
            self._save_outputs(
                output_dir,
                image_path,
                segmentation_result,
                landmarks
            )
        
        return {
            'segmentation': segmentation_result['segmentation'],
            'landmarks': landmarks,
            'per_vertebra_masks': segmentation_result['per_vertebra']
        }
    
    def predict_batch(
        self,
        image_paths: List[str],
        output_dir: str,
        save_intermediates: bool = False
    ) -> Dict[str, Dict]:
        """
        Run pipeline on multiple images.
        
        Args:
            image_paths: List of paths to CT images
            output_dir: Directory to save all outputs
            save_intermediates: Whether to save intermediate results
        
        Returns:
            Dictionary mapping image names to results
        """
        results = {}
        
        for i, image_path in enumerate(image_paths):
            print(f"\n{'='*60}")
            print(f"Processing {i+1}/{len(image_paths)}")
            print(f"{'='*60}")
            
            try:
                result = self.predict(
                    image_path,
                    output_dir=output_dir,
                    save_intermediates=save_intermediates
                )
                results[Path(image_path).stem] = {
                    'status': 'success',
                    'num_vertebrae': len(result['landmarks'])
                }
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results[Path(image_path).stem] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Save summary
        summary_path = Path(output_dir) / 'batch_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _save_outputs(
        self,
        output_dir: str,
        image_path: str,
        segmentation_result: Dict,
        landmarks: Dict,
        spine_centroid: Optional[np.ndarray] = None,
        save_intermediates: bool = False,
        spine_result: Optional[Dict] = None,
        vertebrae_result: Optional[Dict] = None
    ):
        """Save pipeline outputs to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get base name
        base_name = Path(image_path).stem
        if base_name.endswith('_ct'):
            base_name = base_name[:-3]
        
        # Save segmentation
        seg_path = output_path / f"{base_name}_segmentation.nii.gz"
        sitk.WriteImage(segmentation_result['segmentation'], str(seg_path))
        print(f"  Saved segmentation: {seg_path}")
        
        # Save landmarks as JSON
        landmarks_json = []
        for name, coords in landmarks.items():
            landmarks_json.append({
                'name': name,
                'x': float(coords[0]),
                'y': float(coords[1]),
                'z': float(coords[2])
            })
        
        landmarks_path = output_path / f"{base_name}_landmarks.json"
        with open(landmarks_path, 'w') as f:
            json.dump(landmarks_json, f, indent=2)
        print(f"  Saved landmarks: {landmarks_path}")
        
        # Save spine centroid
        if spine_centroid is not None:
            centroid_data = {
                'x': float(spine_centroid[0]),
                'y': float(spine_centroid[1]),
                'z': float(spine_centroid[2])
            }
            centroid_path = output_path / f"{base_name}_spine_centroid.json"
            with open(centroid_path, 'w') as f:
                json.dump(centroid_data, f, indent=2)
        
        # Save intermediate heatmaps if requested
        if save_intermediates:
            intermediates_dir = output_path / 'intermediates' / base_name
            intermediates_dir.mkdir(parents=True, exist_ok=True)
            
            if spine_result and 'heatmap' in spine_result:
                heatmap = sitk.GetImageFromArray(spine_result['heatmap'])
                sitk.WriteImage(heatmap, str(intermediates_dir / 'spine_heatmap.nii.gz'))
            
            if vertebrae_result and 'heatmaps' in vertebrae_result:
                for i, name in enumerate(self.vertebrae_predictor.VERTEBRA_NAMES):
                    if vertebrae_result['valid_mask'][i]:
                        heatmap = sitk.GetImageFromArray(vertebrae_result['heatmaps'][i])
                        sitk.WriteImage(heatmap, str(intermediates_dir / f'{name}_heatmap.nii.gz'))


def run_inference(
    image_path: str,
    model_dir: str,
    output_dir: str,
    fold: int = 0,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Convenience function to run full pipeline inference.
    
    Args:
        image_path: Path to CT image
        model_dir: Directory containing trained models for all stages
        output_dir: Directory to save outputs
        fold: Fold index for model selection
        device: Device to run on
    
    Returns:
        Pipeline results
    """
    model_path = Path(model_dir)
    
    # Find checkpoints
    spine_ckpt = model_path / 'spine_localization' / 'checkpoints' / f'fold_{fold}' / 'best.pth'
    vertebrae_ckpt = model_path / 'vertebrae_localization' / 'checkpoints' / f'fold_{fold}' / 'best.pth'
    segmentation_ckpt = model_path / 'vertebrae_segmentation' / 'checkpoints' / f'fold_{fold}' / 'best.pth'
    
    # Initialize pipeline
    pipeline = FullPipelinePredictor(
        spine_checkpoint=str(spine_ckpt),
        vertebrae_checkpoint=str(vertebrae_ckpt),
        segmentation_checkpoint=str(segmentation_ckpt),
        device=device
    )
    
    # Run inference
    result = pipeline.predict(image_path, output_dir=output_dir, save_intermediates=True)
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run full vertebrae segmentation pipeline')
    parser.add_argument('--image', type=str, required=True, help='Path to CT image')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory with trained models')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--fold', type=int, default=0, help='Model fold to use')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    run_inference(
        image_path=args.image,
        model_dir=args.model_dir,
        output_dir=args.output,
        fold=args.fold,
        device=args.device
    )
