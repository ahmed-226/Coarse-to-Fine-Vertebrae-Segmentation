"""
Main inference script for the vertebrae segmentation pipeline.

Runs the full 3-stage pipeline on CT images.

Usage:
    # Single image
    python infer.py --image ct.nii.gz --model_dir outputs/ --output results/

    # Batch processing
    python infer.py --image_dir images/ --model_dir outputs/ --output results/

    # With precomputed landmarks (skip stage 1 and 2)
    python infer.py --image ct.nii.gz --landmarks landmarks.json --model_dir outputs/ --output results/
"""
import argparse
from pathlib import Path
from typing import List
import glob

from inference import FullPipelinePredictor, run_inference


def find_images(image_path: str) -> List[str]:
    """Find all NIfTI images in a directory or return single path."""
    path = Path(image_path)
    
    if path.is_file():
        return [str(path)]
    elif path.is_dir():
        patterns = ['*.nii', '*.nii.gz']
        images = []
        for pattern in patterns:
            images.extend(glob.glob(str(path / pattern)))
            images.extend(glob.glob(str(path / '**' / pattern), recursive=True))
        return sorted(set(images))
    else:
        raise ValueError(f"Invalid path: {image_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run Vertebrae Segmentation Inference'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to CT image or directory containing images'
    )
    
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Directory containing trained models for all stages'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--fold',
        type=int,
        default=0,
        help='Model fold to use (default: 0)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on (cuda/cpu)'
    )
    
    parser.add_argument(
        '--landmarks',
        type=str,
        default=None,
        help='Path to precomputed landmarks JSON (skips stage 1 and 2)'
    )
    
    parser.add_argument(
        '--save_intermediates',
        action='store_true',
        help='Save intermediate heatmaps'
    )
    
    args = parser.parse_args()
    
    # Find images
    image_paths = find_images(args.image)
    print(f"Found {len(image_paths)} images to process")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    model_dir = Path(args.model_dir)
    
    pipeline = FullPipelinePredictor(
        spine_checkpoint=str(model_dir / 'spine_localization' / 'checkpoints' / f'fold_{args.fold}' / 'best.pth'),
        vertebrae_checkpoint=str(model_dir / 'vertebrae_localization' / 'checkpoints' / f'fold_{args.fold}' / 'best.pth'),
        segmentation_checkpoint=str(model_dir / 'vertebrae_segmentation' / 'checkpoints' / f'fold_{args.fold}' / 'best.pth'),
        device=args.device
    )
    
    # Process images
    if args.landmarks and len(image_paths) == 1:
        # Use precomputed landmarks
        result = pipeline.predict_with_landmarks(
            image_paths[0],
            args.landmarks,
            output_dir=str(output_path)
        )
    elif len(image_paths) == 1:
        # Single image
        result = pipeline.predict(
            image_paths[0],
            output_dir=str(output_path),
            save_intermediates=args.save_intermediates
        )
    else:
        # Batch processing
        results = pipeline.predict_batch(
            image_paths,
            output_dir=str(output_path),
            save_intermediates=args.save_intermediates
        )
        
        # Print summary
        print("\n" + "="*60)
        print("Batch Processing Summary")
        print("="*60)
        for name, info in results.items():
            status = info['status']
            if status == 'success':
                print(f"  {name}: {info['num_vertebrae']} vertebrae detected")
            else:
                print(f"  {name}: ERROR - {info['error']}")
    
    print("\n" + "="*60)
    print(f"Results saved to: {output_path}")
    print("="*60)


if __name__ == '__main__':
    main()
