"""
Main training script for the vertebrae segmentation pipeline.

Supports training all three stages:
1. Spine Localization
2. Vertebrae Localization
3. Vertebrae Segmentation

Usage:
    # Train Stage 1 (Spine Localization)
    python train.py --stage spine --csv data.csv --output outputs/spine

    # Train Stage 2 (Vertebrae Localization)
    python train.py --stage vertebrae --csv data.csv --output outputs/vertebrae

    # Train Stage 3 (Vertebrae Segmentation)
    python train.py --stage segmentation --csv data.csv --output outputs/seg

    # Train all folds for a stage
    python train.py --stage spine --csv data.csv --output outputs/ --all_folds
"""

import sys
import argparse
from pathlib import Path

# Fix imports: add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from config import (
    SpineLocalizationConfig,
    VertebraeLocalizationConfig,
    VertebraeSegmentationConfig
)
from training import (
    train_spine_localization,
    train_all_folds_spine,
    train_vertebrae_localization,
    train_all_folds_vertebrae,
    train_vertebrae_segmentation,
    train_all_folds_segmentation
)


def main():
    parser = argparse.ArgumentParser(
        description='Train Vertebrae Segmentation Pipeline'
    )
    
    parser.add_argument(
        '--stage',
        type=str,
        required=True,
        choices=['spine', 'vertebrae', 'segmentation', 'all'],
        help='Training stage: spine, vertebrae, segmentation, or all'
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to dataset CSV file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for checkpoints and logs'
    )
    
    parser.add_argument(
        '--fold',
        type=int,
        default=-1,
        help='Fold to train (-1 for all folds)'
    )
    
    parser.add_argument(
        '--num_folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to train on (cuda/cpu)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Checkpoint to resume from'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Override batch size'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Override learning rate'
    )
    
    parser.add_argument(
        '--spine_model_path',
        type=str,
        default=None,
        help='Path to Stage 1 (Spine Localization) checkpoint to use for Stage 2 training'
    )
    
    parser.add_argument(
        '--vertebrae_model_path',
        type=str,
        default=None,
        help='Path to Stage 2 (Vertebrae Localization) checkpoint to use for Stage 3 training'
    )
    
    parser.add_argument(
        '--multi_gpu',
        action='store_true',
        help='Use multiple GPUs if available'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Train stages
    if args.stage == 'spine' or args.stage == 'all':
        print("\n" + "="*60)
        print("Training Stage 1: Spine Localization")
        print("="*60)
        
        config = SpineLocalizationConfig()
        if args.epochs:
            config.num_epochs = args.epochs
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.lr:
            config.learning_rate = args.lr
        
        stage_output = output_path / 'spine_localization'
        
        if args.fold < 0:
            train_all_folds_spine(
                csv_path=args.csv,
                output_dir=str(stage_output),
                num_folds=args.num_folds,
                device=args.device,
                config=config,
                multi_gpu=args.multi_gpu
            )
        else:
            train_spine_localization(
                csv_path=args.csv,
                output_dir=str(stage_output),
                fold=args.fold,
                num_folds=args.num_folds,
                device=args.device,
                config=config,
                resume_from=args.resume,
                multi_gpu=args.multi_gpu
            )
    
    if args.stage == 'vertebrae' or args.stage == 'all':
        print("\n" + "="*60)
        print("Training Stage 2: Vertebrae Localization")
        print("="*60)
        
        config = VertebraeLocalizationConfig()
        if args.epochs:
            config.num_epochs = args.epochs
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.lr:
            config.learning_rate = args.lr
        
        stage_output = output_path / 'vertebrae_localization'
        
        # Determine spine model path: use provided path or fall back to expected path
        if args.spine_model_path:
            spine_model_dir_resolved = Path(args.spine_model_path).parent if Path(args.spine_model_path).is_file() else Path(args.spine_model_path)
        else:
            spine_model_dir_resolved = output_path / 'spine_localization' / 'checkpoints'
        
        if args.fold < 0:
            train_all_folds_vertebrae(
                csv_path=args.csv,
                output_dir=str(stage_output),
                num_folds=args.num_folds,
                device=args.device,
                config=config,
                spine_model_dir=str(spine_model_dir_resolved) if spine_model_dir_resolved.exists() else None,
                multi_gpu=args.multi_gpu
            )
        else:
            # If specific model path provided, use it directly; otherwise construct path
            if args.spine_model_path:
                spine_model_to_use = Path(args.spine_model_path)
            else:
                spine_model_to_use = spine_model_dir_resolved / f'fold_{args.fold}' / 'best.pth'
            
            train_vertebrae_localization(
                csv_path=args.csv,
                output_dir=str(stage_output),
                fold=args.fold,
                num_folds=args.num_folds,
                device=args.device,
                config=config,
                spine_model_path=str(spine_model_to_use) if spine_model_to_use.exists() else None,
                resume_from=args.resume,
                multi_gpu=args.multi_gpu
            )
    
    if args.stage == 'segmentation' or args.stage == 'all':
        print("\n" + "="*60)
        print("Training Stage 3: Vertebrae Segmentation")
        print("="*60)
        
        config = VertebraeSegmentationConfig()
        if args.epochs:
            config.num_epochs = args.epochs
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.lr:
            config.learning_rate = args.lr
        
        stage_output = output_path / 'vertebrae_segmentation'
        
        # Determine vertebrae localization model path: use provided path or fall back to expected path
        if args.vertebrae_model_path:
            loc_model_dir_resolved = Path(args.vertebrae_model_path).parent if Path(args.vertebrae_model_path).is_file() else Path(args.vertebrae_model_path)
        else:
            loc_model_dir_resolved = output_path / 'vertebrae_localization' / 'checkpoints'
        
        if args.fold < 0:
            train_all_folds_segmentation(
                csv_path=args.csv,
                output_dir=str(stage_output),
                num_folds=args.num_folds,
                device=args.device,
                config=config,
                localization_model_dir=str(loc_model_dir_resolved) if loc_model_dir_resolved.exists() else None,
                multi_gpu=args.multi_gpu
            )
        else:
            # If specific model path provided, use it directly; otherwise construct path
            if args.vertebrae_model_path:
                loc_model_to_use = Path(args.vertebrae_model_path)
            else:
                loc_model_to_use = loc_model_dir_resolved / f'fold_{args.fold}' / 'best.pth'
            
            train_vertebrae_segmentation(
                csv_path=args.csv,
                output_dir=str(stage_output),
                fold=args.fold,
                num_folds=args.num_folds,
                device=args.device,
                config=config,
                localization_model_path=str(loc_model_to_use) if loc_model_to_use.exists() else None,
                resume_from=args.resume,
                multi_gpu=args.multi_gpu
            )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
