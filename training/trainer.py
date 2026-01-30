"""
Base Trainer class for all training stages.
Provides common functionality: logging, checkpointing, metrics tracking.
"""
import os
import time
import json
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from tqdm import tqdm


class BaseTrainer(ABC):
    """
    Base trainer class providing common training functionality.
    
    Subclasses must implement:
        - create_model()
        - create_datasets()
        - create_loss_function()
        - train_step()
        - val_step()
        - compute_metrics()
    """
    
    def __init__(
        self,
        config: Any,
        output_dir: str,
        csv_path: str,
        fold: int = 0,
        num_folds: int = 5,
        device: str = 'cuda',
        seed: int = 42,
        resume_from: Optional[str] = None,
        multi_gpu: bool = False
    ):
        """
        Args:
            config: Stage-specific configuration dataclass
            output_dir: Directory for saving checkpoints and logs
            csv_path: Path to CSV file with dataset info
            fold: Current fold index (0 to num_folds-1)
            num_folds: Total number of folds for cross-validation
            device: Device to train on ('cuda' or 'cpu')
            seed: Random seed for reproducibility
            resume_from: Path to checkpoint to resume from
            multi_gpu: Whether to use multiple GPUs if available
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.csv_path = csv_path
        self.fold = fold
        self.num_folds = num_folds
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        self.resume_from = resume_from
        self.multi_gpu = multi_gpu
        
        # Set seeds for reproducibility
        self._set_seed()
        
        # Create output directories
        self.checkpoint_dir = self.output_dir / 'checkpoints' / f'fold_{fold}'
        self.log_dir = self.output_dir / 'logs' / f'fold_{fold}'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Training state
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[_LRScheduler] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.loss_fn = None
        
        # Metrics tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')  # Lower is better (loss-based)
        self.best_epoch = 0
        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []
        
        # Initialize components
        self._setup()
    
    def _set_seed(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        
        # Deterministic operations (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _setup(self):
        """Initialize model, datasets, optimizer, and scheduler."""
        print(f"\n{'='*60}")
        print(f"Setting up training for Fold {self.fold}/{self.num_folds-1}")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")
        
        # Create model
        self.model = self.create_model()
        self.model = self.model.to(self.device)
        
        # Multi-GPU support
        if self.multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training\n")
            self.model = nn.DataParallel(self.model)
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable\n")
        
        # Create datasets and loaders
        train_dataset, val_dataset = self.create_datasets()
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}\n")
        
        # Create loss function
        self.loss_fn = self.create_loss_function()
        
        # Create optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=getattr(self.config, 'weight_decay', 0.0)
        )
        
        # Create scheduler
        scheduler_type = getattr(self.config, 'scheduler', 'plateau')
        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=1e-7
            )
        
        # Resume from checkpoint if specified
        if self.resume_from:
            self._load_checkpoint(self.resume_from)
    
    @abstractmethod
    def create_model(self) -> nn.Module:
        """Create and return the model for this stage."""
        pass
    
    @abstractmethod
    def create_datasets(self) -> Tuple[Any, Any]:
        """Create and return (train_dataset, val_dataset)."""
        pass
    
    @abstractmethod
    def create_loss_function(self):
        """Create and return the loss function."""
        pass
    
    @abstractmethod
    def train_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step.
        
        Args:
            batch: Dictionary containing batch data
        
        Returns:
            Dictionary containing loss values
        """
        pass
    
    @abstractmethod
    def val_step(self, batch: Dict) -> Dict[str, Any]:
        """
        Perform a single validation step.
        
        Args:
            batch: Dictionary containing batch data
        
        Returns:
            Dictionary containing loss values and predictions
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, predictions: List, targets: List) -> Dict[str, float]:
        """
        Compute evaluation metrics from predictions.
        
        Args:
            predictions: List of model predictions
            targets: List of ground truth targets
        
        Returns:
            Dictionary of metric names to values
        """
        pass
    
    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        
        epoch_losses = {}
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._to_device(batch)
            
            # Training step
            losses = self.train_step(batch)
            
            # Accumulate losses
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value.item()
            
            # Update progress bar
            pbar.set_postfix({k: f"{v.item():.4f}" for k, v in losses.items()})
            
            # Log to TensorBoard
            self.writer.add_scalar('train/loss_step', losses['total'].item(), self.global_step)
            self.global_step += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def val_epoch(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Run one validation epoch."""
        self.model.eval()
        
        epoch_losses = {}
        all_predictions = []
        all_targets = []
        num_batches = len(self.val_loader)
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                batch = self._to_device(batch)
                
                # Validation step
                result = self.val_step(batch)
                losses = result['losses']
                
                # Accumulate losses
                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    if isinstance(value, torch.Tensor):
                        epoch_losses[key] += value.item()
                    else:
                        epoch_losses[key] += value
                
                # Collect predictions for metrics
                if 'predictions' in result:
                    all_predictions.extend(result['predictions'])
                if 'targets' in result:
                    all_targets.extend(result['targets'])
                
                pbar.set_postfix({k: f"{v:.4f}" if isinstance(v, float) else f"{v.item():.4f}" 
                                  for k, v in losses.items()})
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Compute metrics
        metrics = {}
        if all_predictions and all_targets:
            metrics = self.compute_metrics(all_predictions, all_targets)
        
        return epoch_losses, metrics
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.config.num_epochs} epochs...")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Batch size: {self.config.batch_size}\n")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Training
            train_losses = self.train_epoch()
            self.train_history.append({'epoch': epoch, **train_losses})
            
            # Validation
            val_losses, val_metrics = self.val_epoch()
            self.val_history.append({'epoch': epoch, **val_losses, **val_metrics})
            
            # Log to TensorBoard
            for key, value in train_losses.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)
            for key, value in val_losses.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)
            
            # Learning rate scheduling
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('train/lr', current_lr, epoch)
            
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_losses['total'])
            elif self.scheduler is not None:
                self.scheduler.step()
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            self._print_epoch_summary(epoch, train_losses, val_losses, val_metrics, epoch_time)
            
            # Save checkpoint
            is_best = val_losses['total'] < self.best_metric
            if is_best:
                self.best_metric = val_losses['total']
                self.best_epoch = epoch
            
            self._save_checkpoint(is_best)
            
            # Early stopping check
            if hasattr(self.config, 'early_stopping_patience'):
                if epoch - self.best_epoch >= self.config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch - self.best_epoch} epochs without improvement")
                    break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_metric:.4f} at epoch {self.best_epoch}")
        
        # Save training history
        self._save_history()
        
        self.writer.close()
        
        return self.best_metric
    
    def _to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, dict):
                result[key] = self._to_device(value)
            else:
                result[key] = value
        return result
    
    def _print_epoch_summary(
        self,
        epoch: int,
        train_losses: Dict,
        val_losses: Dict,
        val_metrics: Dict,
        epoch_time: float
    ):
        """Print epoch summary to console."""
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{self.config.num_epochs-1} | Time: {epoch_time:.1f}s")
        print(f"{'='*60}")
        
        print(f"Train Loss: {train_losses['total']:.4f}")
        print(f"Val Loss:   {val_losses['total']:.4f}")
        
        if val_metrics:
            print(f"\nMetrics:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")
        
        print(f"\nBest: {self.best_metric:.4f} @ epoch {self.best_epoch}")
        print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config),
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"  New best model saved!")
        
        # Save periodic checkpoints
        if self.current_epoch % getattr(self.config, 'save_every', 10) == 0:
            periodic_path = self.checkpoint_dir / f'epoch_{self.current_epoch:04d}.pth'
            torch.save(checkpoint, periodic_path)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training."""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        self.best_epoch = checkpoint['best_epoch']
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        print(f"Resumed from epoch {self.current_epoch}, best metric: {self.best_metric:.4f}")
    
    def _save_history(self):
        """Save training history to JSON."""
        history = {
            'train': self.train_history,
            'val': self.val_history,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config)
        }
        
        history_path = self.log_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved to {history_path}")
    
    def load_best_model(self) -> nn.Module:
        """Load the best model checkpoint."""
        best_path = self.checkpoint_dir / 'best.pth'
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['best_epoch']}")
        else:
            print("No best checkpoint found, using current model")
        
        return self.model
