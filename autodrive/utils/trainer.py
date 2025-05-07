"""Training utilities for autonomous driving models."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Union, Any, Optional, Callable
from torch.utils.data import DataLoader
from tqdm import tqdm

from autodrive.utils.logger import Logger


class Trainer:
    """
    Trainer class for autonomous driving models.
    This class handles the training and validation of models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        logger: Optional[Logger] = None,
        scheduler: Optional[Any] = None,
        save_dir: str = 'checkpoints',
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer to use
            criterion: Loss function
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use for training (cuda or cpu)
            logger: Logger for logging
            scheduler: Learning rate scheduler
            save_dir: Directory to save checkpoints
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger
        self.scheduler = scheduler
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize best validation loss
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        # Use tqdm for progress bar
        with tqdm(self.train_loader, unit="batch") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Get data
                images = batch['image'].to(self.device)
                controls = batch['controls'].to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss = self.criterion(outputs, controls)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Update progress bar
                pbar.set_description(f"Train Loss: {loss.item():.4f}")
        
        # Calculate average loss
        avg_loss = total_loss / total_samples
        
        return {'loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        # Lists to store predictions and ground truth
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            with tqdm(self.val_loader, unit="batch") as pbar:
                for batch_idx, batch in enumerate(pbar):
                    # Get data
                    images = batch['image'].to(self.device)
                    controls = batch['controls'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    
                    # Calculate loss
                    loss = self.criterion(outputs, controls)
                    
                    # Update metrics
                    batch_size = images.size(0)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
                    
                    # Store predictions and targets
                    all_preds.append(outputs.cpu().numpy())
                    all_targets.append(controls.cpu().numpy())
                    
                    # Update progress bar
                    pbar.set_description(f"Val Loss: {loss.item():.4f}")
        
        # Calculate average loss
        avg_loss = total_loss / total_samples
        
        # Concatenate predictions and targets
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate MAE for each output
        steering_mae = np.mean(np.abs(all_preds[:, 0] - all_targets[:, 0]))
        throttle_mae = np.mean(np.abs(all_preds[:, 1] - all_targets[:, 1]))
        brake_mae = np.mean(np.abs(all_preds[:, 2] - all_targets[:, 2]))
        
        return {
            'loss': avg_loss,
            'steering_mae': steering_mae,
            'throttle_mae': throttle_mae,
            'brake_mae': brake_mae,
            'overall_mae': (steering_mae + throttle_mae + brake_mae) / 3
        }
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Dictionary of training history
        """
        # Log the beginning of training
        if self.logger:
            self.logger.info(f"Starting training for {num_epochs} epochs")
        
        # Initialize history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_steering_mae': [],
            'val_throttle_mae': [],
            'val_brake_mae': [],
            'val_overall_mae': []
        }
        
        # Train for the specified number of epochs
        for epoch in range(1, num_epochs + 1):
            # Print epoch information
            if self.logger:
                self.logger.info(f"Epoch {epoch}/{num_epochs}")
            else:
                print(f"Epoch {epoch}/{num_epochs}")
            
            # Train and validate
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step(val_metrics['loss'])
            
            # Log metrics
            if self.logger:
                self.logger.log_metrics(train_metrics, epoch, prefix='train')
                self.logger.log_metrics(val_metrics, epoch, prefix='val')
            else:
                print(f"Train Loss: {train_metrics['loss']:.4f}")
                print(f"Val Loss: {val_metrics['loss']:.4f}")
                print(f"Val Steering MAE: {val_metrics['steering_mae']:.4f}")
                print(f"Val Throttle MAE: {val_metrics['throttle_mae']:.4f}")
                print(f"Val Brake MAE: {val_metrics['brake_mae']:.4f}")
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_steering_mae'].append(val_metrics['steering_mae'])
            history['val_throttle_mae'].append(val_metrics['throttle_mae'])
            history['val_brake_mae'].append(val_metrics['brake_mae'])
            history['val_overall_mae'].append(val_metrics['overall_mae'])
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics['loss'])
            
            # Save if best model so far
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, val_metrics['loss'], is_best=True)
        
        # Log the end of training
        if self.logger:
            self.logger.info("Training finished")
        
        return history
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False) -> None:
        """
        Save a model checkpoint.
        
        Args:
            epoch: Epoch number
            loss: Validation loss
            is_best: Whether this is the best model so far
        """
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        
        # Add scheduler state if available
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if this is the best so far
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            if self.logger:
                self.logger.info(f"New best model saved at epoch {epoch} with loss {loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint
            
        Returns:
            Epoch number of the checkpoint
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Set best validation loss
        self.best_val_loss = checkpoint['loss']
        
        # Log the loading of the checkpoint
        if self.logger:
            self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
        
        return checkpoint['epoch'] 