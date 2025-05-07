"""Logging utilities for training and evaluation."""

import os
import time
import logging
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Logger class for training and evaluation.
    This class handles logging to both files and TensorBoard.
    """
    
    def __init__(
        self,
        log_dir: str,
        name: Optional[str] = None,
        tensorboard: bool = True,
    ):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to save logs
            name: Name of the logger
            tensorboard: Whether to use TensorBoard
        """
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up file logger
        self.logger = logging.getLogger(name or 'autodrive')
        self.logger.setLevel(logging.INFO)
        
        # Add file handler
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name or "train"}_{timestamp}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Set format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Initialize TensorBoard if requested
        self.tensorboard = None
        if tensorboard:
            tensorboard_dir = os.path.join(log_dir, 'tensorboard', f'{name or "train"}_{timestamp}')
            self.tensorboard = SummaryWriter(tensorboard_dir)
        
        self.info(f"Logger initialized. Logs will be saved to {log_file}")
        if tensorboard:
            self.info(f"TensorBoard logs will be saved to {tensorboard_dir}")
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: int,
        prefix: str = '',
    ) -> None:
        """
        Log metrics to both file and TensorBoard.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Training step or epoch
            prefix: Prefix for metric names
        """
        # Create a message for the file log
        message = f"Step {step} - "
        message += " - ".join([f"{prefix + k if prefix else k}: {v}" for k, v in metrics.items()])
        self.info(message)
        
        # Log to TensorBoard if available
        if self.tensorboard:
            for k, v in metrics.items():
                name = f"{prefix + '/' + k if prefix else k}"
                self.tensorboard.add_scalar(name, v, step)
    
    def log_images(
        self,
        images: Dict[str, Any],
        step: int,
    ) -> None:
        """
        Log images to TensorBoard.
        
        Args:
            images: Dictionary of images to log
            step: Training step or epoch
        """
        if self.tensorboard:
            for name, image in images.items():
                self.tensorboard.add_image(name, image, step)
        else:
            self.warning("TensorBoard not initialized, cannot log images")
    
    def close(self) -> None:
        """Close the logger."""
        if self.tensorboard:
            self.tensorboard.close() 