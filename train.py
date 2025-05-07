#!/usr/bin/env python
"""
Training script for autonomous driving.
This script handles the entire training pipeline for the autonomous driving models.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from autodrive.data.dataset import AutonomousDrivingDataset, AutonomousDrivingDatasetLoader
from autodrive.models.cnn import get_model
from autodrive.augmentation.transforms import DrivingTransforms, EnvironmentalAugmentation
from autodrive.utils.logger import Logger
from autodrive.utils.trainer import Trainer


# Constants
# Kaggle Lyft Udacity Challenge dataset
UDACITY_DATASET_URLS = [
    "https://www.kaggle.com/datasets/kumaresanmanickavelu/lyft-udacity-challenge/download",  # Main Kaggle dataset
    "https://github.com/udacity/self-driving-car/releases/download/autti_data_1/dataset-1.bag.tar.gz",  # Udacity backup 1
    "https://github.com/udacity/self-driving-car/releases/download/autti_data_2/dataset-2.bag.tar.gz"   # Udacity backup 2
]
UDACITY_DATASET_URL = UDACITY_DATASET_URLS[0]  # Default to first URL


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train autonomous driving models")
    
    # Dataset arguments
    parser.add_argument("--data_dir", type=str, default="data", 
                        help="Directory to store dataset")
    parser.add_argument("--dataset_url", type=str, default=UDACITY_DATASET_URL,
                        help="URL to download the dataset from (default: Udacity dataset)")
    parser.add_argument("--use_fallback_datasets", action="store_true",
                        help="Try multiple fallback datasets if the main one fails")
    parser.add_argument("--use_local_dataset", action="store_true",
                        help="Use local dataset without downloading")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split ratio")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="resnet", choices=["basic_cnn", "resnet"],
                        help="Model architecture to use")
    parser.add_argument("--pretrained", action="store_true", 
                        help="Use pretrained weights for ResNet")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for optimizer")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    # Augmentation arguments
    parser.add_argument("--augment", action="store_true",
                        help="Use data augmentation")
    parser.add_argument("--env_augment", action="store_true",
                        help="Use environmental augmentation")
    parser.add_argument("--env_type", type=str, default="rain", choices=["rain", "fog", "night"],
                        help="Type of environmental augmentation")
    parser.add_argument("--severity", type=float, default=0.5,
                        help="Severity of environmental augmentation")
    
    # Output arguments
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory to store logs")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Use TensorBoard for logging")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    
    # Continue training
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint to resume training from")
    
    return parser.parse_args()


def prepare_dataset(args):
    """
    Prepare the dataset.
    
    Args:
        args: Command line arguments
        
    Returns:
        Training and validation dataloaders
    """
    # Use local dataset if specified
    dataset_dir = None
    errors = []
    
    # If user wants to use local dataset, skip download
    if args.use_local_dataset:
        print(f"Using local dataset in directory: {args.data_dir}")
        dataset_dir = args.data_dir
    else:
        # If user specified a custom URL, try that first
        if args.dataset_url != UDACITY_DATASET_URL:
            print(f"Trying user-specified dataset URL: {args.dataset_url}")
            dataset_dir = AutonomousDrivingDatasetLoader.download_dataset(args.dataset_url, args.data_dir)
            if not dataset_dir:
                errors.append(f"User-specified URL failed: {args.dataset_url}")
        
        # If user URL failed or wasn't specified, try the default URLs
        if not dataset_dir and args.use_fallback_datasets:
            for i, url in enumerate(UDACITY_DATASET_URLS):
                if url == args.dataset_url:  # Skip if we already tried this URL
                    continue
                print(f"Trying dataset URL {i+1}/{len(UDACITY_DATASET_URLS)}: {url}")
                dataset_dir = AutonomousDrivingDatasetLoader.download_dataset(url, args.data_dir)
                if dataset_dir:
                    print(f"Successfully downloaded dataset from {url}")
                    break
                else:
                    errors.append(f"URL {i+1} failed: {url}")
        
        if not dataset_dir:
            error_msg = "Failed to download dataset"
            if errors:
                error_msg += ":\n" + "\n".join(errors)
            error_msg += "\nPlease check your internet connection or specify a different dataset URL."
            raise RuntimeError(error_msg)
    
    # Prepare necessary directories and files
    img_dir = os.path.join(dataset_dir, "IMG")
    driving_log = os.path.join(dataset_dir, "driving_log.csv")
    
    # Split data into training and validation sets
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    try:
        df = pd.read_csv(driving_log)
        
        # Check if the CSV has the expected columns
        required_columns = ['center', 'steering', 'throttle', 'brake']
        if not all(col in df.columns for col in required_columns):
            print(f"Warning: CSV doesn't have all required columns. Available columns: {df.columns.tolist()}")
            
            # If the CSV doesn't have the expected format, try to adapt
            if 'center' not in df.columns and len(df.columns) >= 1:
                # Assume first column is the image path
                df = df.rename(columns={df.columns[0]: 'center'})
                print(f"Renamed first column to 'center': {df.columns[0]}")
            
            # Add missing columns with default values if needed
            for col, default_val in [('steering', 0.0), ('throttle', 0.5), ('brake', 0.0)]:
                if col not in df.columns:
                    df[col] = default_val
                    print(f"Added missing column '{col}' with default value {default_val}")
        
        # Split the data
        train_df, val_df = train_test_split(df, test_size=args.val_split, random_state=42)
        
        # Save split dataframes to disk
        train_csv = os.path.join(args.data_dir, "train.csv")
        val_csv = os.path.join(args.data_dir, "val.csv")
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        
        # Get transforms
        transform = None
        if args.augment:
            transform = DrivingTransforms.get_train_transform()
        
        # If environmental augmentation is requested, compose with the environmental transform
        if args.env_augment:
            env_transform = EnvironmentalAugmentation.get_augmentation_transform(args.env_type, args.severity)
            
            # Compose the transforms
            orig_transform = transform
            
            def composed_transform(sample):
                if orig_transform:
                    sample = orig_transform(sample)
                sample = env_transform(sample)
                return sample
            
            transform = composed_transform
        
        # Prepare dataloaders
        train_loader, val_loader = AutonomousDrivingDatasetLoader.prepare_dataloaders(
            csv_train=train_csv,
            csv_val=val_csv,
            img_dir=img_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            transform=transform
        )
        
        return train_loader, val_loader
    
    except Exception as e:
        raise RuntimeError(f"Error preparing dataset: {e}")


def main():
    """Main function to train the model."""
    args = parse_args()
    
    # Set up directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set up device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Set up logger
    logger = Logger(args.log_dir, tensorboard=args.tensorboard)
    logger.info(f"Starting training with arguments: {args}")
    
    # Prepare dataset
    train_loader, val_loader = prepare_dataset(args)
    logger.info(f"Dataset prepared. Train samples: {len(train_loader.dataset)}, "
               f"Val samples: {len(val_loader.dataset)}")
    
    # Build model
    model = get_model(args.model, pretrained=args.pretrained)
    logger.info(f"Model '{args.model}' created")
    
    # Set up optimizer, loss function, and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Set up trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger=logger,
        scheduler=scheduler,
        save_dir=args.save_dir
    )
    
    # Resume training if checkpoint is provided
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed training from epoch {start_epoch}")
    
    # Train the model
    history = trainer.train(args.epochs)
    
    # Log final message
    logger.info("Training completed successfully")
    logger.close()


if __name__ == "__main__":
    main() 