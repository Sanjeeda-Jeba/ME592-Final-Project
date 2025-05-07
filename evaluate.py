#!/usr/bin/env python
"""
Evaluation script for autonomous driving models.
This script evaluates trained models on various environmental conditions.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from autodrive.data.dataset import AutonomousDrivingDataset
from autodrive.models.cnn import get_model
from autodrive.augmentation.transforms import EnvironmentalAugmentation
from autodrive.evaluation.metrics import DrivingMetrics
from autodrive.utils.logger import Logger


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate autonomous driving models")
    
    # Dataset arguments
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing the dataset")
    parser.add_argument("--csv_file", type=str, default="data/val.csv",
                        help="CSV file with validation data")
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Directory containing images")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="resnet", choices=["basic_cnn", "resnet"],
                        help="Model architecture to use")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    # Environmental conditions
    parser.add_argument("--environments", nargs="+", default=["clean", "rain", "fog", "night"],
                        help="Environments to evaluate on")
    parser.add_argument("--severity", type=float, default=0.5,
                        help="Severity of environmental effects")
    
    # Output arguments
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to save results")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for evaluation")
    
    return parser.parse_args()


def create_environmental_dataloader(csv_file, img_dir, env_type, severity, batch_size, num_workers):
    """
    Create a dataloader with a specific environmental condition.
    
    Args:
        csv_file: Path to the CSV file with data
        img_dir: Directory with images
        env_type: Type of environment ('clean', 'rain', 'fog', 'night')
        severity: Severity of the environmental effect
        batch_size: Batch size for the dataloader
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader with the specified environmental condition
    """
    # Create the base dataset
    dataset = AutonomousDrivingDataset(csv_file=csv_file, img_dir=img_dir)
    
    # Apply environmental augmentation if needed
    if env_type != "clean":
        env_transform = EnvironmentalAugmentation.get_augmentation_transform(env_type, severity)
        
        # Create a new dataset with the transformation
        dataset_transformed = AutonomousDrivingDataset(
            csv_file=csv_file,
            img_dir=img_dir,
            transform=env_transform
        )
        
        # Replace the dataset
        dataset = dataset_transformed
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return dataloader


def evaluate_model(model, dataloader, device):
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to use for evaluation
        
    Returns:
        Dictionary of evaluation results
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Get data
            images = batch['image'].to(device)
            controls = batch['controls'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Store predictions and targets
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(controls.cpu().numpy())
    
    # Concatenate predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    mae = DrivingMetrics.compute_mae(all_targets, all_preds)
    rmse = DrivingMetrics.compute_rmse(all_targets, all_preds)
    
    return {
        'predictions': all_preds,
        'targets': all_targets,
        'mae': mae,
        'rmse': rmse
    }


def main():
    """Main function to evaluate the model."""
    args = parse_args()
    
    # Set up directories
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set up device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Set up logger
    logger = Logger(args.results_dir, name="evaluation", tensorboard=False)
    logger.info(f"Starting evaluation with arguments: {args}")
    
    # Load model
    model = get_model(args.model)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Model '{args.model}' loaded from checkpoint: {args.checkpoint}")
    
    # Evaluate on each environment
    results = {}
    for env_type in args.environments:
        logger.info(f"Evaluating on {env_type} environment...")
        
        # Skip 'clean' environment or use the appropriate augmentation
        if env_type == "clean":
            dataloader = create_environmental_dataloader(
                args.csv_file, args.img_dir, "clean", 0.0, args.batch_size, args.num_workers
            )
        else:
            dataloader = create_environmental_dataloader(
                args.csv_file, args.img_dir, env_type, args.severity, args.batch_size, args.num_workers
            )
        
        # Evaluate model
        env_results = evaluate_model(model, dataloader, device)
        results[env_type] = env_results
        
        # Log metrics
        logger.info(f"{env_type} MAE - Steering: {env_results['mae']['steering']:.4f}, "
                   f"Throttle: {env_results['mae']['throttle']:.4f}, "
                   f"Brake: {env_results['mae']['brake']:.4f}, "
                   f"Overall: {env_results['mae']['overall']:.4f}")
        
        # Save plots for this environment
        DrivingMetrics.plot_actual_vs_predicted(
            env_results['targets'], env_results['predictions'],
            save_path=os.path.join(args.results_dir, f"{env_type}_actual_vs_predicted.png")
        )
        
        DrivingMetrics.plot_error_distribution(
            env_results['targets'], env_results['predictions'],
            save_path=os.path.join(args.results_dir, f"{env_type}_error_distribution.png")
        )
    
    # Compare environments
    env_mae_results = {env: results[env]['mae'] for env in results}
    env_rmse_results = {env: results[env]['rmse'] for env in results}
    
    DrivingMetrics.compare_environments(
        env_mae_results, 'mae',
        save_path=os.path.join(args.results_dir, "environment_comparison_mae.png")
    )
    
    DrivingMetrics.compare_environments(
        env_rmse_results, 'rmse',
        save_path=os.path.join(args.results_dir, "environment_comparison_rmse.png")
    )
    
    logger.info("Evaluation completed successfully")
    logger.close()


if __name__ == "__main__":
    main() 