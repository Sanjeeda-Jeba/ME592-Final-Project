"""Metrics for evaluating autonomous driving models."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error


class DrivingMetrics:
    """
    Class for computing and visualizing evaluation metrics for driving models.
    """
    
    @staticmethod
    def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute Mean Absolute Error (MAE) for steering, throttle, and brake.
        
        Args:
            y_true: Ground truth values of shape (n_samples, 3)
            y_pred: Predicted values of shape (n_samples, 3)
            
        Returns:
            Dictionary of MAE values for each output
        """
        mae_steering = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
        mae_throttle = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
        mae_brake = mean_absolute_error(y_true[:, 2], y_pred[:, 2])
        mae_overall = mean_absolute_error(y_true, y_pred)
        
        return {
            'steering': mae_steering,
            'throttle': mae_throttle,
            'brake': mae_brake,
            'overall': mae_overall
        }
    
    @staticmethod
    def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute Root Mean Square Error (RMSE) for steering, throttle, and brake.
        
        Args:
            y_true: Ground truth values of shape (n_samples, 3)
            y_pred: Predicted values of shape (n_samples, 3)
            
        Returns:
            Dictionary of RMSE values for each output
        """
        rmse_steering = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
        rmse_throttle = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
        rmse_brake = np.sqrt(mean_squared_error(y_true[:, 2], y_pred[:, 2]))
        rmse_overall = np.sqrt(mean_squared_error(y_true, y_pred))
        
        return {
            'steering': rmse_steering,
            'throttle': rmse_throttle,
            'brake': rmse_brake,
            'overall': rmse_overall
        }
    
    @staticmethod
    def plot_actual_vs_predicted(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_samples: int = 100,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot actual vs predicted values.
        
        Args:
            y_true: Ground truth values of shape (n_samples, 3)
            y_pred: Predicted values of shape (n_samples, 3)
            n_samples: Number of samples to plot
            save_path: Path to save the plot, if None, the plot is shown
        """
        # Make sure we don't try to plot more samples than we have
        n_samples = min(n_samples, len(y_true))
        
        # Only plot a subset of the data for clarity
        indices = np.random.choice(len(y_true), n_samples, replace=False)
        
        labels = ['Steering', 'Throttle', 'Brake']
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        for i, (label, ax) in enumerate(zip(labels, axs)):
            ax.plot(y_true[indices, i], label='Actual', marker='o')
            ax.plot(y_pred[indices, i], label='Predicted', marker='x')
            ax.set_title(f'{label} - Actual vs Predicted')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_error_distribution(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        bins: int = 30,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot the distribution of errors.
        
        Args:
            y_true: Ground truth values of shape (n_samples, 3)
            y_pred: Predicted values of shape (n_samples, 3)
            bins: Number of bins for the histogram
            save_path: Path to save the plot, if None, the plot is shown
        """
        errors = y_pred - y_true
        
        labels = ['Steering', 'Throttle', 'Brake']
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        for i, (label, ax) in enumerate(zip(labels, axs)):
            ax.hist(errors[:, i], bins=bins)
            ax.set_title(f'{label} - Error Distribution')
            ax.set_xlabel('Error')
            ax.set_ylabel('Frequency')
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def compare_models(
        model_results: Dict[str, Dict[str, float]],
        metric: str = 'mae',
        save_path: Optional[str] = None
    ) -> None:
        """
        Compare multiple models based on a specified metric.
        
        Args:
            model_results: Dictionary mapping model names to their metric results
            metric: Metric to compare ('mae' or 'rmse')
            save_path: Path to save the plot, if None, the plot is shown
        """
        model_names = list(model_results.keys())
        labels = ['Steering', 'Throttle', 'Brake', 'Overall']
        
        metric_values = {}
        for label in labels:
            metric_values[label] = [model_results[model][metric][label.lower()] for model in model_names]
        
        x = np.arange(len(model_names))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, label in enumerate(labels):
            ax.bar(x + i * width, metric_values[label], width, label=label)
        
        ax.set_xlabel('Model')
        ax.set_ylabel(f'{metric.upper()} Value')
        ax.set_title(f'Model Comparison - {metric.upper()}')
        ax.set_xticks(x + width * (len(labels) - 1) / 2)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.grid(True, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def compare_environments(
        environment_results: Dict[str, Dict[str, float]],
        metric: str = 'mae',
        save_path: Optional[str] = None
    ) -> None:
        """
        Compare model performance across different environments.
        
        Args:
            environment_results: Dictionary mapping environment names to their metric results
            metric: Metric to compare ('mae' or 'rmse')
            save_path: Path to save the plot, if None, the plot is shown
        """
        environment_names = list(environment_results.keys())
        labels = ['Steering', 'Throttle', 'Brake', 'Overall']
        
        metric_values = {}
        for label in labels:
            metric_values[label] = [environment_results[env][metric][label.lower()] for env in environment_names]
        
        x = np.arange(len(environment_names))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, label in enumerate(labels):
            ax.bar(x + i * width, metric_values[label], width, label=label)
        
        ax.set_xlabel('Environment')
        ax.set_ylabel(f'{metric.upper()} Value')
        ax.set_title(f'Environment Comparison - {metric.upper()}')
        ax.set_xticks(x + width * (len(labels) - 1) / 2)
        ax.set_xticklabels(environment_names)
        ax.legend()
        ax.grid(True, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show() 