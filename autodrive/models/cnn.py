"""CNN models for autonomous driving behavior cloning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Dict, Optional, List, Union, Any


class BasicCNN(nn.Module):
    """Basic CNN model for behavior cloning."""
    
    def __init__(self, input_channels: int = 3, output_dim: int = 3):
        """
        Initialize the CNN model.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            output_dim: Number of output dimensions (3 for steering, throttle, brake)
        """
        super(BasicCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(24)
        self.bn2 = nn.BatchNorm2d(36)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(1152, 100)  # Size depends on input image dimensions
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Convolutional layers with ReLU and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x


class ResNetModel(nn.Module):
    """ResNet-based model for behavior cloning."""
    
    def __init__(self, output_dim: int = 3, pretrained: bool = True):
        """
        Initialize the ResNet model.
        
        Args:
            output_dim: Number of output dimensions (3 for steering, throttle, brake)
            pretrained: Whether to use pretrained weights
        """
        super(ResNetModel, self).__init__()
        
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Replace the last fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.resnet(x)


def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to get a model by name.
    
    Args:
        model_name: Name of the model to get
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        Model instance
    """
    models_dict = {
        'basic_cnn': BasicCNN,
        'resnet': ResNetModel,
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models_dict.keys())}")
    
    return models_dict[model_name](**kwargs) 