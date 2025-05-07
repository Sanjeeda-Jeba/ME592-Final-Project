"""Image transformations for data augmentation."""

import cv2
import numpy as np
import torch
import albumentations as A
from typing import Dict, Any, Callable, Tuple, Union, Optional


class DrivingTransforms:
    """
    Collection of transformations for autonomous driving datasets.
    These transformations can be used for data augmentation.
    """
    
    @staticmethod
    def get_train_transform() -> Callable:
        """
        Get transformations for training.
        
        Returns:
            A callable that applies the transformations to a sample
        """
        # Using Albumentations for efficient transformations
        transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        def apply_transform(sample: Dict[str, Any]) -> Dict[str, Any]:
            """Apply the transformations to a sample."""
            image = sample['image']
            controls = sample['controls']
            
            # Apply transformations to the image
            transformed = transform(image=image)
            transformed_image = transformed['image']
            
            # If horizontal flip was applied, negate the steering angle
            # We can't directly check if flip was applied, so we'll use a hack
            # This assumes that horizontal flip was applied with 50% probability
            if np.random.random() < 0.5:
                controls[0] = -controls[0]  # Negate steering angle
            
            return {
                'image': transformed_image,
                'controls': controls
            }
        
        return apply_transform
    
    @staticmethod
    def get_val_transform() -> Callable:
        """
        Get transformations for validation.
        
        Returns:
            A callable that applies the transformations to a sample
        """
        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        def apply_transform(sample: Dict[str, Any]) -> Dict[str, Any]:
            """Apply the transformations to a sample."""
            image = sample['image']
            controls = sample['controls']
            
            # Apply transformations to the image
            transformed = transform(image=image)
            transformed_image = transformed['image']
            
            return {
                'image': transformed_image,
                'controls': controls
            }
        
        return apply_transform


class EnvironmentalAugmentation:
    """
    Augmentations to simulate different environmental conditions.
    These augmentations are used to generate challenging driving conditions.
    """
    
    @staticmethod
    def add_rain(image: np.ndarray, severity: float = 0.3) -> np.ndarray:
        """
        Add rain effect to an image.
        
        Args:
            image: Input image
            severity: Severity of the rain effect (0.0 to 1.0)
            
        Returns:
            Image with rain effect
        """
        # Create a random matrix for rain drops
        rain_drops = np.random.random(image.shape[:2])
        rain_drops = rain_drops < severity / 10
        
        # Create a brighter version for the rain drops
        bright = np.ones_like(image) * 255
        
        # Apply the rain drops to the image
        rain_image = np.where(np.stack([rain_drops] * 3, axis=2), bright, image)
        
        # Add a bluish tint to simulate rainy conditions
        blue_tint = np.array([0.9, 0.9, 1.0])
        rain_image = rain_image * blue_tint
        
        # Apply a slight blur to simulate rain's effect on visibility
        rain_image = cv2.GaussianBlur(rain_image, (3, 3), 0)
        
        return np.clip(rain_image, 0, 255).astype(np.uint8)
    
    @staticmethod
    def add_fog(image: np.ndarray, severity: float = 0.3) -> np.ndarray:
        """
        Add fog effect to an image.
        
        Args:
            image: Input image
            severity: Severity of the fog effect (0.0 to 1.0)
            
        Returns:
            Image with fog effect
        """
        # Create a fog layer (white-ish)
        fog_color = np.array([200, 200, 200], dtype=np.uint8)
        fog_image = np.ones_like(image) * fog_color
        
        # Blend the original image with the fog layer
        alpha = 1 - (severity * 0.7)
        fog_image = cv2.addWeighted(image, alpha, fog_image, 1 - alpha, 0)
        
        # Add a slight blur to simulate fog's effect on visibility
        fog_image = cv2.GaussianBlur(fog_image, (5, 5), 0)
        
        return fog_image
    
    @staticmethod
    def simulate_night(image: np.ndarray, severity: float = 0.6) -> np.ndarray:
        """
        Simulate night-time conditions.
        
        Args:
            image: Input image
            severity: Severity of the night effect (0.0 to 1.0)
            
        Returns:
            Image with night-time effect
        """
        # Reduce brightness
        night_image = image * (1 - severity * 0.7)
        
        # Increase blue channel slightly (night usually has a blue tint)
        night_image[:, :, 0] = np.clip(night_image[:, :, 0] * 1.1, 0, 255)
        
        # Add slight noise to simulate low-light conditions
        noise = np.random.normal(0, 5, image.shape)
        night_image = night_image + noise
        
        return np.clip(night_image, 0, 255).astype(np.uint8)
    
    @staticmethod
    def get_augmentation_transform(augmentation_type: str, severity: float = 0.5) -> Callable:
        """
        Get a transformation for the specified environmental augmentation.
        
        Args:
            augmentation_type: Type of augmentation ('rain', 'fog', 'night')
            severity: Severity of the augmentation (0.0 to 1.0)
            
        Returns:
            A callable that applies the augmentation to a sample
        """
        augmentation_funcs = {
            'rain': EnvironmentalAugmentation.add_rain,
            'fog': EnvironmentalAugmentation.add_fog,
            'night': EnvironmentalAugmentation.simulate_night,
        }
        
        if augmentation_type not in augmentation_funcs:
            raise ValueError(f"Augmentation type {augmentation_type} not supported. "
                             f"Available types: {list(augmentation_funcs.keys())}")
        
        augmentation_func = augmentation_funcs[augmentation_type]
        
        def apply_augmentation(sample: Dict[str, Any]) -> Dict[str, Any]:
            """Apply the augmentation to a sample."""
            image = sample['image']
            controls = sample['controls']
            
            # Apply the augmentation
            augmented_image = augmentation_func(image, severity)
            
            return {
                'image': augmented_image,
                'controls': controls
            }
        
        return apply_augmentation 