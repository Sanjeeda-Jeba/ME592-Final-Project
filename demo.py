#!/usr/bin/env python
"""
Demo script for visualizing model predictions.
This script loads a trained model and visualizes its predictions on sample images.
"""

import os
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from autodrive.models.cnn import get_model
from autodrive.augmentation.transforms import EnvironmentalAugmentation


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Demo for autonomous driving model")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="resnet", choices=["basic_cnn", "resnet"],
                        help="Model architecture to use")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    
    # Input arguments
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input image or directory of images")
    
    # Environmental condition
    parser.add_argument("--env_type", type=str, default=None, choices=[None, "rain", "fog", "night"],
                        help="Type of environmental augmentation to apply")
    parser.add_argument("--severity", type=float, default=0.5,
                        help="Severity of environmental augmentation")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="demo_output",
                        help="Directory to save output visualizations")
    parser.add_argument("--show", action="store_true",
                        help="Show visualizations in addition to saving them")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for evaluation")
    
    return parser.parse_args()


def preprocess_image(image_path, env_type=None, severity=0.5):
    """
    Preprocess an image for model input.
    
    Args:
        image_path: Path to the image
        env_type: Type of environmental augmentation to apply
        severity: Severity of environmental augmentation
        
    Returns:
        Preprocessed image tensor and original image
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply environmental augmentation if requested
    original_image = image.copy()
    if env_type:
        env_aug = EnvironmentalAugmentation()
        if env_type == "rain":
            image = env_aug.add_rain(image, severity)
        elif env_type == "fog":
            image = env_aug.add_fog(image, severity)
        elif env_type == "night":
            image = env_aug.simulate_night(image, severity)
    
    # Resize image if needed
    image = cv2.resize(image, (224, 224))
    
    # Normalize and convert to tensor
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    return image_tensor, original_image, image


def visualize_prediction(image, prediction, output_path=None, show=False):
    """
    Visualize model prediction on an image.
    
    Args:
        image: Input image
        prediction: Model prediction [steering, throttle, brake]
        output_path: Path to save the visualization
        show: Whether to show the visualization
    """
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Display image
    plt.subplot(2, 1, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")
    
    # Display prediction
    plt.subplot(2, 1, 2)
    labels = ["Steering", "Throttle", "Brake"]
    colors = ["blue", "green", "red"]
    
    plt.bar(labels, prediction, color=colors)
    plt.title("Model Prediction")
    plt.ylim(-1, 1)
    
    # Add text values
    for i, v in enumerate(prediction):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top")
    
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


def main():
    """Main function for the demo."""
    args = parse_args()
    
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    model = get_model(args.model)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model '{args.model}' loaded from checkpoint: {args.checkpoint}")
    
    # Process input
    if os.path.isdir(args.input):
        # Process all images in directory
        image_files = [f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_paths = [os.path.join(args.input, f) for f in image_files]
    else:
        # Process single image
        image_paths = [args.input]
    
    # Process each image
    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        
        # Preprocess image
        image_tensor, original_image, processed_image = preprocess_image(
            image_path, args.env_type, args.severity
        )
        
        # Get model prediction
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            prediction = model(image_tensor).cpu().numpy()[0]
        
        # Visualize prediction
        output_path = os.path.join(
            args.output_dir, 
            f"{os.path.splitext(os.path.basename(image_path))[0]}"
            f"{'_' + args.env_type if args.env_type else ''}.png"
        )
        
        visualize_prediction(
            processed_image, 
            prediction, 
            output_path=output_path, 
            show=args.show
        )
    
    print(f"Processed {len(image_paths)} images. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 