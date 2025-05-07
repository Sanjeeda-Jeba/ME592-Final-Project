"""Utilities for downloading datasets."""

import os
import subprocess
import zipfile
import tarfile
import urllib.request
from typing import Optional, List


def download_dataset(url: str, target_dir: str) -> Optional[str]:
    """
    Download and extract a dataset from a URL.
    
    Args:
        url: URL to download the dataset from
        target_dir: Directory to save the dataset
        
    Returns:
        Path to the extracted dataset or None if failed
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # Extract the file name from the URL
    file_name = os.path.basename(url.split('?')[0])
    if not file_name:
        file_name = "dataset.zip"  # Default name if we can't extract
    
    # Handle different file types
    if "kaggle.com" in url:
        # Kaggle requires API authentication, recommend manual download
        print(f"Kaggle URL detected: {url}")
        print("To download from Kaggle, you need to:")
        print("1. Create a Kaggle account and get your API credentials")
        print("2. Download the dataset manually from your browser and place it in the data directory")
        print("3. Or install kaggle CLI with: pip install kaggle")
        
        # Check if kaggle is installed
        try:
            subprocess.run(["kaggle", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Kaggle CLI detected. Attempting download with kaggle CLI...")
            
            # Parse dataset name from URL
            # Format: https://www.kaggle.com/datasets/username/datasetname/download
            parts = url.split("/")
            if "datasets" in parts:
                dataset_idx = parts.index("datasets")
                if dataset_idx + 2 < len(parts):
                    dataset = f"{parts[dataset_idx+1]}/{parts[dataset_idx+2]}"
                    
                    # Download the dataset
                    try:
                        subprocess.run(
                            ["kaggle", "datasets", "download", "-d", dataset, "-p", target_dir],
                            check=True
                        )
                        print(f"Dataset downloaded to {target_dir}")
                        
                        # Find the downloaded zip file
                        zip_files = [f for f in os.listdir(target_dir) if f.endswith(".zip")]
                        if zip_files:
                            file_name = zip_files[0]
                            file_path = os.path.join(target_dir, file_name)
                            
                            # Extract the dataset
                            extract_dataset(file_path, target_dir)
                            return os.path.join(target_dir, "extracted")
                    except subprocess.CalledProcessError as e:
                        print(f"Failed to download dataset: {e}")
                        print("Please download the dataset manually and place it in the data directory")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Kaggle CLI not found or not properly configured")
            print("Please download the dataset manually and place it in the data directory")
        
        return None
    
    # Standard URL download
    file_path = os.path.join(target_dir, file_name)
    
    # Download only if not already downloaded
    if not os.path.exists(file_path):
        print(f"Downloading dataset from {url}...")
        try:
            urllib.request.urlretrieve(url, file_path)
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None
    
    # Extract the dataset
    return extract_dataset(file_path, target_dir)


def extract_dataset(file_path: str, target_dir: str) -> Optional[str]:
    """
    Extract a dataset from a zip or tar file.
    
    Args:
        file_path: Path to the zip or tar file
        target_dir: Directory to extract to
        
    Returns:
        Path to the extracted directory or None if failed
    """
    extract_dir = os.path.join(target_dir, "extracted")
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        print(f"Extracting dataset to {extract_dir}...")
        
        try:
            if file_path.endswith(".zip"):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif file_path.endswith((".tar.gz", ".tgz")):
                with tarfile.open(file_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(extract_dir)
            elif file_path.endswith(".tar"):
                with tarfile.open(file_path, 'r') as tar_ref:
                    tar_ref.extractall(extract_dir)
            else:
                print(f"Unsupported file format: {file_path}")
                return None
        except Exception as e:
            print(f"Error extracting dataset: {e}")
            return None
    
    return extract_dir


def prepare_img_directory(extract_dir: str) -> Optional[str]:
    """
    Find or create an IMG directory for the dataset.
    
    Args:
        extract_dir: Directory containing the extracted dataset
        
    Returns:
        Path to the IMG directory or None if failed
    """
    img_dir = os.path.join(extract_dir, "IMG")
    if os.path.exists(img_dir):
        return img_dir
    
    # Look for common image file extensions
    import glob
    
    # Look for directories that might contain images
    possible_img_dirs = []
    for root, dirs, files in os.walk(extract_dir):
        for dir_name in dirs:
            if dir_name.lower() in ["images", "imgs", "img", "frames"]:
                possible_img_dirs.append(os.path.join(root, dir_name))
    
    if possible_img_dirs:
        # Use the first directory found
        img_dir = possible_img_dirs[0]
        os.makedirs(os.path.join(extract_dir, "IMG"), exist_ok=True)
        return img_dir
    
    # Look for image files directly
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(glob.glob(os.path.join(extract_dir, f"**/*{ext}"), recursive=True))
    
    if image_files:
        # Create IMG directory and link/copy files if needed
        os.makedirs(img_dir, exist_ok=True)
        import shutil
        
        print(f"Found {len(image_files)} images. Organizing into IMG directory...")
        for i, img_path in enumerate(image_files[:5]):  # Print first few for debugging
            print(f"  Example image: {img_path}")
        
        # Copy images to IMG directory if they're not already in it
        img_count = 0
        for i, img_path in enumerate(image_files):
            if img_dir not in img_path:
                dest_path = os.path.join(img_dir, f"image_{i:05d}{os.path.splitext(img_path)[1]}")
                shutil.copy(img_path, dest_path)
                img_count += 1
        
        if img_count > 0:
            print(f"Copied {img_count} images to {img_dir}")
        
        return img_dir
    
    print("No images found in the extracted dataset")
    return None 