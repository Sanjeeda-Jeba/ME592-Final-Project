#!/usr/bin/env python
"""
Script to fix the dataset CSV file by updating image paths and ensuring proper column structure.
"""

import os
import pandas as pd

def fix_csv(csv_path, img_dir):
    """
    Fix the CSV file by updating image paths and ensuring proper column structure.
    
    Args:
        csv_path: Path to the original CSV file
        img_dir: Path to the IMG directory
    """
    print(f"Reading CSV file: {csv_path}")
    
    # Read the CSV file
    df = pd.read_csv(csv_path, header=None)
    
    # Check if the dataframe has the expected number of columns
    if df.shape[1] != 7:
        print(f"Warning: CSV has {df.shape[1]} columns instead of the expected 7")
        # If fewer than 7 columns, add missing ones
        while df.shape[1] < 7:
            df[df.shape[1]] = 0.0  # Add default column with zeros
    
    # Assign column names
    df.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    
    # Get the absolute path to the IMG directory
    img_dir_abs = os.path.abspath(img_dir)
    
    # Function to extract filename from path and join with local IMG directory
    def fix_path(path):
        filename = os.path.basename(path.strip())
        return os.path.join(img_dir_abs, filename)
    
    # Fix image paths
    df['center'] = df['center'].apply(fix_path)
    df['left'] = df['left'].apply(fix_path)
    df['right'] = df['right'].apply(fix_path)
    
    # Ensure all control values are float
    for col in ['steering', 'throttle', 'brake', 'speed']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN values
    orig_len = len(df)
    df = df.dropna()
    if len(df) < orig_len:
        print(f"Dropped {orig_len - len(df)} rows with NaN values")
    
    # Create a backup of the original CSV
    backup_path = csv_path + '.backup'
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy(csv_path, backup_path)
        print(f"Created backup of original CSV at {backup_path}")
    
    # Save the fixed CSV
    fixed_path = csv_path
    df.to_csv(fixed_path, index=False)
    print(f"Fixed CSV saved to {fixed_path}")
    print(f"Number of entries: {len(df)}")
    
    # Print a few sample rows
    print("\nSample rows:")
    print(df.head())
    
    return fixed_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix dataset CSV file")
    parser.add_argument("--csv_path", type=str, default="data/archive/driving_log.csv",
                        help="Path to the CSV file to fix")
    parser.add_argument("--img_dir", type=str, default="data/archive/IMG",
                        help="Path to the IMG directory")
    
    args = parser.parse_args()
    fix_csv(args.csv_path, args.img_dir) 