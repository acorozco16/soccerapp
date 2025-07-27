#!/usr/bin/env python3
"""
Create YOLO dataset directly from our processed data
"""

import os
import shutil
import yaml
from pathlib import Path

def create_yolo_dataset():
    """Create YOLO dataset structure"""
    
    # Paths
    source_images = Path("training_data/processed_dataset/images/high_quality")
    source_labels = Path("training_data/processed_dataset/labels")
    
    # Create YOLO dataset structure
    yolo_dir = Path("training_data/datasets/yolo_soccer")
    train_images = yolo_dir / "images" / "train"
    train_labels = yolo_dir / "labels" / "train"
    val_images = yolo_dir / "images" / "val"  
    val_labels = yolo_dir / "labels" / "val"
    
    # Create directories
    for dir_path in [train_images, train_labels, val_images, val_labels]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = list(source_images.glob("*.jpg"))
    print(f"Found {len(image_files)} images")
    
    # Split 80/20 train/val
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Copy training files
    for img_file in train_files:
        label_file = source_labels / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.copy2(img_file, train_images / img_file.name)
            shutil.copy2(label_file, train_labels / label_file.name)
    
    # Copy validation files
    for img_file in val_files:
        label_file = source_labels / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.copy2(img_file, val_images / img_file.name)
            shutil.copy2(label_file, val_labels / label_file.name)
    
    # Create YOLO config file
    config = {
        'path': str(yolo_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['soccer_ball']
    }
    
    with open(yolo_dir / "dataset.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"YOLO dataset created:")
    print(f"- Training images: {len(train_files)}")
    print(f"- Validation images: {len(val_files)}")
    print(f"- Config: {yolo_dir / 'dataset.yaml'}")
    
    return str(yolo_dir / "dataset.yaml")

if __name__ == "__main__":
    config_path = create_yolo_dataset()
    print(f"\nYOLO dataset ready at: {config_path}")