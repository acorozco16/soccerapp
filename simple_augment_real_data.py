#!/usr/bin/env python3
"""
Simple data augmentation for real ball dataset using OpenCV
"""

import cv2
import numpy as np
from pathlib import Path
import random
import shutil

def augment_real_dataset():
    """Apply simple augmentations to real dataset"""
    
    print("🚀 Augmenting Real Ball Dataset")
    print("=" * 35)
    
    # Paths
    source_dir = Path("training_data/real_dataset")
    output_dir = Path("training_data/real_dataset_augmented")
    
    # Create output structure
    output_images = output_dir / "images"
    output_labels = output_dir / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    # Copy original files first
    source_images = source_dir / "images"
    source_labels = source_dir / "labels"
    
    if not source_images.exists():
        print("❌ Source images directory not found")
        return
    
    print(f"📁 Processing images from: {source_images}")
    
    # Get all images
    image_files = list(source_images.glob("*.jpg"))
    print(f"📊 Found {len(image_files)} images")
    
    augmented_count = 0
    
    for img_path in image_files:
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
            
        # Find corresponding label
        label_path = source_labels / f"{img_path.stem}.txt"
        
        # Copy original
        shutil.copy2(img_path, output_images / img_path.name)
        if label_path.exists():
            shutil.copy2(label_path, output_labels / f"{img_path.stem}.txt")
        
        # Generate augmentations
        augmentations = [
            ("rot90", rotate_90),
            ("rot180", rotate_180),
            ("rot270", rotate_270),
            ("bright", increase_brightness),
            ("dark", decrease_brightness),
            ("contrast", increase_contrast),
            ("blur", add_blur),
            ("flip_h", flip_horizontal),
        ]
        
        for aug_name, aug_func in augmentations:
            try:
                # Apply augmentation
                aug_image = aug_func(image.copy())
                
                # Save augmented image
                aug_img_name = f"{img_path.stem}_{aug_name}.jpg"
                cv2.imwrite(str(output_images / aug_img_name), aug_image)
                
                # Handle labels for rotations and flips
                if label_path.exists():
                    aug_label_name = f"{img_path.stem}_{aug_name}.txt"
                    if aug_name in ["rot90", "rot180", "rot270", "flip_h"]:
                        transform_labels(label_path, output_labels / aug_label_name, aug_name)
                    else:
                        # Copy labels unchanged for brightness/contrast/blur
                        shutil.copy2(label_path, output_labels / aug_label_name)
                
                augmented_count += 1
                
            except Exception as e:
                print(f"⚠️ Failed to augment {img_path.name} with {aug_name}: {e}")
    
    # Create dataset.yaml
    create_yaml(output_dir)
    
    # Count results
    final_images = len(list(output_images.glob("*.jpg")))
    final_labels = len(list(output_labels.glob("*.txt")))
    
    print(f"✅ Augmentation complete!")
    print(f"   Original images: {len(image_files)}")
    print(f"   Augmented images: {final_images}")
    print(f"   Labels created: {final_labels}")
    print(f"   Dataset expanded by: {final_images / len(image_files):.1f}x")

def rotate_90(image):
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

def rotate_180(image):
    return cv2.rotate(image, cv2.ROTATE_180)

def rotate_270(image):
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def increase_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.add(hsv[:,:,2], 30)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def decrease_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.subtract(hsv[:,:,2], 30)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def increase_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.multiply(lab[:,:,0], 1.3)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def add_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def flip_horizontal(image):
    return cv2.flip(image, 1)

def transform_labels(input_label_path, output_label_path, transform_type):
    """Transform YOLO labels for geometric transformations"""
    
    if not input_label_path.exists():
        return
    
    with open(input_label_path, 'r') as f:
        lines = f.readlines()
    
    transformed_lines = []
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            class_id = parts[0]
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Transform coordinates
            if transform_type == "rot90":
                # 90° clockwise: (x,y) -> (1-y, x)
                new_x = 1.0 - y_center
                new_y = x_center
                new_width = height
                new_height = width
            elif transform_type == "rot180":
                # 180°: (x,y) -> (1-x, 1-y)
                new_x = 1.0 - x_center
                new_y = 1.0 - y_center
                new_width = width
                new_height = height
            elif transform_type == "rot270":
                # 270° clockwise: (x,y) -> (y, 1-x)
                new_x = y_center
                new_y = 1.0 - x_center
                new_width = height
                new_height = width
            elif transform_type == "flip_h":
                # Horizontal flip: (x,y) -> (1-x, y)
                new_x = 1.0 - x_center
                new_y = y_center
                new_width = width
                new_height = height
            else:
                # No transformation needed
                new_x, new_y = x_center, y_center
                new_width, new_height = width, height
            
            transformed_lines.append(f"{class_id} {new_x:.6f} {new_y:.6f} {new_width:.6f} {new_height:.6f}\n")
    
    with open(output_label_path, 'w') as f:
        f.writelines(transformed_lines)

def create_yaml(output_dir):
    """Create dataset.yaml for augmented dataset"""
    
    yaml_content = f"""path: {output_dir.absolute()}
train: images
val: images  # Use same images for validation in small dataset

names:
  0: ball

nc: 1
"""
    
    with open(output_dir / "dataset.yaml", 'w') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    augment_real_dataset()