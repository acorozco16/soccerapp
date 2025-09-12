#!/usr/bin/env python3
"""
Data augmentation pipeline to improve YOLO model robustness
"""

import cv2
import numpy as np
from pathlib import Path
import yaml
import random
import shutil
from typing import List, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

class YOLODataAugmenter:
    def __init__(self, dataset_path: str, output_path: str):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.images_dir = self.dataset_path / "images"
        self.labels_dir = self.dataset_path / "labels"
        
        # Create output directories
        self.output_images_train = self.output_path / "images" / "train"
        self.output_images_val = self.output_path / "images" / "val"
        self.output_labels_train = self.output_path / "labels" / "train"
        self.output_labels_val = self.output_path / "labels" / "val"
        
        for dir_path in [self.output_images_train, self.output_images_val, 
                        self.output_labels_train, self.output_labels_val]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Define augmentation pipeline
        self.transform = A.Compose([
            # Rotation augmentations (key for your use case)
            A.Rotate(limit=45, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
            
            # Lighting/contrast augmentations
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
            
            # Blur and noise (simulate motion blur)
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.2)
            ], p=0.4),
            
            # Noise
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            
            # Perspective changes
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            
            # Flip augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def load_yolo_annotations(self, label_path: Path) -> List[Tuple[int, float, float, float, float]]:
        """Load YOLO format annotations"""
        annotations = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        annotations.append((class_id, x_center, y_center, width, height))
        return annotations

    def save_yolo_annotations(self, annotations: List[Tuple[int, float, float, float, float]], 
                            output_path: Path):
        """Save YOLO format annotations"""
        with open(output_path, 'w') as f:
            for class_id, x_center, y_center, width, height in annotations:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def augment_image_and_labels(self, image_path: Path, label_path: Path, 
                               output_image_path: Path, output_label_path: Path,
                               num_augmentations: int = 5):
        """Apply augmentations to a single image and its labels"""
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        annotations = self.load_yolo_annotations(label_path)
        
        # Convert YOLO to albumentations format
        bboxes = []
        class_labels = []
        for class_id, x_center, y_center, width, height in annotations:
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(class_id)
        
        # Copy original
        original_output_image = output_image_path.parent / f"{output_image_path.stem}_orig{output_image_path.suffix}"
        original_output_label = output_label_path.parent / f"{output_label_path.stem}_orig{output_label_path.suffix}"
        
        cv2.imwrite(str(original_output_image), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        self.save_yolo_annotations(annotations, original_output_label)
        
        # Generate augmentations
        for i in range(num_augmentations):
            try:
                # Apply augmentation
                if bboxes:  # Only augment if there are bboxes
                    transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    augmented_image = transformed['image']
                    augmented_bboxes = transformed['bboxes']
                    augmented_labels = transformed['class_labels']
                else:
                    # Just augment image if no bboxes
                    transformed = A.Compose([
                        A.Rotate(limit=45, p=0.7),
                        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
                        A.HorizontalFlip(p=0.5),
                    ])(image=image)
                    augmented_image = transformed['image']
                    augmented_bboxes = []
                    augmented_labels = []
                
                # Save augmented image
                aug_image_path = output_image_path.parent / f"{output_image_path.stem}_aug{i}{output_image_path.suffix}"
                cv2.imwrite(str(aug_image_path), cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
                
                # Save augmented labels
                aug_label_path = output_label_path.parent / f"{output_label_path.stem}_aug{i}{output_label_path.suffix}"
                augmented_annotations = []
                for j, (x_center, y_center, width, height) in enumerate(augmented_bboxes):
                    if j < len(augmented_labels):
                        augmented_annotations.append((augmented_labels[j], x_center, y_center, width, height))
                
                self.save_yolo_annotations(augmented_annotations, aug_label_path)
                
            except Exception as e:
                print(f"Warning: Augmentation {i} failed for {image_path}: {e}")
                continue

    def augment_dataset(self, augmentations_per_image: int = 5):
        """Augment the entire dataset"""
        
        print(f"🔄 Starting data augmentation...")
        print(f"   Source: {self.dataset_path}")
        print(f"   Output: {self.output_path}")
        print(f"   Augmentations per image: {augmentations_per_image}")
        
        # Process train images
        train_images_dir = self.images_dir / "train"
        train_labels_dir = self.labels_dir / "train"
        
        if train_images_dir.exists():
            train_images = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png"))
            print(f"   Found {len(train_images)} training images")
            
            for img_path in train_images:
                label_path = train_labels_dir / f"{img_path.stem}.txt"
                output_img = self.output_images_train / img_path.name
                output_label = self.output_labels_train / f"{img_path.stem}.txt"
                
                self.augment_image_and_labels(
                    img_path, label_path, output_img, output_label, 
                    augmentations_per_image
                )
        
        # Process validation images
        val_images_dir = self.images_dir / "val"
        val_labels_dir = self.labels_dir / "val"
        
        if val_images_dir.exists():
            val_images = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png"))
            print(f"   Found {len(val_images)} validation images")
            
            for img_path in val_images:
                label_path = val_labels_dir / f"{img_path.stem}.txt"
                output_img = self.output_images_val / img_path.name
                output_label = self.output_labels_val / f"{img_path.stem}.txt"
                
                # Fewer augmentations for validation to prevent overfitting
                self.augment_image_and_labels(
                    img_path, label_path, output_img, output_label, 
                    max(1, augmentations_per_image // 3)
                )
        
        # Copy dataset.yaml with updated paths
        self.create_dataset_yaml()
        
        # Count final dataset
        final_train_count = len(list(self.output_images_train.glob("*.jpg"))) + len(list(self.output_images_train.glob("*.png")))
        final_val_count = len(list(self.output_images_val.glob("*.jpg"))) + len(list(self.output_images_val.glob("*.png")))
        
        print(f"✅ Augmentation complete!")
        print(f"   Training images: {final_train_count}")
        print(f"   Validation images: {final_val_count}")
        print(f"   Total dataset size: {final_train_count + final_val_count}")

    def create_dataset_yaml(self):
        """Create dataset.yaml for the augmented dataset"""
        dataset_config = {
            'path': str(self.output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {0: 'ball'},
            'nc': 1
        }
        
        yaml_path = self.output_path / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"✅ Dataset config saved: {yaml_path}")

def main():
    print("🚀 Soccer Ball Detection - Data Augmentation Pipeline")
    print("=" * 55)
    
    # Paths
    original_dataset = "training_data/datasets/real_ball_dataset"
    augmented_dataset = "training_data/datasets/real_ball_dataset_augmented"
    
    # Check if albumentations is installed
    try:
        import albumentations
        print("✅ Albumentations library found")
    except ImportError:
        print("❌ Installing albumentations library...")
        os.system("python3.12 -m pip install albumentations --break-system-packages")
        import albumentations
    
    # Create augmenter
    augmenter = YOLODataAugmenter(original_dataset, augmented_dataset)
    
    # Run augmentation
    augmenter.augment_dataset(augmentations_per_image=8)  # More aggressive augmentation
    
    print("\n🎯 Next Steps:")
    print("1. Review augmented images in training_data/datasets/real_ball_dataset_augmented/")
    print("2. Train new model with: python3.12 train_augmented_model.py")
    print("3. Test improved model performance")

if __name__ == "__main__":
    main()