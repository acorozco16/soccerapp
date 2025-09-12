#!/usr/bin/env python3
"""
Dataset Management and YOLO Export System
Manages training datasets, creates train/val splits, and exports in YOLO format
"""

import os
import json
import shutil
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetVersion:
    version: str
    created_at: datetime
    total_images: int
    train_images: int
    val_images: int
    test_images: int
    quality_distribution: Dict[str, int]
    diversity_metrics: Dict[str, float]
    description: str
    path: str


@dataclass
class DatasetStats:
    total_images: int
    total_labels: int
    class_distribution: Dict[str, int]
    quality_distribution: Dict[str, int]
    lighting_distribution: Dict[str, int]
    visibility_distribution: Dict[str, int]
    avg_boxes_per_image: float
    bbox_size_stats: Dict[str, float]
    diversity_score: float


class DatasetManager:
    def __init__(self, processed_data_dir: str, datasets_dir: str = "./datasets"):
        self.processed_data_dir = Path(processed_data_dir)
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Version control
        self.versions_file = self.datasets_dir / "versions.json"
        self.versions: List[DatasetVersion] = []
        self._load_versions()
        
        # Class names for YOLO
        self.class_names = ['soccer_ball']
        
        # Quality weights for sampling
        self.quality_weights = {
            'high': 1.0,
            'medium': 0.7,
            'low': 0.3
        }
    
    def _load_versions(self):
        """Load existing dataset versions"""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
                    data = json.load(f)
                    for version_data in data.get('versions', []):
                        version = DatasetVersion(
                            version=version_data['version'],
                            created_at=datetime.fromisoformat(version_data['created_at']),
                            total_images=version_data['total_images'],
                            train_images=version_data['train_images'],
                            val_images=version_data['val_images'],
                            test_images=version_data['test_images'],
                            quality_distribution=version_data['quality_distribution'],
                            diversity_metrics=version_data['diversity_metrics'],
                            description=version_data['description'],
                            path=version_data['path']
                        )
                        self.versions.append(version)
                logger.info(f"Loaded {len(self.versions)} dataset versions")
            except Exception as e:
                logger.error(f"Error loading versions: {e}")
    
    def _save_versions(self):
        """Save dataset versions"""
        data = {
            'versions': [asdict(v) for v in self.versions],
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.versions_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def analyze_processed_data(self) -> DatasetStats:
        """Analyze processed data and compute statistics"""
        logger.info("Analyzing processed data...")
        
        # Load annotations
        annotations_file = self.processed_data_dir / "metadata" / "annotations.json"
        if not annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
        
        with open(annotations_file, 'r') as f:
            data = json.load(f)
            annotations = data.get('annotations', [])
        
        if not annotations:
            raise ValueError("No annotations found")
        
        # Initialize counters
        total_images = len(annotations)
        total_labels = 0
        quality_dist = {'high': 0, 'medium': 0, 'low': 0}
        lighting_dist = {'good': 0, 'poor': 0, 'mixed': 0}
        visibility_dist = {'clear': 0, 'partial': 0, 'difficult': 0}
        
        bbox_sizes = []
        boxes_per_image = []
        
        # Analyze each annotation
        for ann in annotations:
            # Quality distribution
            quality_dist[ann['quality_category']] += 1
            lighting_dist[ann['lighting_condition']] += 1
            visibility_dist[ann['ball_visibility']] += 1
            
            # Bounding box statistics
            bboxes = ann['bounding_boxes']
            total_labels += len(bboxes)
            boxes_per_image.append(len(bboxes))
            
            for bbox in bboxes:
                # Calculate bbox area (normalized)
                area = bbox['width'] * bbox['height']
                bbox_sizes.append(area)
        
        # Calculate diversity score
        diversity_score = self._calculate_diversity_score(
            quality_dist, lighting_dist, visibility_dist
        )
        
        # Compute statistics
        stats = DatasetStats(
            total_images=total_images,
            total_labels=total_labels,
            class_distribution={'soccer_ball': total_labels},
            quality_distribution=quality_dist,
            lighting_distribution=lighting_dist,
            visibility_distribution=visibility_dist,
            avg_boxes_per_image=np.mean(boxes_per_image) if boxes_per_image else 0,
            bbox_size_stats={
                'mean': float(np.mean(bbox_sizes)) if bbox_sizes else 0,
                'std': float(np.std(bbox_sizes)) if bbox_sizes else 0,
                'min': float(np.min(bbox_sizes)) if bbox_sizes else 0,
                'max': float(np.max(bbox_sizes)) if bbox_sizes else 0
            },
            diversity_score=diversity_score
        )
        
        logger.info(f"Dataset analysis complete: {total_images} images, {total_labels} labels")
        return stats
    
    def _calculate_diversity_score(self, quality_dist: Dict, lighting_dist: Dict, 
                                 visibility_dist: Dict) -> float:
        """Calculate dataset diversity score (0-1)"""
        def entropy(distribution: Dict) -> float:
            """Calculate entropy of a distribution"""
            total = sum(distribution.values())
            if total == 0:
                return 0
            
            entropy_val = 0
            for count in distribution.values():
                if count > 0:
                    p = count / total
                    entropy_val -= p * np.log2(p)
            
            return entropy_val
        
        # Calculate entropy for each dimension
        quality_entropy = entropy(quality_dist)
        lighting_entropy = entropy(lighting_dist)
        visibility_entropy = entropy(visibility_dist)
        
        # Normalize entropies (log2(3) is max entropy for 3 categories)
        max_entropy = np.log2(3)
        
        quality_norm = quality_entropy / max_entropy if max_entropy > 0 else 0
        lighting_norm = lighting_entropy / max_entropy if max_entropy > 0 else 0
        visibility_norm = visibility_entropy / max_entropy if max_entropy > 0 else 0
        
        # Overall diversity score
        diversity_score = (quality_norm + lighting_norm + visibility_norm) / 3
        
        return float(diversity_score)
    
    def create_balanced_split(self, annotations: List[Dict], train_ratio: float = 0.7, 
                            val_ratio: float = 0.2, test_ratio: float = 0.1, 
                            stratify_by: str = 'quality_category') -> Tuple[List, List, List]:
        """Create balanced train/val/test split"""
        # Group annotations by stratification key
        groups = {}
        for ann in annotations:
            key = ann[stratify_by]
            if key not in groups:
                groups[key] = []
            groups[key].append(ann)
        
        train_set, val_set, test_set = [], [], []
        
        # Split each group proportionally
        for group_name, group_annotations in groups.items():
            if len(group_annotations) < 3:
                # If too few samples, put all in training
                train_set.extend(group_annotations)
                continue
            
            # First split: train vs (val+test)
            train_group, temp_group = train_test_split(
                group_annotations, 
                train_size=train_ratio,
                random_state=42
            )
            
            # Second split: val vs test
            if len(temp_group) >= 2:
                val_size = val_ratio / (val_ratio + test_ratio)
                val_group, test_group = train_test_split(
                    temp_group,
                    train_size=val_size,
                    random_state=42
                )
            else:
                val_group = temp_group
                test_group = []
            
            train_set.extend(train_group)
            val_set.extend(val_group)
            test_set.extend(test_group)
        
        # Shuffle the final sets
        random.shuffle(train_set)
        random.shuffle(val_set)
        random.shuffle(test_set)
        
        logger.info(f"Split created: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")
        return train_set, val_set, test_set
    
    def copy_dataset_files(self, annotations: List[Dict], output_dir: Path, 
                          split_name: str) -> List[str]:
        """Copy images and labels to output directory"""
        images_dir = output_dir / split_name / "images"
        labels_dir = output_dir / split_name / "labels"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        copied_files = []
        
        for ann in tqdm(annotations, desc=f"Copying {split_name} files"):
            try:
                # Copy image
                src_image_path = Path(ann['image_path'])
                if not src_image_path.exists():
                    logger.warning(f"Image not found: {src_image_path}")
                    continue
                
                dst_image_path = images_dir / src_image_path.name
                shutil.copy2(src_image_path, dst_image_path)
                
                # Create label file
                label_filename = src_image_path.stem + ".txt"
                label_path = labels_dir / label_filename
                
                with open(label_path, 'w') as f:
                    for bbox in ann['bounding_boxes']:
                        # YOLO format: class_id x_center y_center width height
                        line = f"0 {bbox['x_center']:.6f} {bbox['y_center']:.6f} {bbox['width']:.6f} {bbox['height']:.6f}\n"
                        f.write(line)
                
                copied_files.append(str(dst_image_path))
                
            except Exception as e:
                logger.error(f"Error copying files for {ann['image_path']}: {e}")
        
        return copied_files
    
    def create_yolo_dataset(self, version: str, description: str = "", 
                           max_images: Optional[int] = None,
                           quality_filter: Optional[List[str]] = None) -> DatasetVersion:
        """Create a YOLO format dataset"""
        logger.info(f"Creating YOLO dataset version {version}")
        
        # Load annotations
        annotations_file = self.processed_data_dir / "metadata" / "annotations.json"
        with open(annotations_file, 'r') as f:
            data = json.load(f)
            all_annotations = data.get('annotations', [])
        
        # Filter annotations
        filtered_annotations = []
        for ann in all_annotations:
            # Quality filter
            if quality_filter and ann['quality_category'] not in quality_filter:
                continue
            
            # Check if image exists
            if not Path(ann['image_path']).exists():
                continue
            
            filtered_annotations.append(ann)
        
        # Limit number of images if specified
        if max_images and len(filtered_annotations) > max_images:
            # Weighted sampling based on quality
            weights = []
            for ann in filtered_annotations:
                weight = self.quality_weights.get(ann['quality_category'], 0.5)
                weights.append(weight)
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Sample without replacement
            indices = np.random.choice(
                len(filtered_annotations), 
                size=max_images, 
                replace=False, 
                p=weights
            )
            
            filtered_annotations = [filtered_annotations[i] for i in indices]
        
        logger.info(f"Using {len(filtered_annotations)} annotations for dataset {version}")
        
        # Create train/val/test split
        train_set, val_set, test_set = self.create_balanced_split(filtered_annotations)
        
        # Create output directory
        dataset_dir = self.datasets_dir / f"yolo_v{version}"
        dataset_dir.mkdir(exist_ok=True)
        
        # Copy files for each split
        train_files = self.copy_dataset_files(train_set, dataset_dir, "train")
        val_files = self.copy_dataset_files(val_set, dataset_dir, "val")
        test_files = self.copy_dataset_files(test_set, dataset_dir, "test")
        
        # Create YOLO config file
        config = {
            'path': str(dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,  # number of classes
            'names': self.class_names
        }
        
        config_path = dataset_dir / "dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Calculate quality distribution
        quality_dist = {'high': 0, 'medium': 0, 'low': 0}
        for ann in filtered_annotations:
            quality_dist[ann['quality_category']] += 1
        
        # Calculate diversity metrics
        lighting_dist = {'good': 0, 'poor': 0, 'mixed': 0}
        visibility_dist = {'clear': 0, 'partial': 0, 'difficult': 0}
        
        for ann in filtered_annotations:
            lighting_dist[ann['lighting_condition']] += 1
            visibility_dist[ann['ball_visibility']] += 1
        
        diversity_score = self._calculate_diversity_score(
            quality_dist, lighting_dist, visibility_dist
        )
        
        # Create dataset version
        dataset_version = DatasetVersion(
            version=version,
            created_at=datetime.now(),
            total_images=len(filtered_annotations),
            train_images=len(train_set),
            val_images=len(val_set),
            test_images=len(test_set),
            quality_distribution=quality_dist,
            diversity_metrics={
                'diversity_score': diversity_score,
                'lighting_entropy': self._calculate_entropy(lighting_dist),
                'visibility_entropy': self._calculate_entropy(visibility_dist)
            },
            description=description,
            path=str(dataset_dir)
        )
        
        # Save metadata
        metadata = {
            'version_info': asdict(dataset_version),
            'config': config,
            'statistics': {
                'quality_distribution': quality_dist,
                'lighting_distribution': lighting_dist,
                'visibility_distribution': visibility_dist,
                'total_bounding_boxes': sum(len(ann['bounding_boxes']) for ann in filtered_annotations)
            },
            'file_counts': {
                'train_images': len(train_files),
                'val_images': len(val_files),
                'test_images': len(test_files)
            }
        }
        
        metadata_path = dataset_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Add to versions list and save
        self.versions.append(dataset_version)
        self._save_versions()
        
        logger.info(f"YOLO dataset v{version} created successfully at {dataset_dir}")
        return dataset_version
    
    def _calculate_entropy(self, distribution: Dict) -> float:
        """Calculate entropy of a distribution"""
        total = sum(distribution.values())
        if total == 0:
            return 0
        
        entropy_val = 0
        for count in distribution.values():
            if count > 0:
                p = count / total
                entropy_val -= p * np.log2(p)
        
        return entropy_val
    
    def compare_datasets(self, version1: str, version2: str) -> Dict:
        """Compare two dataset versions"""
        v1 = next((v for v in self.versions if v.version == version1), None)
        v2 = next((v for v in self.versions if v.version == version2), None)
        
        if not v1 or not v2:
            raise ValueError(f"Dataset version not found")
        
        comparison = {
            'version1': {
                'version': v1.version,
                'total_images': v1.total_images,
                'quality_dist': v1.quality_distribution,
                'diversity_score': v1.diversity_metrics.get('diversity_score', 0)
            },
            'version2': {
                'version': v2.version,
                'total_images': v2.total_images,
                'quality_dist': v2.quality_distribution,
                'diversity_score': v2.diversity_metrics.get('diversity_score', 0)
            },
            'differences': {
                'image_count_diff': v2.total_images - v1.total_images,
                'diversity_diff': v2.diversity_metrics.get('diversity_score', 0) - v1.diversity_metrics.get('diversity_score', 0)
            }
        }
        
        return comparison
    
    def export_dataset_summary(self) -> Dict:
        """Export comprehensive dataset summary"""
        summary = {
            'total_versions': len(self.versions),
            'versions': [asdict(v) for v in self.versions],
            'latest_version': asdict(self.versions[-1]) if self.versions else None,
            'export_time': datetime.now().isoformat()
        }
        
        # Save summary
        summary_path = self.datasets_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary


def main():
    """Example usage"""
    manager = DatasetManager(
        processed_data_dir="./training_data/processed_dataset",
        datasets_dir="./training_data/datasets"
    )
    
    try:
        # Analyze current data
        stats = manager.analyze_processed_data()
        print("\n=== DATASET STATISTICS ===")
        print(f"Total images: {stats.total_images}")
        print(f"Total labels: {stats.total_labels}")
        print(f"Quality distribution: {stats.quality_distribution}")
        print(f"Diversity score: {stats.diversity_score:.3f}")
        
        # Create YOLO dataset
        version = datetime.now().strftime("%Y%m%d_%H%M")
        dataset_version = manager.create_yolo_dataset(
            version=version,
            description="Soccer ball detection dataset with balanced quality distribution",
            max_images=1000,
            quality_filter=['high', 'medium']  # Only high and medium quality
        )
        
        print(f"\n=== YOLO DATASET CREATED ===")
        print(f"Version: {dataset_version.version}")
        print(f"Total images: {dataset_version.total_images}")
        print(f"Train/Val/Test: {dataset_version.train_images}/{dataset_version.val_images}/{dataset_version.test_images}")
        print(f"Path: {dataset_version.path}")
        
        # Export summary
        summary = manager.export_dataset_summary()
        print(f"\nDataset summary exported with {summary['total_versions']} versions")
        
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")


if __name__ == "__main__":
    main()