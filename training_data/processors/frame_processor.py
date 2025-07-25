#!/usr/bin/env python3
"""
Frame Processing and Automated Labeling System
Processes scraped frames and creates bounding box labels for YOLO training
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    x_center: float  # Normalized 0-1
    y_center: float  # Normalized 0-1
    width: float     # Normalized 0-1
    height: float    # Normalized 0-1
    confidence: float
    class_id: int = 0  # 0 for soccer ball
    
    def to_yolo_format(self) -> str:
        """Convert to YOLO annotation format"""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"
    
    def to_pixel_coords(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert normalized coords to pixel coordinates (x1, y1, x2, y2)"""
        x1 = int((self.x_center - self.width/2) * img_width)
        y1 = int((self.y_center - self.height/2) * img_height)
        x2 = int((self.x_center + self.width/2) * img_width)
        y2 = int((self.y_center + self.height/2) * img_height)
        return x1, y1, x2, y2


@dataclass
class FrameAnnotation:
    image_path: str
    image_hash: str
    width: int
    height: int
    bounding_boxes: List[BoundingBox]
    quality_category: str  # 'high', 'medium', 'low'
    lighting_condition: str  # 'good', 'poor', 'mixed'
    ball_visibility: str   # 'clear', 'partial', 'difficult'
    created_at: datetime
    
    def to_dict(self) -> Dict:
        return {
            'image_path': self.image_path,
            'image_hash': self.image_hash,
            'width': self.width,
            'height': self.height,
            'bounding_boxes': [asdict(bb) for bb in self.bounding_boxes],
            'quality_category': self.quality_category,
            'lighting_condition': self.lighting_condition,
            'ball_visibility': self.ball_visibility,
            'created_at': self.created_at.isoformat()
        }


class FrameProcessor:
    def __init__(self, scraped_data_dir: str, output_dir: str = "./processed_dataset"):
        self.scraped_data_dir = Path(scraped_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output subdirectories
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.metadata_dir = self.output_dir / "metadata"
        self.duplicates_dir = self.output_dir / "duplicates"
        
        for dir_path in [self.images_dir, self.labels_dir, self.metadata_dir, self.duplicates_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Quality categories
        self.quality_dirs = {
            'high': self.images_dir / 'high_quality',
            'medium': self.images_dir / 'medium_quality', 
            'low': self.images_dir / 'low_quality'
        }
        
        for dir_path in self.quality_dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        # Track processed images and duplicates
        self.processed_hashes: Set[str] = set()
        self.annotations: List[FrameAnnotation] = []
        
        # Load existing processed data
        self._load_existing_data()
        
        # Ball detection parameters (enhanced from main app)
        self.ball_color_ranges = {
            "orange_clean": {
                "lower": np.array([5, 100, 100]),
                "upper": np.array([15, 255, 255])
            },
            "orange_muddy": {
                "lower": np.array([8, 50, 80]),
                "upper": np.array([25, 200, 180])
            },
            "white_clean": {
                "lower": np.array([0, 0, 200]),
                "upper": np.array([180, 30, 255])
            },
            "white_dirty": {
                "lower": np.array([0, 0, 150]),
                "upper": np.array([180, 50, 230])
            }
        }
        
        # Hough circle parameters
        self.hough_params = [
            {"dp": 1, "minDist": 30, "param1": 50, "param2": 30, "minRadius": 8, "maxRadius": 60},
            {"dp": 1, "minDist": 25, "param1": 30, "param2": 20, "minRadius": 10, "maxRadius": 50},
        ]
    
    def _load_existing_data(self):
        """Load existing processed annotations"""
        annotations_file = self.metadata_dir / "annotations.json"
        if annotations_file.exists():
            try:
                with open(annotations_file, 'r') as f:
                    data = json.load(f)
                    for item in data.get('annotations', []):
                        # Reconstruct BoundingBox objects
                        bboxes = []
                        for bb_data in item.get('bounding_boxes', []):
                            bbox = BoundingBox(**bb_data)
                            bboxes.append(bbox)
                        
                        annotation = FrameAnnotation(
                            image_path=item['image_path'],
                            image_hash=item['image_hash'],
                            width=item['width'],
                            height=item['height'],
                            bounding_boxes=bboxes,
                            quality_category=item['quality_category'],
                            lighting_condition=item['lighting_condition'],
                            ball_visibility=item['ball_visibility'],
                            created_at=datetime.fromisoformat(item['created_at'])
                        )
                        self.annotations.append(annotation)
                        self.processed_hashes.add(item['image_hash'])
                
                logger.info(f"Loaded {len(self.annotations)} existing annotations")
            except Exception as e:
                logger.error(f"Error loading existing annotations: {e}")
    
    def _save_annotations(self):
        """Save current annotations"""
        data = {
            'annotations': [ann.to_dict() for ann in self.annotations],
            'total_count': len(self.annotations),
            'last_updated': datetime.now().isoformat()
        }
        
        annotations_file = self.metadata_dir / "annotations.json"
        with open(annotations_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def calculate_image_hash(self, image_path: str) -> str:
        """Calculate hash of image for duplicate detection"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                return hashlib.md5(image_data).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {image_path}: {e}")
            return ""
    
    def assess_image_quality(self, image: np.ndarray) -> Tuple[str, str, float]:
        """Assess image quality and categorize"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Brightness assessment
        brightness = np.mean(gray)
        
        # Contrast assessment
        contrast = np.std(gray)
        
        # Blur assessment using Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Lighting condition
        if 50 <= brightness <= 200 and contrast > 30:
            lighting = "good"
        elif brightness < 50 or brightness > 200:
            lighting = "poor"
        else:
            lighting = "mixed"
        
        # Overall quality score
        brightness_score = 1 - abs(brightness - 128) / 128
        contrast_score = min(contrast / 60, 1.0)
        blur_score_norm = min(blur_score / 150, 1.0)
        
        overall_score = (brightness_score + contrast_score + blur_score_norm) / 3
        
        # Quality category
        if overall_score > 0.75:
            quality = "high"
        elif overall_score > 0.5:
            quality = "medium"
        else:
            quality = "low"
        
        return quality, lighting, overall_score
    
    def detect_soccer_ball(self, image: np.ndarray) -> List[BoundingBox]:
        """Detect soccer ball and create bounding boxes"""
        bounding_boxes = []
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        
        # Try different color ranges
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for color_name, ranges in self.ball_color_ranges.items():
            color_mask = cv2.inRange(hsv, ranges["lower"], ranges["upper"])
            combined_mask = cv2.bitwise_or(combined_mask, color_mask)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Method 1: Hough Circle Detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(combined_mask, (9, 9), 2)
        
        for params in self.hough_params:
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                **params
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :2]:  # Take up to 2 best circles
                    x, y, r = circle
                    
                    # Calculate confidence based on mask overlap
                    circle_mask = np.zeros_like(combined_mask)
                    cv2.circle(circle_mask, (x, y), r, 255, -1)
                    overlap = cv2.bitwise_and(combined_mask, circle_mask)
                    confidence = np.sum(overlap) / (np.pi * r * r * 255) if r > 0 else 0
                    
                    if confidence > 0.3:  # Minimum confidence threshold
                        # Convert to normalized YOLO format
                        x_center = x / w
                        y_center = y / h
                        width = (2 * r) / w
                        height = (2 * r) / h
                        
                        bbox = BoundingBox(
                            x_center=x_center,
                            y_center=y_center,
                            width=width,
                            height=height,
                            confidence=confidence
                        )
                        bounding_boxes.append(bbox)
        
        # Method 2: Contour Detection for irregular shapes
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 8000:  # Reasonable ball size
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > 0.4:  # Reasonably circular
                    # Get bounding rectangle
                    x, y, rect_w, rect_h = cv2.boundingRect(contour)
                    
                    # Convert to normalized YOLO format
                    x_center = (x + rect_w/2) / w
                    y_center = (y + rect_h/2) / h
                    width = rect_w / w
                    height = rect_h / h
                    
                    confidence = circularity * 0.7  # Lower confidence for contour
                    
                    bbox = BoundingBox(
                        x_center=x_center,
                        y_center=y_center,
                        width=width,
                        height=height,
                        confidence=confidence
                    )
                    
                    # Check if this bbox is too similar to existing ones
                    is_duplicate = False
                    for existing_bbox in bounding_boxes:
                        if (abs(bbox.x_center - existing_bbox.x_center) < 0.1 and
                            abs(bbox.y_center - existing_bbox.y_center) < 0.1):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        bounding_boxes.append(bbox)
        
        # Filter and return best detections
        bounding_boxes.sort(key=lambda x: x.confidence, reverse=True)
        return bounding_boxes[:3]  # Return top 3 detections
    
    def assess_ball_visibility(self, bounding_boxes: List[BoundingBox]) -> str:
        """Assess ball visibility category"""
        if not bounding_boxes:
            return "none"
        
        best_confidence = max(bbox.confidence for bbox in bounding_boxes)
        
        if best_confidence > 0.7:
            return "clear"
        elif best_confidence > 0.4:
            return "partial"
        else:
            return "difficult"
    
    def process_frame(self, frame_path: str) -> Optional[FrameAnnotation]:
        """Process a single frame and create annotation"""
        try:
            # Calculate hash first
            image_hash = self.calculate_image_hash(frame_path)
            
            # Skip if already processed
            if image_hash in self.processed_hashes:
                return None
            
            # Load image
            image = cv2.imread(frame_path)
            if image is None:
                logger.error(f"Cannot load image: {frame_path}")
                return None
            
            h, w = image.shape[:2]
            
            # Assess quality
            quality, lighting, quality_score = self.assess_image_quality(image)
            
            # Skip very low quality images
            if quality_score < 0.3:
                logger.debug(f"Skipping low quality image: {frame_path}")
                return None
            
            # Detect ball
            bounding_boxes = self.detect_soccer_ball(image)
            
            # Skip frames with no ball detected
            if not bounding_boxes:
                return None
            
            # Assess ball visibility
            ball_visibility = self.assess_ball_visibility(bounding_boxes)
            
            # Create annotation
            annotation = FrameAnnotation(
                image_path=frame_path,
                image_hash=image_hash,
                width=w,
                height=h,
                bounding_boxes=bounding_boxes,
                quality_category=quality,
                lighting_condition=lighting,
                ball_visibility=ball_visibility,
                created_at=datetime.now()
            )
            
            # Mark as processed
            self.processed_hashes.add(image_hash)
            
            return annotation
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_path}: {e}")
            return None
    
    def copy_and_organize_image(self, annotation: FrameAnnotation) -> str:
        """Copy image to organized directory structure"""
        # Determine target directory based on quality
        target_dir = self.quality_dirs[annotation.quality_category]
        
        # Create unique filename using hash
        original_path = Path(annotation.image_path)
        new_filename = f"{annotation.image_hash}{original_path.suffix}"
        new_path = target_dir / new_filename
        
        # Copy image
        import shutil
        shutil.copy2(annotation.image_path, new_path)
        
        return str(new_path)
    
    def create_yolo_label_file(self, annotation: FrameAnnotation, image_path: str):
        """Create YOLO format label file"""
        label_filename = Path(image_path).stem + ".txt"
        
        # Determine label directory based on quality
        if annotation.quality_category == "high":
            label_dir = self.labels_dir / "high_quality"
        elif annotation.quality_category == "medium":
            label_dir = self.labels_dir / "medium_quality"
        else:
            label_dir = self.labels_dir / "low_quality"
        
        label_dir.mkdir(exist_ok=True)
        label_path = label_dir / label_filename
        
        # Write YOLO annotations
        with open(label_path, 'w') as f:
            for bbox in annotation.bounding_boxes:
                f.write(bbox.to_yolo_format() + "\n")
    
    def create_visualization(self, annotation: FrameAnnotation, image_path: str):
        """Create visualization with bounding boxes"""
        try:
            image = cv2.imread(annotation.image_path)
            if image is None:
                return
            
            h, w = image.shape[:2]
            
            # Draw bounding boxes
            for bbox in annotation.bounding_boxes:
                x1, y1, x2, y2 = bbox.to_pixel_coords(w, h)
                
                # Choose color based on confidence
                if bbox.confidence > 0.7:
                    color = (0, 255, 0)  # Green for high confidence
                elif bbox.confidence > 0.4:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 0, 255)  # Red for low confidence
                
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"{bbox.confidence:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Save visualization
            viz_dir = self.output_dir / "visualizations" / annotation.quality_category
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            viz_filename = f"{annotation.image_hash}_viz.jpg"
            viz_path = viz_dir / viz_filename
            cv2.imwrite(str(viz_path), image)
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
    
    def process_all_frames(self, max_frames: Optional[int] = None) -> Dict:
        """Process all frames from scraped data"""
        logger.info("Starting frame processing...")
        
        # Find all frame files
        frame_files = []
        frames_dir = self.scraped_data_dir / "frames"
        
        if frames_dir.exists():
            for video_dir in frames_dir.iterdir():
                if video_dir.is_dir():
                    for frame_file in video_dir.glob("*.jpg"):
                        frame_files.append(str(frame_file))
        
        if max_frames:
            frame_files = frame_files[:max_frames]
        
        logger.info(f"Found {len(frame_files)} frames to process")
        
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        for frame_path in tqdm(frame_files, desc="Processing frames"):
            try:
                annotation = self.process_frame(frame_path)
                
                if annotation:
                    # Copy and organize image
                    new_image_path = self.copy_and_organize_image(annotation)
                    
                    # Update annotation with new path
                    annotation.image_path = new_image_path
                    
                    # Create YOLO label file
                    self.create_yolo_label_file(annotation, new_image_path)
                    
                    # Create visualization
                    self.create_visualization(annotation, new_image_path)
                    
                    # Add to annotations list
                    self.annotations.append(annotation)
                    processed_count += 1
                    
                    # Save periodically
                    if processed_count % 100 == 0:
                        self._save_annotations()
                        logger.info(f"Processed {processed_count} frames...")
                
                else:
                    skipped_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing {frame_path}: {e}")
                error_count += 1
        
        # Final save
        self._save_annotations()
        
        # Generate summary
        summary = self.generate_dataset_summary()
        summary.update({
            'processing_stats': {
                'total_frames_found': len(frame_files),
                'frames_processed': processed_count,
                'frames_skipped': skipped_count,
                'errors': error_count
            }
        })
        
        # Save processing summary
        summary_path = self.metadata_dir / f"processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Frame processing complete! Summary saved to {summary_path}")
        return summary
    
    def generate_dataset_summary(self) -> Dict:
        """Generate comprehensive dataset summary"""
        if not self.annotations:
            return {"total_annotations": 0}
        
        # Quality distribution
        quality_dist = {"high": 0, "medium": 0, "low": 0}
        lighting_dist = {"good": 0, "poor": 0, "mixed": 0}
        visibility_dist = {"clear": 0, "partial": 0, "difficult": 0}
        
        total_bboxes = 0
        confidence_scores = []
        
        for ann in self.annotations:
            quality_dist[ann.quality_category] += 1
            lighting_dist[ann.lighting_condition] += 1
            visibility_dist[ann.ball_visibility] += 1
            
            total_bboxes += len(ann.bounding_boxes)
            for bbox in ann.bounding_boxes:
                confidence_scores.append(bbox.confidence)
        
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        return {
            "total_annotations": len(self.annotations),
            "total_bounding_boxes": total_bboxes,
            "average_confidence": avg_confidence,
            "quality_distribution": quality_dist,
            "lighting_distribution": lighting_dist,
            "visibility_distribution": visibility_dist,
            "confidence_stats": {
                "mean": float(np.mean(confidence_scores)) if confidence_scores else 0,
                "std": float(np.std(confidence_scores)) if confidence_scores else 0,
                "min": float(np.min(confidence_scores)) if confidence_scores else 0,
                "max": float(np.max(confidence_scores)) if confidence_scores else 0
            }
        }


def main():
    """Example usage"""
    processor = FrameProcessor(
        scraped_data_dir="./training_data/scraped_data",
        output_dir="./training_data/processed_dataset"
    )
    
    try:
        summary = processor.process_all_frames(max_frames=1000)
        
        print("\n=== PROCESSING SUMMARY ===")
        print(f"Total annotations: {summary['total_annotations']}")
        print(f"Total bounding boxes: {summary['total_bounding_boxes']}")
        print(f"Average confidence: {summary['average_confidence']:.3f}")
        print(f"Quality distribution: {summary['quality_distribution']}")
        print(f"Processing stats: {summary['processing_stats']}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")


if __name__ == "__main__":
    main()