#!/usr/bin/env python3
"""
Simple Frame Processor - Processes YouTube collected frames for training
"""

import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def process_youtube_frames():
    """Process collected YouTube frames into training format"""
    
    # Source and destination paths
    source_dir = Path("../collected_data/youtube/scraper_0/frames")
    dest_images_dir = Path("../processed_dataset/images/high_quality")
    dest_labels_dir = Path("../processed_dataset/labels") 
    metadata_dir = Path("../processed_dataset/metadata")
    
    # Create directories
    for dir_path in [dest_images_dir, dest_labels_dir, metadata_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Processing YouTube frames...")
    
    processed_count = 0
    annotations = []
    
    # Process each video directory
    for video_dir in source_dir.iterdir():
        if not video_dir.is_dir():
            continue
            
        print(f"Processing video: {video_dir.name}")
        
        # Process each frame in the video
        for frame_jpg in video_dir.glob("*.jpg"):
            frame_json = frame_jpg.with_suffix('.json')
            
            if not frame_json.exists():
                continue
                
            # Load annotation
            with open(frame_json, 'r') as f:
                annotation = json.load(f)
            
            # Skip frames with low ball visibility
            if annotation.get('ball_visibility_score', 0) < 0.3:
                continue
                
            # Copy image to training directory
            dest_image = dest_images_dir / f"{video_dir.name}_{frame_jpg.name}"
            shutil.copy2(frame_jpg, dest_image)
            
            # Create YOLO format label (assuming ball is centered for now)
            label_file = dest_labels_dir / f"{video_dir.name}_{frame_jpg.stem}.txt"
            with open(label_file, 'w') as f:
                # YOLO format: class_id center_x center_y width height (all normalized 0-1)
                # For now, assume ball is in center with small bounding box
                f.write("0 0.5 0.5 0.1 0.1\n")  # class 0 (ball), center position, 10% size
            
            # Add to annotations
            annotations.append({
                'image': str(dest_image.name),
                'label': str(label_file.name),
                'ball_visibility': annotation.get('ball_visibility_score', 0),
                'video_id': annotation.get('video_id', video_dir.name)
            })
            
            processed_count += 1
    
    # Save annotations metadata
    metadata = {
        'total_annotations': len(annotations),
        'total_images': processed_count,
        'annotations': annotations
    }
    
    with open(metadata_dir / 'annotations.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Processed {processed_count} frames")
    print(f"Images saved to: {dest_images_dir}")
    print(f"Labels saved to: {dest_labels_dir}")
    print(f"Metadata saved to: {metadata_dir / 'annotations.json'}")
    
    return processed_count

if __name__ == "__main__":
    count = process_youtube_frames()
    print(f"Ready for training with {count} images!")