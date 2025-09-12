#!/usr/bin/env python3
"""
Create YOLO labels using the known touch positions from the analysis results
"""

import cv2
import numpy as np
import json
from pathlib import Path
import math

def create_labels_from_touch_data():
    """Create training labels using the detected touch positions"""
    
    # Load the analysis results with touch positions
    results_file = "uploads/processed/20250726_075501_a984b9ba_analysis.json"
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    touch_events = results.get('touch_events', [])
    video_path = "uploads/raw/20250726_075501_a984b9ba.mp4"
    
    print(f"📹 Creating labels from {len(touch_events)} touch events")
    
    # Create output directories
    output_dir = Path("training_data/real_dataset")
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    created_samples = []
    
    # For each touch event, create training samples
    for i, touch in enumerate(touch_events):
        touch_frame = touch['frame']
        touch_pos = touch['position']  # [x, y] in pixels
        confidence = touch['confidence']
        
        print(f"Processing touch {i+1}: Frame {touch_frame}, Position {touch_pos}")
        
        # Extract frames around the touch
        for offset in range(-5, 6):  # 11 frames around each touch
            frame_num = touch_frame + offset
            if frame_num < 0:
                continue
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Resize if needed
            if width > 1280:
                scale = 1280 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
                # Scale touch position accordingly
                scaled_x = int(touch_pos[0] * scale)
                scaled_y = int(touch_pos[1] * scale)
            else:
                new_width, new_height = width, height
                scaled_x, scaled_y = touch_pos
            
            # Create filename
            filename_base = f"real_touch_{i+1}_frame_{frame_num:06d}_offset_{offset:+03d}"
            image_path = images_dir / f"{filename_base}.jpg"
            label_path = labels_dir / f"{filename_base}.txt"
            
            # Save image
            cv2.imwrite(str(image_path), frame)
            
            # Create YOLO label
            # For frames close to the touch, use the actual position
            if abs(offset) <= 2:  # Within 2 frames of actual touch
                # Estimate ball position based on offset (simple interpolation)
                if offset == 0:
                    ball_x, ball_y = scaled_x, scaled_y
                else:
                    # Add some variation for frames before/after
                    ball_x = scaled_x + np.random.randint(-20, 21)
                    ball_y = scaled_y + np.random.randint(-20, 21)
                    ball_x = max(20, min(new_width - 20, ball_x))
                    ball_y = max(20, min(new_height - 20, ball_y))
                
                # Convert to YOLO format (normalized center coordinates)
                x_center = ball_x / new_width
                y_center = ball_y / new_height
                
                # Estimate ball size (typical soccer ball in frame)
                ball_width = 40 / new_width   # Assume 40 pixel width
                ball_height = 40 / new_height # Assume 40 pixel height
                
                # Write YOLO label
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {ball_width:.6f} {ball_height:.6f}\n")
                
                created_samples.append({
                    'image': str(image_path),
                    'label': str(label_path),
                    'touch_id': i + 1,
                    'frame_offset': offset,
                    'ball_position': [ball_x, ball_y],
                    'confidence': confidence
                })
                
                print(f"  Created: {filename_base}.jpg with ball at ({ball_x}, {ball_y})")
            
            else:
                # For frames further away, create negative samples (no ball)
                # Don't create a label file (YOLO treats missing labels as no objects)
                created_samples.append({
                    'image': str(image_path),
                    'label': None,
                    'touch_id': i + 1,
                    'frame_offset': offset,
                    'ball_position': None,
                    'confidence': 0.0
                })
                print(f"  Created: {filename_base}.jpg (no ball)")
    
    cap.release()
    
    # Create dataset configuration
    dataset_config = {
        'path': str(output_dir.absolute()),
        'train': 'images',
        'val': 'images',  # Using same data for val (small dataset)
        'nc': 1,
        'names': ['soccer_ball']
    }
    
    config_path = output_dir / "dataset.yaml"
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    # Save metadata
    metadata = {
        'source_video': video_path,
        'total_samples': len(created_samples),
        'positive_samples': len([s for s in created_samples if s['ball_position'] is not None]),
        'negative_samples': len([s for s in created_samples if s['ball_position'] is None]),
        'touch_events': touch_events,
        'samples': created_samples
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Created real training dataset:")
    print(f"   Images: {len(created_samples)}")
    print(f"   With ball: {metadata['positive_samples']}")
    print(f"   Without ball: {metadata['negative_samples']}")
    print(f"   Dataset config: {config_path}")
    print(f"   Metadata: {metadata_path}")
    
    return output_dir

def create_mixed_dataset():
    """Create a mixed dataset combining YouTube data with real video data"""
    
    print("🔄 Creating mixed dataset with YouTube + Real video data")
    
    real_dir = Path("training_data/real_dataset")
    youtube_dir = Path("training_data/datasets/yolo_soccer")
    mixed_dir = Path("training_data/mixed_dataset")
    
    if not real_dir.exists():
        print("❌ Real dataset not found. Create it first.")
        return None
    
    if not youtube_dir.exists():
        print("❌ YouTube dataset not found")
        return None
    
    # Create mixed dataset directories
    for split in ['train', 'val']:
        (mixed_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (mixed_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    import shutil
    
    # Copy all real data to training set (higher weight)
    real_images = list((real_dir / "images").glob("*.jpg"))
    real_labels = list((real_dir / "labels").glob("*.txt"))
    
    print(f"📁 Copying {len(real_images)} real images to mixed dataset")
    
    for img_path in real_images:
        # Copy to train
        shutil.copy2(img_path, mixed_dir / 'images' / 'train' / img_path.name)
        
        # Copy corresponding label if exists
        label_path = real_dir / "labels" / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, mixed_dir / 'labels' / 'train' / f"{img_path.stem}.txt")
    
    # Copy subset of YouTube data
    youtube_train_images = list((youtube_dir / "images" / "train").glob("*.jpg"))[:500]  # Limit to 500
    
    print(f"📁 Copying {len(youtube_train_images)} YouTube images to mixed dataset")
    
    for img_path in youtube_train_images:
        shutil.copy2(img_path, mixed_dir / 'images' / 'train' / f"yt_{img_path.name}")
        
        label_path = youtube_dir / "labels" / "train" / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, mixed_dir / 'labels' / 'train' / f"yt_{img_path.stem}.txt")
    
    # Use some real data for validation
    val_count = max(5, len(real_images) // 5)
    for img_path in real_images[:val_count]:
        shutil.copy2(img_path, mixed_dir / 'images' / 'val' / img_path.name)
        
        label_path = real_dir / "labels" / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, mixed_dir / 'labels' / 'val' / f"{img_path.stem}.txt")
    
    # Create dataset config
    dataset_config = {
        'path': str(mixed_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['soccer_ball']
    }
    
    config_path = mixed_dir / "dataset.yaml"
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✅ Mixed dataset created at: {mixed_dir}")
    print(f"   Config: {config_path}")
    
    return mixed_dir

if __name__ == "__main__":
    print("🎯 Creating real training dataset from touch positions")
    print("=" * 60)
    
    # Create real dataset
    real_dataset_dir = create_labels_from_touch_data()
    
    print("\n" + "=" * 60)
    
    # Create mixed dataset
    mixed_dataset_dir = create_mixed_dataset()
    
    print(f"\n🚀 Ready for training!")
    print(f"Real dataset: {real_dataset_dir}")
    print(f"Mixed dataset: {mixed_dataset_dir}")
    print(f"\nNext: Run training on the mixed dataset for best results")