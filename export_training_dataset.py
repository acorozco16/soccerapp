#!/usr/bin/env python3
"""
Export labeled dataset for YOLO training
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm

def export_dataset():
    """Export annotations to YOLO format"""
    
    # Paths
    base_dir = Path(__file__).resolve().parent
    annotations_file = base_dir / "training_data" / "annotations.json"
    labeling_queue = base_dir / "training_data" / "labeling_queue"
    
    # Output paths
    output_dir = base_dir / "training_data" / "yolo_dataset_3k"
    train_images = output_dir / "train" / "images"
    train_labels = output_dir / "train" / "labels"
    val_images = output_dir / "val" / "images"
    val_labels = output_dir / "val" / "labels"
    
    # Create directories
    for dir_path in [train_images, train_labels, val_images, val_labels]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Filter only annotated frames (with actual ball positions)
    annotated_frames = {k: v for k, v in annotations.items() if len(v) > 0}
    
    print(f"Total frames with annotations: {len(annotations)}")
    print(f"Frames with balls: {len(annotated_frames)}")
    print(f"Empty frames (no ball): {len(annotations) - len(annotated_frames)}")
    
    # Split 80/20 for train/val
    all_files = list(annotations.keys())
    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"\nDataset split:")
    print(f"Training: {len(train_files)} frames")
    print(f"Validation: {len(val_files)} frames")
    
    # Process files
    processed = 0
    for filename, balls in tqdm(annotations.items(), desc="Exporting dataset"):
        # Source image
        src_image = labeling_queue / filename
        if not src_image.exists():
            continue
        
        # Determine if train or val
        if filename in train_files:
            dst_image = train_images / filename
            dst_label = train_labels / filename.replace('.jpg', '.txt')
        else:
            dst_image = val_images / filename
            dst_label = val_labels / filename.replace('.jpg', '.txt')
        
        # Copy image
        shutil.copy2(src_image, dst_image)
        
        # Create label file
        with open(dst_label, 'w') as f:
            for ball in balls:
                # YOLO format: class_id x_center y_center width height
                x_center = ball['x']
                y_center = ball['y']
                width = 0.05  # 5% of image width (adjust based on typical ball size)
                height = 0.05  # 5% of image height
                
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        processed += 1
    
    # Create dataset.yaml
    dataset_yaml = output_dir / "dataset.yaml"
    with open(dataset_yaml, 'w') as f:
        f.write(f"""path: {output_dir.absolute()}
train: train/images
val: val/images

names:
  0: ball

nc: 1  # number of classes
""")
    
    print(f"\n✅ Dataset exported successfully!")
    print(f"📁 Location: {output_dir}")
    print(f"📊 Total images processed: {processed}")
    print(f"🎯 Ready for YOLO training!")
    
    return str(output_dir)

if __name__ == "__main__":
    export_dataset()