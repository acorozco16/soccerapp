#!/usr/bin/env python3
"""
Train YOLO model with 3k labeled dataset
"""

from ultralytics import YOLO
import torch
from pathlib import Path

def train_model():
    """Train YOLOv8 on our labeled dataset"""
    
    print("🚀 Starting YOLO training with 3k labeled frames...")
    print(f"💻 Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Initialize model - start from pretrained
    model = YOLO('yolov8s.pt')
    
    # Dataset path
    dataset_path = Path(__file__).parent / "training_data" / "yolo_dataset_3k" / "dataset.yaml"
    
    # Training parameters optimized for soccer ball detection
    results = model.train(
        data=str(dataset_path),
        epochs=50,  # Reduced for testing
        imgsz=640,
        batch=16,
        patience=10,  # Early stopping
        save=True,
        device='cpu',  # Use CPU for Mac
        workers=4,
        project='runs/detect/ball_3k',
        name='train',
        exist_ok=True,
        pretrained=True,
        optimizer='Adam',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=0.05,
        cls=0.5,
        dfl=1.5,
        label_smoothing=0.0,
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        conf=0.001,  # Low confidence for testing
        iou=0.7,
        max_det=300,
        plots=True,
        save_json=False,
        save_hybrid=False,
        half=False,
        freeze=None,
        multi_scale=False,
        single_cls=True,  # Only detecting balls
        augment=True,
        close_mosaic=10,
        amp=False  # Automatic mixed precision off for CPU
    )
    
    print("\n✅ Training complete!")
    print(f"📊 Best model saved to: runs/detect/ball_3k/train/weights/best.pt")
    
    # Quick validation
    print("\n🔍 Running validation...")
    metrics = model.val()
    
    print(f"\n📈 Results:")
    print(f"   mAP50: {metrics.box.map50:.3f}")
    print(f"   mAP50-95: {metrics.box.map:.3f}")
    print(f"   Precision: {metrics.box.mp:.3f}")
    print(f"   Recall: {metrics.box.mr:.3f}")
    
    return model

if __name__ == "__main__":
    model = train_model()