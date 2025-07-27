#!/usr/bin/env python3
"""
Train improved YOLO model with augmented data
"""

from ultralytics import YOLO
from pathlib import Path
import yaml

def train_augmented_model():
    """Train YOLO model with augmented dataset"""
    
    print("🚀 Training Improved YOLO Model with Augmented Data")
    print("=" * 55)
    
    # Check if augmented dataset exists
    dataset_path = Path("training_data/datasets/real_ball_dataset_augmented")
    yaml_path = dataset_path / "dataset.yaml"
    
    if not yaml_path.exists():
        print("❌ Augmented dataset not found. Run augment_training_data.py first.")
        return False
    
    # Load base model
    model = YOLO('yolov8n.pt')  # Start fresh for better generalization
    
    print(f"📊 Training Configuration:")
    print(f"   Dataset: {dataset_path}")
    print(f"   Base model: YOLOv8n")
    print(f"   Training strategy: Multi-stage with augmented data")
    
    # Training parameters optimized for augmented data
    results = model.train(
        data=str(yaml_path),
        epochs=50,           # More epochs due to larger dataset
        imgsz=640,
        batch=8,             # Larger batch size
        name='real_detector_v3_augmented',
        lr0=0.01,           # Higher learning rate initially
        lrf=0.01,           # Final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=0.05,           # Lower box loss weight
        cls=0.5,            # Standard classification loss
        dfl=1.5,            # Distribution focal loss
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
        hsv_h=0.015,        # Additional HSV augmentation
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,        # No rotation (handled by albumentations)
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,    # No perspective (handled by albumentations)
        flipud=0.0,         # No flip (handled by albumentations)
        fliplr=0.5,
        mosaic=1.0,         # Enable mosaic augmentation
        mixup=0.1,          # Enable mixup
        copy_paste=0.1,     # Enable copy-paste augmentation
        patience=10,        # Early stopping patience
        save=True,
        save_period=5,      # Save every 5 epochs
        cache=False,        # Don't cache (large dataset)
        device='',          # Auto-detect device
        workers=8,          # More workers for data loading
        project='training_data/experiments',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',  # AdamW optimizer
        verbose=True,
        seed=42,           # Reproducible results
        deterministic=True,
        single_cls=True,   # Single class (ball)
        rect=False,        # No rectangular training
        cos_lr=True,       # Cosine learning rate scheduler
        close_mosaic=10,   # Disable mosaic in last 10 epochs
        resume=False,
        amp=True,          # Automatic mixed precision
        fraction=1.0,      # Use full dataset
        profile=False,
        freeze=None,       # Don't freeze any layers
        multi_scale=True,  # Multi-scale training
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True,
        save_json=True
    )
    
    print(f"\n✅ Training complete!")
    print(f"📈 Final Results:")
    print(f"   mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"   mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    
    # Save model info
    model_path = Path("training_data/experiments/real_detector_v3_augmented/weights/best.pt")
    print(f"📁 Best model saved to: {model_path}")
    
    return True

def update_video_processor():
    """Update video processor to use new model"""
    
    print("\n🔧 Updating video processor to use improved model...")
    
    processor_path = Path("backend/video_processor.py")
    
    if not processor_path.exists():
        print("❌ Video processor not found")
        return False
    
    # Read current content
    with open(processor_path, 'r') as f:
        content = f.read()
    
    # Update model path to v3
    old_path = 'training_data/experiments/real_detector_v2/weights/best.pt'
    new_path = 'training_data/experiments/real_detector_v3_augmented/weights/best.pt'
    
    if old_path in content:
        content = content.replace(old_path, new_path)
        
        # Also update the confidence threshold for better model
        content = content.replace(
            'self.yolo_confidence_threshold = 0.01',
            'self.yolo_confidence_threshold = 0.02'  # Slightly higher for better model
        )
        
        # Write updated content
        with open(processor_path, 'w') as f:
            f.write(content)
        
        print("✅ Video processor updated!")
        print("   - Model path updated to v3_augmented")
        print("   - Confidence threshold updated to 0.02")
        return True
    else:
        print("⚠️ Model path not found in video processor")
        return False

def main():
    """Main training pipeline"""
    
    # Step 1: Train augmented model
    success = train_augmented_model()
    
    if not success:
        print("❌ Training failed")
        return
    
    # Step 2: Update video processor
    update_video_processor()
    
    print("\n🎯 Next Steps:")
    print("1. Restart the backend server")
    print("2. Test with the same video to compare results")
    print("3. Look for more green circles (YOLO v3) in debug frames")
    print("4. Check if confidence scores improve")

if __name__ == "__main__":
    main()