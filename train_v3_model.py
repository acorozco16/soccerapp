#!/usr/bin/env python3
"""
Train YOLO v3 with augmented real data
"""

from ultralytics import YOLO
from pathlib import Path
import yaml

def train_v3_model():
    """Train improved model with augmented real data"""
    
    print("🚀 Training YOLO v3 with Augmented Real Data")
    print("=" * 45)
    
    # Check dataset
    dataset_path = Path("training_data/real_dataset_augmented")
    yaml_path = dataset_path / "dataset.yaml"
    
    if not yaml_path.exists():
        print("❌ Augmented dataset not found")
        return False
    
    # Count images
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    image_count = len(list(images_dir.glob("*.jpg")))
    label_count = len(list(labels_dir.glob("*.txt")))
    
    print(f"📊 Dataset Statistics:")
    print(f"   Images: {image_count}")
    print(f"   Labels: {label_count}")
    print(f"   Augmentation ratio: 9x original data")
    
    # Load model
    model = YOLO('yolov8n.pt')  # Start fresh
    
    print(f"\n🔧 Training Configuration:")
    print(f"   Base model: YOLOv8n")
    print(f"   Training approach: Real data focused")
    print(f"   Epochs: 40 (appropriate for small dataset)")
    
    # Train with optimized parameters for real data
    try:
        results = model.train(
            data=str(yaml_path),
            epochs=40,
            imgsz=640,
            batch=4,              # Small batch for small dataset
            name='real_detector_v3',
            lr0=0.01,            # Higher learning rate
            lrf=0.1,             # Higher final LR ratio
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=0.05,
            cls=0.5,
            dfl=1.5,
            save=True,
            save_period=10,       # Save every 10 epochs
            cache=False,
            device='',
            workers=4,
            project='training_data/experiments',
            exist_ok=True,
            pretrained=True,
            optimizer='SGD',      # SGD for small datasets
            verbose=True,
            seed=42,
            deterministic=True,
            single_cls=True,
            rect=False,
            cos_lr=True,
            close_mosaic=10,
            resume=False,
            amp=True,
            fraction=1.0,
            profile=False,
            freeze=None,
            plots=True,
            save_json=True,
            # Disable heavy augmentations (we already augmented)
            hsv_h=0.005,
            hsv_s=0.3,
            hsv_v=0.2,
            degrees=0.0,
            translate=0.05,
            scale=0.2,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.0,           # No flip (already augmented)
            mosaic=0.5,          # Light mosaic
            mixup=0.0,           # No mixup for small dataset
            copy_paste=0.0       # No copy-paste
        )
        
        print(f"\n✅ Training completed!")
        
        # Get results
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"📈 Final Metrics:")
            for key, value in metrics.items():
                if 'mAP' in key or 'precision' in key or 'recall' in key:
                    print(f"   {key}: {value:.4f}")
        
        model_path = Path("training_data/experiments/real_detector_v3/weights/best.pt")
        print(f"💾 Model saved: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def update_video_processor_v3():
    """Update video processor to use v3 model"""
    
    print("\n🔧 Updating video processor...")
    
    processor_path = Path("backend/video_processor.py")
    
    if not processor_path.exists():
        print("❌ Video processor not found")
        return False
    
    # Read current content
    with open(processor_path, 'r') as f:
        content = f.read()
    
    # Update model path
    old_path = 'training_data/experiments/real_detector_v2/weights/best.pt'
    new_path = 'training_data/experiments/real_detector_v3/weights/best.pt'
    
    if old_path in content:
        content = content.replace(old_path, new_path)
        
        # Update confidence threshold
        content = content.replace(
            'self.yolo_confidence_threshold = 0.01',
            'self.yolo_confidence_threshold = 0.015'  # Slightly higher for v3
        )
        
        # Write updated content
        with open(processor_path, 'w') as f:
            f.write(content)
        
        print("✅ Video processor updated!")
        print("   - Model: real_detector_v3")
        print("   - Confidence threshold: 0.015")
        return True
    else:
        print("⚠️ Could not find model path to update")
        return False

def main():
    """Main execution"""
    
    print("Starting YOLO v3 training pipeline...")
    
    # Train model
    success = train_v3_model()
    
    if not success:
        print("❌ Training failed, cannot proceed")
        return
    
    # Update processor
    update_video_processor_v3()
    
    print("\n🎯 Next Steps:")
    print("1. Restart backend server")
    print("2. Test same video to compare results")
    print("3. Expect more YOLO v3 detections (green circles)")
    print("4. Check for improved confidence scores")

if __name__ == "__main__":
    main()