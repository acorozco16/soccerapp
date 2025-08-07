#!/usr/bin/env python3
"""
Test current model performance before retraining
"""

import cv2
import numpy as np
from pathlib import Path

def test_current_model_performance():
    """Test the current model to establish baseline"""
    print("🧪 Testing current model performance...")
    
    # Check if we can import ultralytics locally
    try:
        from ultralytics import YOLO
        local_test = True
        print("✅ Using local ultralytics")
    except ImportError:
        local_test = False
        print("⚠️ ultralytics not available locally")
        print("📊 Current model info:")
        model_path = Path(__file__).parent / "models/soccer_ball_trained.pt"
        if model_path.exists():
            size_mb = model_path.stat().st_size / 1024 / 1024
            print(f"   - File: {model_path}")
            print(f"   - Size: {size_mb:.1f} MB")
            print(f"   - Last modified: July 27th, 17:12")
            print(f"   - Trained on: ~4,234 images")
        return None
    
    if local_test:
        # Load model and run basic test
        model_path = Path(__file__).parent / "models/soccer_ball_trained.pt"
        
        if not model_path.exists():
            print("❌ Model file not found")
            return None
            
        try:
            model = YOLO(str(model_path))
            print(f"✅ Model loaded successfully")
            print(f"📊 Model size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
            
            # Test on sample images if available
            sample_dir = Path(__file__).parent / "training_data/keepups_dataset/images"
            if sample_dir.exists():
                sample_images = list(sample_dir.glob("*.jpg"))[:3]
                print(f"\n🧪 Testing on {len(sample_images)} sample images:")
                
                for img_path in sample_images:
                    results = model(str(img_path), conf=0.05, verbose=False)
                    detections = len(results[0].boxes) if results[0].boxes is not None else 0
                    print(f"   - {img_path.name}: {detections} detection(s)")
            
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None

if __name__ == "__main__":
    print("🎯 Current Status:")
    print("   - Baseline: 19/25 detections (76%)")
    print("   - Target: 24/25 detections (95%)")
    print("   - Dataset: 7,335 labeled images ready")
    print()
    
    model = test_current_model_performance()
    
    print("\n📋 Recommendations:")
    if model:
        print("✅ Model is working - ready to test on original video")
        print("🔄 Recommend retraining with expanded 7,335-image dataset")
    else:
        print("🔄 Need to retrain model with ultralytics (use Colab)")
        print("📁 Dataset ready at: training_data/keepups_dataset/")
    
    print("\n🎯 Next steps:")
    print("1. Retrain model with 7,335 images (Colab)")
    print("2. Test new model on original video")
    print("3. Compare with 19/25 baseline")
    print("4. Aim for 24/25 (95%) accuracy")