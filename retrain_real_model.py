#!/usr/bin/env python3
"""
Retrain the real data model with more epochs for better performance
"""

from ultralytics import YOLO
import sys
from pathlib import Path

def retrain_real_model():
    """Retrain with more epochs for better results"""
    
    print("🎯 Retraining real data model with more epochs")
    
    real_dataset = Path("training_data/real_dataset/dataset.yaml")
    
    if not real_dataset.exists():
        print(f"❌ Real dataset not found: {real_dataset}")
        return False
    
    try:
        # Load pretrained model
        model = YOLO('yolov8n.pt')
        print("✅ Loaded YOLOv8 nano model")
        
        print(f"🚀 Starting proper training on real data...")
        print(f"   Dataset: {real_dataset}")
        print(f"   Images: 55 (25 with ball, 30 without)")
        
        # Proper training with more epochs
        results = model.train(
            data=str(real_dataset),
            epochs=30,      # More epochs for better learning
            imgsz=640,
            batch=4,        # Small batch for small dataset
            name='real_detector_v2',
            project='training_data/experiments',
            save_period=10,
            patience=10,
            lr0=0.005,      # Moderate learning rate
            warmup_epochs=3,
            verbose=True,
            device='cpu'    # Ensure CPU training
        )
        
        print("✅ Training completed!")
        
        model_path = Path("training_data/experiments/real_detector_v2/weights/best.pt")
        print(f"📍 Model saved at: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    print("⚡ Retraining Real Data Model")
    print("=" * 35)
    
    # Retrain with proper epochs
    success = retrain_real_model()
    
    if success:
        print("\n✅ Training completed! Run quick_test.py to check performance")
    else:
        print("❌ Training failed")