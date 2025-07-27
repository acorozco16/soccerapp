#!/usr/bin/env python3
"""
Train Soccer Ball Detection Model
"""

from ultralytics import YOLO
import sys

def train_model():
    """Train the soccer ball detection model"""
    try:
        print("🚀 Starting soccer ball detection training...")
        
        # Load YOLO model
        model = YOLO('yolov8n.pt')
        
        # Train the model
        results = model.train(
            data='training_data/datasets/yolo_soccer/dataset.yaml',
            epochs=50,
            imgsz=640,
            batch=16,
            name='soccer_ball_detector',
            project='training_data/experiments',
            save_period=10,  # Save every 10 epochs
            patience=10      # Early stopping if no improvement
        )
        
        print("✅ Training completed successfully!")
        print(f"Best model saved at: {results.save_dir}/weights/best.pt")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏸️ Training paused by user")
        print("You can resume by running this script again")
        return False
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)