#!/usr/bin/env python3
"""
Train improved YOLO model with expanded 7,335-image dataset
"""

from ultralytics import YOLO
from pathlib import Path
import os

def train_improved_model():
    print("🚀 Starting YOLO training with expanded dataset...")
    
    # Dataset path
    dataset_path = Path(__file__).parent / "training_data/keepups_dataset/dataset.yaml"
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    print(f"📊 Dataset: {dataset_path}")
    print(f"🎯 Training with 7,335 images and 6,935 ball annotations")
    
    # Load YOLO model (start with pretrained YOLOv8n)
    model = YOLO('yolov8n.pt')
    
    # Train the model
    print("🔥 Starting training...")
    results = model.train(
        data=str(dataset_path),
        epochs=100,           # More epochs for better convergence
        imgsz=640,           # Standard image size
        batch=16,            # Adjust based on your GPU memory
        patience=20,         # Early stopping patience
        save=True,           # Save checkpoints
        cache=True,          # Cache images for faster training
        device=0,            # Use GPU if available, else CPU
        workers=8,           # Number of worker threads
        project='runs/train',
        name='improved_soccer_ball',
        exist_ok=True
    )
    
    # Export the best model
    best_model_path = results.save_dir / "weights/best.pt"
    final_model_path = models_dir / "soccer_ball_improved.pt"
    
    if best_model_path.exists():
        import shutil
        shutil.copy2(best_model_path, final_model_path)
        print(f"✅ Model saved to: {final_model_path}")
    else:
        print("❌ Training completed but best model not found")
    
    print("\n📊 Training Summary:")
    print(f"Final mAP: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"Training time: {results.results_dict.get('train_time', 'N/A')} seconds")
    
    return final_model_path

if __name__ == "__main__":
    model_path = train_improved_model()
    
    print(f"\n🎯 Next steps:")
    print(f"1. Test improved model against original video")
    print(f"2. Compare with baseline 19/25 (76%) accuracy")
    print(f"3. Target: 95% accuracy (24/25 detections)")
    print(f"4. Model ready at: {model_path}")