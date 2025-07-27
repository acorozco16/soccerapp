#!/usr/bin/env python3
"""
Simple training script for real data only - fast training
"""

from ultralytics import YOLO
import sys
from pathlib import Path

def train_simple_real_model():
    """Train a simple model using only real data for fast results"""
    
    print("🎯 Fast training on real data only")
    
    real_dataset = Path("training_data/real_dataset/dataset.yaml")
    
    if not real_dataset.exists():
        print(f"❌ Real dataset not found: {real_dataset}")
        return False
    
    try:
        # Load pretrained model
        model = YOLO('yolov8n.pt')
        print("✅ Loaded YOLOv8 nano model")
        
        print(f"🚀 Starting fast training on real data...")
        print(f"   Dataset: {real_dataset}")
        print(f"   Images: 55 (25 with ball, 30 without)")
        
        # Very fast training - just 10 epochs
        results = model.train(
            data=str(real_dataset),
            epochs=10,      # Fast training
            imgsz=640,
            batch=2,        # Very small batch
            name='fast_real_detector',
            project='training_data/experiments',
            save_period=5,
            patience=5,
            lr0=0.01,       # Higher learning rate
            warmup_epochs=1,
            verbose=True
        )
        
        print("✅ Fast training completed!")
        
        model_path = Path("training_data/experiments/fast_real_detector/weights/best.pt")
        print(f"📍 Model saved at: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def test_fast_model():
    """Test the fast-trained model"""
    
    model_path = "training_data/experiments/fast_real_detector/weights/best.pt"
    
    if not Path(model_path).exists():
        print("❌ Fast model not found")
        return
    
    print(f"\n🧪 Testing fast-trained model")
    
    from ultralytics import YOLO
    import cv2
    
    video_path = "uploads/raw/20250726_075501_a984b9ba.mp4"
    
    try:
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        detections = 0
        confidence_scores = []
        
        print("🔍 Running detection on test video...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Test every 5th frame
            if frame_count % 5 != 0:
                continue
            
            # Run detection
            results = model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    detections += 1
                    
                    for box in boxes:
                        conf = float(box.conf[0])
                        confidence_scores.append(conf)
                        print(f"Frame {frame_count}: Ball detected with confidence {conf:.3f}")
        
        cap.release()
        
        # Calculate results
        processed_frames = frame_count // 5
        detection_rate = (detections / processed_frames) * 100 if processed_frames > 0 else 0
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        max_confidence = max(confidence_scores) if confidence_scores else 0
        
        print(f"\n📊 FAST MODEL RESULTS:")
        print(f"   Detection rate: {detection_rate:.1f}%")
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"   Max confidence: {max_confidence:.3f}")
        print(f"   Total detections: {detections}/{processed_frames} frames")
        
        # Compare to original
        if detection_rate > 5.0:
            print("🎉 MAJOR IMPROVEMENT! Over 5% detection rate!")
        elif detection_rate > 2.4:
            print("✅ Improvement over original 2.4%")
        else:
            print("❌ No improvement over original")
        
        return {
            'detection_rate': detection_rate,
            'avg_confidence': avg_confidence,
            'max_confidence': max_confidence
        }
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return None

if __name__ == "__main__":
    print("⚡ Fast Real Data Training")
    print("=" * 30)
    
    # Train fast model
    success = train_simple_real_model()
    
    if success:
        # Test the model
        results = test_fast_model()
        
        if results and results['detection_rate'] > 2.4:
            print("\n🎯 SUCCESS! The real data approach works!")
            print("You can now train longer for even better results")
        else:
            print("\n⚠️ Need more training or better data")
    else:
        print("❌ Training failed")