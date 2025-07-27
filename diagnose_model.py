#!/usr/bin/env python3
"""
Diagnose why the real_detector_v2 model isn't working on the test video
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

def diagnose_model():
    """Deep dive into why the model isn't detecting anything"""
    
    model_path = "training_data/experiments/real_detector_v2/weights/best.pt"
    video_path = "uploads/raw/20250726_075501_a984b9ba.mp4"
    
    print("🔍 Diagnosing model performance issues")
    print(f"Model: {model_path}")
    print(f"Video: {video_path}")
    
    try:
        # Load model
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
        
        # Test on a few frames with detailed output
        cap = cv2.VideoCapture(video_path)
        
        # Test frames around known touch events
        touch_frames = [15, 69, 141, 252, 333]  # From our previous analysis
        
        print("\n🎯 Testing on frames where we know balls exist:")
        
        for touch_frame in touch_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, touch_frame)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            print(f"\nFrame {touch_frame}:")
            
            # Run detection with very low confidence threshold
            results = model(frame, conf=0.01, verbose=True)  # Very low threshold
            
            for i, result in enumerate(results):
                boxes = result.boxes
                
                if boxes is not None and len(boxes) > 0:
                    print(f"  Found {len(boxes)} detections:")
                    for j, box in enumerate(boxes):
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        print(f"    Detection {j+1}: confidence={conf:.4f}, bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
                else:
                    print(f"  No detections found")
                
                # Check raw model outputs
                if hasattr(result, 'probs') and result.probs is not None:
                    print(f"  Raw probabilities: {result.probs}")
        
        cap.release()
        
        # Test on training images to verify model works
        print("\n🧪 Testing on training images:")
        
        training_images_dir = Path("training_data/real_dataset/images")
        if training_images_dir.exists():
            test_images = list(training_images_dir.glob("*.jpg"))[:3]  # Test first 3
            
            for img_path in test_images:
                print(f"\nTraining image: {img_path.name}")
                
                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Run detection
                results = model(img, conf=0.01, verbose=True)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        print(f"  Found {len(boxes)} detections on training image")
                        for box in boxes:
                            conf = float(box.conf[0])
                            print(f"    Confidence: {conf:.4f}")
                    else:
                        print(f"  No detections on training image")
        
        # Model info
        print(f"\n📊 Model Information:")
        print(f"Model classes: {model.names}")
        print(f"Model device: {model.device}")
        
    except Exception as e:
        print(f"❌ Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()

def test_confidence_thresholds():
    """Test different confidence thresholds"""
    
    model_path = "training_data/experiments/real_detector_v2/weights/best.pt"
    video_path = "uploads/raw/20250726_075501_a984b9ba.mp4"
    
    print("\n🎚️ Testing different confidence thresholds:")
    
    try:
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        
        # Test on frame where we know there's a ball
        cap.set(cv2.CAP_PROP_POS_FRAMES, 15)  # Known touch frame
        ret, frame = cap.read()
        
        if ret:
            thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
            
            for threshold in thresholds:
                results = model(frame, conf=threshold, verbose=False)
                
                total_detections = 0
                for result in results:
                    if result.boxes is not None:
                        total_detections += len(result.boxes)
                
                print(f"  Confidence {threshold}: {total_detections} detections")
        
        cap.release()
        
    except Exception as e:
        print(f"❌ Error testing thresholds: {e}")

if __name__ == "__main__":
    diagnose_model()
    test_confidence_thresholds()