#!/usr/bin/env python3
"""
Quick test to see if we have a better model after training with real data
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

def test_available_models():
    """Test all available models on the original video"""
    
    video_path = "uploads/raw/20250726_075501_a984b9ba.mp4"
    
    # Find all model files
    models_to_test = []
    
    experiments_dir = Path("training_data/experiments")
    if experiments_dir.exists():
        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir():
                best_model = exp_dir / "weights" / "best.pt"
                if best_model.exists():
                    models_to_test.append((exp_dir.name, str(best_model)))
    
    if not models_to_test:
        print("❌ No trained models found")
        return
    
    print(f"🔍 Testing {len(models_to_test)} models on original video")
    
    results = {}
    
    for model_name, model_path in models_to_test:
        print(f"\n📊 Testing: {model_name}")
        
        try:
            model = YOLO(model_path)
            cap = cv2.VideoCapture(video_path)
            
            frame_count = 0
            detections = 0
            confidence_scores = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Test every 10th frame for speed
                if frame_count % 10 != 0:
                    continue
                
                # Run detection
                results_yolo = model(frame, verbose=False)
                
                for result in results_yolo:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        detections += 1
                        
                        for box in boxes:
                            conf = float(box.conf[0])
                            confidence_scores.append(conf)
            
            cap.release()
            
            # Calculate metrics
            processed_frames = frame_count // 10
            detection_rate = (detections / processed_frames) * 100 if processed_frames > 0 else 0
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            max_confidence = max(confidence_scores) if confidence_scores else 0
            
            results[model_name] = {
                'detection_rate': detection_rate,
                'avg_confidence': avg_confidence,
                'max_confidence': max_confidence,
                'total_detections': detections,
                'processed_frames': processed_frames
            }
            
            print(f"   Detection rate: {detection_rate:.1f}%")
            print(f"   Average confidence: {avg_confidence:.3f}")
            print(f"   Max confidence: {max_confidence:.3f}")
            print(f"   Total detections: {detections}/{processed_frames} frames")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary
    print(f"\n📈 MODEL PERFORMANCE SUMMARY:")
    print("=" * 50)
    
    if results:
        for model_name, metrics in sorted(results.items(), 
                                        key=lambda x: x[1]['detection_rate'], 
                                        reverse=True):
            print(f"{model_name}:")
            print(f"  🎯 Detection Rate: {metrics['detection_rate']:.1f}%")
            print(f"  🎪 Avg Confidence: {metrics['avg_confidence']:.3f}")
            print(f"  ⭐ Max Confidence: {metrics['max_confidence']:.3f}")
            print()
        
        best_model = max(results.items(), key=lambda x: x[1]['detection_rate'])
        print(f"🏆 Best Model: {best_model[0]}")
        print(f"   Detection Rate: {best_model[1]['detection_rate']:.1f}%")
        
        # Check if we improved from original 2.4%
        if best_model[1]['detection_rate'] > 5.0:
            print("✅ SIGNIFICANT IMPROVEMENT over original 2.4%!")
        elif best_model[1]['detection_rate'] > 2.4:
            print("✅ Slight improvement over original 2.4%")
        else:
            print("❌ No improvement over original 2.4%")
    
    return results

def check_training_progress():
    """Check if any training is still in progress"""
    
    experiments_dir = Path("training_data/experiments")
    
    print("🔍 Checking training progress...")
    
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir():
            results_csv = exp_dir / "results.csv"
            if results_csv.exists():
                print(f"📊 {exp_dir.name}:")
                
                # Read last few lines of results
                with open(results_csv, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip().split(',')
                        if len(last_line) > 5:
                            epoch = last_line[0].strip()
                            try:
                                precision = float(last_line[5]) if last_line[5] else 0
                                recall = float(last_line[6]) if last_line[6] else 0
                                map50 = float(last_line[7]) if last_line[7] else 0
                                print(f"   Epoch {epoch}: P={precision:.3f}, R={recall:.3f}, mAP50={map50:.3f}")
                            except:
                                print(f"   Latest epoch: {epoch}")
                
                # Check if best model exists
                best_model = exp_dir / "weights" / "best.pt"
                if best_model.exists():
                    print(f"   ✅ Best model available")
                else:
                    print(f"   ⏳ Training in progress...")
                print()

if __name__ == "__main__":
    print("🚀 Quick Model Performance Test")
    print("=" * 40)
    
    # Check training status
    check_training_progress()
    
    # Test available models
    test_available_models()