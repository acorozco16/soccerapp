#!/usr/bin/env python3
"""
Test models with appropriate confidence thresholds
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

def test_models_properly():
    """Test models with optimized confidence thresholds"""
    
    video_path = "uploads/raw/20250726_075501_a984b9ba.mp4"
    
    # Test different models with appropriate thresholds
    models_to_test = [
        ("Original YouTube Model", "training_data/experiments/soccer_ball_detector/weights/best.pt", 0.2),
        ("Real Data Model v2", "training_data/experiments/real_detector_v2/weights/best.pt", 0.05),  # Lower threshold
    ]
    
    print(f"🔍 Testing models with optimized confidence thresholds")
    
    results = {}
    
    for model_name, model_path, conf_threshold in models_to_test:
        if not Path(model_path).exists():
            print(f"⏭️ Skipping {model_name} - model not found")
            continue
        
        print(f"\n📊 Testing: {model_name} (conf >= {conf_threshold})")
        
        try:
            model = YOLO(model_path)
            cap = cv2.VideoCapture(video_path)
            
            frame_count = 0
            detections = 0
            high_conf_detections = 0
            confidence_scores = []
            detection_frames = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Test every 5th frame
                if frame_count % 5 != 0:
                    continue
                
                # Run detection with model-specific threshold
                results_yolo = model(frame, conf=conf_threshold, verbose=False)
                
                frame_had_detection = False
                for result in results_yolo:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        if not frame_had_detection:
                            detections += 1
                            detection_frames.append(frame_count)
                            frame_had_detection = True
                        
                        for box in boxes:
                            conf = float(box.conf[0])
                            confidence_scores.append(conf)
                            
                            if conf > 0.3:  # High confidence
                                high_conf_detections += 1
                            
                            # Print significant detections
                            if conf > 0.2:
                                print(f"   Frame {frame_count}: Ball detected with confidence {conf:.3f}")
            
            cap.release()
            
            # Calculate metrics
            processed_frames = frame_count // 5
            detection_rate = (detections / processed_frames) * 100 if processed_frames > 0 else 0
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            max_confidence = max(confidence_scores) if confidence_scores else 0
            
            results[model_name] = {
                'detection_rate': detection_rate,
                'avg_confidence': avg_confidence,
                'max_confidence': max_confidence,
                'high_conf_detections': high_conf_detections,
                'total_detections': len(confidence_scores),
                'detection_frames': detection_frames[:10]  # First 10 detection frames
            }
            
            print(f"   Detection rate: {detection_rate:.1f}%")
            print(f"   Average confidence: {avg_confidence:.3f}")
            print(f"   Max confidence: {max_confidence:.3f}")
            print(f"   High confidence (>0.3): {high_conf_detections}")
            print(f"   Total detections: {len(confidence_scores)}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary comparison
    print(f"\n📈 OPTIMIZED MODEL COMPARISON:")
    print("=" * 60)
    
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  🎯 Detection Rate: {metrics['detection_rate']:.1f}%")
        print(f"  🎪 Avg Confidence: {metrics['avg_confidence']:.3f}")
        print(f"  ⭐ Max Confidence: {metrics['max_confidence']:.3f}")
        print(f"  🔥 High Confidence Detections: {metrics['high_conf_detections']}")
        print(f"  📊 Total Ball Detections: {metrics['total_detections']}")
        print(f"  📍 Detection Frames: {metrics['detection_frames']}")
        print()
    
    # Determine best model
    if results:
        best_model = max(results.items(), 
                        key=lambda x: x[1]['detection_rate'])
        print(f"🏆 Best performing model: {best_model[0]}")
        print(f"   Detection Rate: {best_model[1]['detection_rate']:.1f}%")
        
        # Check improvement
        best_rate = best_model[1]['detection_rate']
        if best_rate > 15:
            print("🎉 MAJOR IMPROVEMENT! Over 15% detection rate!")
        elif best_rate > 5:
            print("✅ SIGNIFICANT IMPROVEMENT! Over 5% detection rate!")
        elif best_rate > 2.4:
            print("✅ Improvement over original 2.4%")
        else:
            print("❌ No improvement over original")
    
    return results

def analyze_touch_coverage():
    """Check how well we cover the known touch events"""
    
    # Known touch events from analysis
    known_touches = [15, 69, 141, 252, 333]
    
    print(f"\n🎯 Analyzing coverage of known touch events:")
    print(f"Known touch frames: {known_touches}")
    
    model_path = "training_data/experiments/real_detector_v2/weights/best.pt"
    video_path = "uploads/raw/20250726_075501_a984b9ba.mp4"
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    detected_touches = 0
    
    for touch_frame in known_touches:
        cap.set(cv2.CAP_PROP_POS_FRAMES, touch_frame)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Test with low confidence
        results = model(frame, conf=0.05, verbose=False)
        
        best_confidence = 0
        detection_count = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                detection_count = len(boxes)
                for box in boxes:
                    conf = float(box.conf[0])
                    best_confidence = max(best_confidence, conf)
        
        if best_confidence > 0.05:
            detected_touches += 1
            status = "✅ DETECTED"
        else:
            status = "❌ MISSED"
        
        print(f"Frame {touch_frame}: {status} (best conf: {best_confidence:.3f}, {detection_count} detections)")
    
    cap.release()
    
    coverage = (detected_touches / len(known_touches)) * 100
    print(f"\n📊 Touch Coverage: {detected_touches}/{len(known_touches)} = {coverage:.1f}%")
    
    return coverage

if __name__ == "__main__":
    print("🎯 Proper Model Testing with Optimized Thresholds")
    print("=" * 55)
    
    # Test models properly
    results = test_models_properly()
    
    # Check touch coverage
    coverage = analyze_touch_coverage()
    
    print(f"\n🎉 SUMMARY:")
    if coverage >= 80:
        print(f"✅ Excellent touch coverage: {coverage:.1f}%")
    elif coverage >= 60:
        print(f"✅ Good touch coverage: {coverage:.1f}%")
    else:
        print(f"⚠️ Touch coverage needs improvement: {coverage:.1f}%")