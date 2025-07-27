#!/usr/bin/env python3
"""
Debug script to compare YOLO vs Hough detection on the same frames
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def debug_detections():
    """Compare YOLO and Hough on test video frames"""
    
    print("🔍 YOLO vs Hough Detection Comparison")
    print("=" * 40)
    
    # Load YOLO model
    model_path = "training_data/experiments/real_detector_v2/weights/best.pt"
    if not Path(model_path).exists():
        print("❌ YOLO model not found")
        return
    
    model = YOLO(model_path)
    
    # Get latest video
    video_path = "uploads/raw/20250726_135247_1044291d.mp4"  # Your latest video
    if not Path(video_path).exists():
        print("❌ Video not found")
        return
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    yolo_detection_count = 0
    hough_detection_count = 0
    
    print(f"📹 Analyzing video: {video_path}")
    
    while cap.isOpened() and frame_count < 50:  # Test first 50 frames
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % 5 == 0:  # Every 5th frame
            print(f"\n🔍 Frame {frame_count}:")
            
            # Test YOLO
            yolo_results = model(frame, conf=0.001, verbose=False)  # Very low threshold
            yolo_detections = []
            for result in yolo_results:
                if result.boxes is not None:
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        yolo_detections.append(conf)
                        yolo_detection_count += 1
            
            if yolo_detections:
                max_conf = max(yolo_detections)
                print(f"   YOLO: {len(yolo_detections)} detections, max confidence: {max_conf:.4f}")
            else:
                print(f"   YOLO: No detections")
            
            # Test Hough
            hough_detections = detect_hough_circles(frame)
            if len(hough_detections) > 0:
                hough_detection_count += len(hough_detections)
                print(f"   Hough: {len(hough_detections)} circles detected")
            else:
                print(f"   Hough: No circles")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\n📊 Summary:")
    print(f"   Frames analyzed: {frame_count}")
    print(f"   YOLO detections: {yolo_detection_count}")
    print(f"   Hough detections: {hough_detection_count}")
    
    if yolo_detection_count == 0:
        print(f"\n❌ YOLO Issue: No detections found!")
        print(f"   This explains why Hough is winning - YOLO isn't detecting anything")
        print(f"   Possible causes:")
        print(f"   - Model not trained on this video type/orientation")
        print(f"   - Confidence threshold too high")
        print(f"   - Video preprocessing issues")
    else:
        print(f"\n✅ YOLO is detecting balls, but confidence may be too low")

def detect_hough_circles(frame):
    """Simple Hough circle detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, 
        dp=1, minDist=50, 
        param1=50, param2=30, 
        minRadius=8, maxRadius=60
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0, :]
    return []

if __name__ == "__main__":
    debug_detections()