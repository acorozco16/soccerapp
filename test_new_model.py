#!/usr/bin/env python3
"""
Test the newly trained YOLO model directly on the video
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

def test_yolo_model():
    """Test the new YOLO model on the original video"""
    
    # Load the trained model
    model_path = "training_data/experiments/soccer_ball_detector/weights/best.pt"
    model = YOLO(model_path)
    
    print(f"🔍 Testing model: {model_path}")
    
    # Find the test video
    uploads_dir = Path("uploads/raw")
    video_files = list(uploads_dir.glob("*.mp4"))
    
    if not video_files:
        print("❌ No video files found in uploads directory")
        return
    
    video_path = video_files[0]  # Use first video found
    print(f"📹 Testing on video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("❌ Failed to open video")
        return
    
    frame_count = 0
    ball_detections = 0
    high_confidence_detections = 0
    
    print("🚀 Processing frames...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames for faster processing
        if frame_count % 5 != 0:
            continue
        
        # Run YOLO detection
        results = model(frame, verbose=False)
        
        # Check for ball detections
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                ball_detections += 1
                
                # Check confidence levels
                for box in boxes:
                    confidence = float(box.conf[0])
                    if confidence > 0.5:
                        high_confidence_detections += 1
                    
                    print(f"Frame {frame_count}: Ball detected with confidence {confidence:.3f}")
    
    cap.release()
    
    # Results
    processed_frames = frame_count // 5
    detection_rate = (ball_detections / processed_frames) * 100 if processed_frames > 0 else 0
    high_conf_rate = (high_confidence_detections / processed_frames) * 100 if processed_frames > 0 else 0
    
    print(f"\n📊 YOLO Model Test Results:")
    print(f"Total frames processed: {processed_frames}")
    print(f"Frames with ball detections: {ball_detections}")
    print(f"High confidence detections (>0.5): {high_confidence_detections}")
    print(f"Detection rate: {detection_rate:.1f}%")
    print(f"High confidence rate: {high_conf_rate:.1f}%")
    
    if detection_rate > 50:
        print("✅ YOLO model appears to be working well!")
    elif detection_rate > 20:
        print("⚠️ YOLO model has moderate performance")
    else:
        print("❌ YOLO model has poor performance on this video")

if __name__ == "__main__":
    test_yolo_model()