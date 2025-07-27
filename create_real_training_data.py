#!/usr/bin/env python3
"""
Extract frames from the actual test video to create real training data
"""

import cv2
import numpy as np
import json
from pathlib import Path
import math

def extract_frames_for_annotation():
    """Extract frames from test video for manual annotation"""
    
    # Get the test video
    video_path = "uploads/raw/20250726_075501_a984b9ba.mp4"
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    print(f"📹 Extracting frames from: {video_path}")
    
    # Create output directory
    output_dir = Path("training_data/real_video_frames")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Failed to open video")
        return
    
    frame_count = 0
    saved_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Extract every 30th frame (about 1 frame per second)
    frame_interval = 30
    
    frames_info = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save every nth frame
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            frame_filename = f"frame_{frame_count:06d}_{timestamp:.2f}s.jpg"
            frame_path = output_dir / frame_filename
            
            # Resize frame for consistency
            height, width = frame.shape[:2]
            if width > 1280:
                scale = 1280 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            cv2.imwrite(str(frame_path), frame)
            
            frames_info.append({
                "filename": frame_filename,
                "frame_number": frame_count,
                "timestamp": timestamp,
                "size": [frame.shape[1], frame.shape[0]]  # width, height
            })
            
            saved_count += 1
            print(f"Saved frame {saved_count}: {frame_filename}")
        
        frame_count += 1
        
        # Progress update
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
    
    cap.release()
    
    # Save frames info
    info_file = output_dir / "frames_info.json"
    with open(info_file, 'w') as f:
        json.dump({
            "source_video": video_path,
            "total_frames_extracted": saved_count,
            "frame_interval": frame_interval,
            "fps": fps,
            "frames": frames_info
        }, f, indent=2)
    
    print(f"\n✅ Extracted {saved_count} frames to: {output_dir}")
    print(f"📄 Frame info saved to: {info_file}")
    print(f"\nNext steps:")
    print(f"1. Manually review frames in: {output_dir}")
    print(f"2. Use annotation tool to mark ball positions")
    print(f"3. Create YOLO labels with real coordinates")
    print(f"4. Retrain model with this realistic data")

def analyze_touch_frames():
    """Analyze the frames where touches were detected to understand patterns"""
    
    # Load the analysis results
    results_file = "uploads/processed/20250726_075501_a984b9ba_analysis.json"
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    touch_events = results.get('touch_events', [])
    
    print(f"📊 Analyzing {len(touch_events)} touch events:")
    print(f"Video duration: {results['video_duration']}s")
    print(f"FPS: {results['metadata']['fps']}")
    
    for i, touch in enumerate(touch_events, 1):
        frame_num = touch['frame']
        timestamp = touch['timestamp']
        position = touch['position']
        method = touch['detection_method']
        confidence = touch['confidence']
        
        print(f"Touch {i}: Frame {frame_num} @ {timestamp:.2f}s")
        print(f"  Position: {position}")
        print(f"  Method: {method}")
        print(f"  Confidence: {confidence}")
        print()
    
    # Suggest which frames to focus on for annotation
    print("🎯 Recommended frames for manual annotation:")
    for touch in touch_events:
        frame_start = max(0, touch['frame'] - 30)  # 1 second before
        frame_end = touch['frame'] + 30  # 1 second after
        print(f"  Frames {frame_start}-{frame_end} (around touch at {touch['timestamp']:.1f}s)")

if __name__ == "__main__":
    print("🚀 Creating real training data from your test video")
    print("=" * 50)
    
    # Extract frames
    extract_frames_for_annotation()
    
    print("\n" + "=" * 50)
    
    # Analyze touch events
    analyze_touch_frames()