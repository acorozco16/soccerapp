#!/usr/bin/env python3
"""
Simple ball annotation tool for marking ball positions in video frames
"""

import cv2
import numpy as np
import json
from pathlib import Path
import math

class BallAnnotator:
    def __init__(self, frames_dir: str):
        self.frames_dir = Path(frames_dir)
        self.annotations = {}
        self.current_frame_idx = 0
        self.frame_files = sorted(list(self.frames_dir.glob("*.jpg")))
        self.current_frame = None
        self.current_filename = None
        self.ball_position = None
        self.window_name = "Ball Annotator - Click on ball, press 's' to save, 'n' for next, 'q' to quit"
        
        if not self.frame_files:
            raise ValueError(f"No JPG files found in {frames_dir}")
        
        print(f"Found {len(self.frame_files)} frames to annotate")
        print("Instructions:")
        print("- Click on the ball to mark its position")
        print("- Press 's' to save annotation and move to next frame")
        print("- Press 'n' to skip frame (no ball visible)")
        print("- Press 'q' to quit and save all annotations")
        print("- Press 'b' to go back to previous frame")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to mark ball position"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ball_position = (x, y)
            self.display_frame()
    
    def display_frame(self):
        """Display current frame with annotation"""
        if self.current_frame is None:
            return
        
        display_frame = self.current_frame.copy()
        
        # Draw ball position if marked
        if self.ball_position:
            cv2.circle(display_frame, self.ball_position, 20, (0, 255, 0), 3)
            cv2.putText(display_frame, "Ball", 
                       (self.ball_position[0] - 20, self.ball_position[1] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add frame info
        frame_info = f"Frame {self.current_frame_idx + 1}/{len(self.frame_files)}: {self.current_filename}"
        cv2.putText(display_frame, frame_info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        instructions = "Click ball | 's'=save | 'n'=skip | 'b'=back | 'q'=quit"
        cv2.putText(display_frame, instructions, (10, display_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow(self.window_name, display_frame)
    
    def load_frame(self, idx: int):
        """Load frame at given index"""
        if 0 <= idx < len(self.frame_files):
            self.current_frame_idx = idx
            self.current_filename = self.frame_files[idx].name
            self.current_frame = cv2.imread(str(self.frame_files[idx]))
            
            # Load existing annotation if available
            if self.current_filename in self.annotations:
                ann = self.annotations[self.current_filename]
                self.ball_position = (ann['x'], ann['y'])
            else:
                self.ball_position = None
            
            self.display_frame()
            return True
        return False
    
    def save_current_annotation(self):
        """Save current frame annotation"""
        if self.ball_position and self.current_filename:
            # Convert to normalized coordinates (0-1)
            height, width = self.current_frame.shape[:2]
            x_norm = self.ball_position[0] / width
            y_norm = self.ball_position[1] / height
            
            self.annotations[self.current_filename] = {
                'x': self.ball_position[0],
                'y': self.ball_position[1],
                'x_norm': x_norm,
                'y_norm': y_norm,
                'width': width,
                'height': height,
                'has_ball': True
            }
            print(f"✅ Saved annotation for {self.current_filename}: ({self.ball_position[0]}, {self.ball_position[1]})")
        elif self.current_filename:
            # Mark as no ball visible
            self.annotations[self.current_filename] = {
                'has_ball': False,
                'width': self.current_frame.shape[1],
                'height': self.current_frame.shape[0]
            }
            print(f"⏭️ Marked {self.current_filename} as no ball visible")
    
    def save_all_annotations(self):
        """Save all annotations to JSON file"""
        output_file = self.frames_dir / "ball_annotations.json"
        
        annotations_data = {
            'total_frames': len(self.frame_files),
            'annotated_frames': len(self.annotations),
            'frames_with_ball': len([a for a in self.annotations.values() if a.get('has_ball', False)]),
            'annotations': self.annotations
        }
        
        with open(output_file, 'w') as f:
            json.dump(annotations_data, f, indent=2)
        
        print(f"💾 Saved all annotations to: {output_file}")
        print(f"📊 Summary: {annotations_data['annotated_frames']} frames annotated, {annotations_data['frames_with_ball']} with ball")
    
    def run(self):
        """Main annotation loop"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Load first frame
        self.load_frame(0)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):  # Quit
                break
            elif key == ord('s'):  # Save and next
                self.save_current_annotation()
                if self.current_frame_idx < len(self.frame_files) - 1:
                    self.load_frame(self.current_frame_idx + 1)
                else:
                    print("✅ All frames completed!")
                    break
            elif key == ord('n'):  # Skip (no ball)
                self.save_current_annotation()  # This will mark as no ball
                if self.current_frame_idx < len(self.frame_files) - 1:
                    self.load_frame(self.current_frame_idx + 1)
                else:
                    print("✅ All frames completed!")
                    break
            elif key == ord('b'):  # Back
                if self.current_frame_idx > 0:
                    self.load_frame(self.current_frame_idx - 1)
            elif key == 27:  # Escape
                break
        
        cv2.destroyAllWindows()
        self.save_all_annotations()

def extract_touch_frames():
    """Extract specific frames around touch events for focused annotation"""
    
    # Load touch events
    results_file = "uploads/processed/20250726_075501_a984b9ba_analysis.json"
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    touch_events = results.get('touch_events', [])
    video_path = "uploads/raw/20250726_075501_a984b9ba.mp4"
    output_dir = Path("training_data/touch_frames")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames_to_extract = []
    
    # For each touch event, extract frames around it
    for i, touch in enumerate(touch_events):
        touch_frame = touch['frame']
        timestamp = touch['timestamp']
        
        # Extract 10 frames before and after touch
        for offset in range(-10, 11):
            frame_num = touch_frame + offset
            if frame_num >= 0:
                frames_to_extract.append({
                    'frame_num': frame_num,
                    'touch_id': i + 1,
                    'offset': offset,
                    'timestamp': timestamp + (offset / fps)
                })
    
    # Remove duplicates
    unique_frames = {}
    for frame_info in frames_to_extract:
        frame_num = frame_info['frame_num']
        if frame_num not in unique_frames:
            unique_frames[frame_num] = frame_info
    
    print(f"🎯 Extracting {len(unique_frames)} frames around {len(touch_events)} touch events")
    
    # Extract frames
    extracted_info = []
    for frame_num in sorted(unique_frames.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            frame_info = unique_frames[frame_num]
            filename = f"touch_{frame_info['touch_id']}_frame_{frame_num:06d}_{frame_info['offset']:+03d}.jpg"
            filepath = output_dir / filename
            
            # Resize for consistency
            height, width = frame.shape[:2]
            if width > 1280:
                scale = 1280 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            cv2.imwrite(str(filepath), frame)
            extracted_info.append({
                'filename': filename,
                'frame_number': frame_num,
                'touch_id': frame_info['touch_id'],
                'offset_from_touch': frame_info['offset'],
                'timestamp': frame_info['timestamp']
            })
            
            print(f"Extracted: {filename}")
    
    cap.release()
    
    # Save extraction info
    info_file = output_dir / "extraction_info.json"
    with open(info_file, 'w') as f:
        json.dump({
            'source_video': video_path,
            'touch_events': touch_events,
            'extracted_frames': len(extracted_info),
            'frames': extracted_info
        }, f, indent=2)
    
    print(f"✅ Extracted {len(extracted_info)} frames to: {output_dir}")
    return output_dir

if __name__ == "__main__":
    print("🎯 Step 1: Extracting frames around touch events")
    frames_dir = extract_touch_frames()
    
    print(f"\n🖱️ Step 2: Starting annotation tool for {frames_dir}")
    print("Make sure you can see the video window, then follow the instructions")
    
    try:
        annotator = BallAnnotator(str(frames_dir))
        annotator.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have X11 forwarding enabled or are running on a desktop environment")