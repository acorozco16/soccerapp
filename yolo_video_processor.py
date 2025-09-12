#!/usr/bin/env python3
"""
Enhanced video processor that uses the trained YOLO model alongside traditional methods
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import asyncio
import logging
import json
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from datetime import datetime
import imageio
from collections import deque
import math
from ultralytics import YOLO

logger = logging.getLogger(__name__)

@dataclass
class TouchEvent:
    timestamp: float
    frame_number: int
    confidence: float
    position: Tuple[int, int]
    detection_method: str
    
    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "frame": self.frame_number,
            "confidence": self.confidence,
            "position": self.position,
            "detection_method": self.detection_method
        }

class BallDetection(NamedTuple):
    position: Tuple[int, int]
    confidence: float
    method: str
    radius: Optional[int] = None
    area: Optional[float] = None

class YOLOVideoProcessor:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Load trained YOLO model
        try:
            model_path = "training_data/experiments/soccer_ball_detector/weights/best.pt"
            self.yolo_model = YOLO(model_path)
            print(f"✅ Loaded YOLO model: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            self.yolo_model = None
        
        # Traditional detection parameters (fallback)
        self.ball_color_ranges = {
            "orange_clean": {
                "lower": np.array([5, 100, 100]),
                "upper": np.array([15, 255, 255])
            },
            "orange_muddy": {
                "lower": np.array([8, 50, 80]),
                "upper": np.array([25, 200, 180])
            },
            "white_clean": {
                "lower": np.array([0, 0, 200]),
                "upper": np.array([180, 30, 255])
            },
        }
        
        # Hough parameters
        self.hough_params = [
            {"dp": 1, "minDist": 50, "param1": 50, "param2": 30, "minRadius": 8, "maxRadius": 60},
            {"dp": 1, "minDist": 40, "param1": 30, "param2": 20, "minRadius": 10, "maxRadius": 50},
        ]
        
        # Touch detection parameters
        self.touch_threshold_pixels = 50
        self.debounce_time = 0.5
        self.min_ball_velocity = 3
        
        # Processing parameters
        self.frame_skip = 3
        self.target_width = 1280
        self.max_fps = 30
        
        # Tracking state
        self.ball_history = deque(maxlen=15)
        self.last_known_position = None
        self.frames_without_detection = 0

    def _detect_ball_yolo(self, frame: np.ndarray) -> List[BallDetection]:
        """YOLO-based ball detection"""
        detections = []
        
        if self.yolo_model is None:
            return detections
        
        try:
            # Run YOLO inference
            results = self.yolo_model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Extract box coordinates and confidence
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Calculate center position
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        # Calculate radius from bounding box
                        radius = int(max(x2 - x1, y2 - y1) / 2)
                        
                        # Only accept detections with minimum confidence
                        if conf > 0.1:  # Lower threshold since model shows low confidence
                            detections.append(BallDetection(
                                position=(center_x, center_y),
                                confidence=float(conf * 2.0),  # Boost confidence for YOLO
                                method="yolo",
                                radius=radius
                            ))
        
        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}")
        
        return detections

    def _detect_ball_traditional(self, frame: np.ndarray) -> List[BallDetection]:
        """Traditional HSV + Hough circle detection"""
        detections = []
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Combine color masks
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for color_name, ranges in self.ball_color_ranges.items():
            color_mask = cv2.inRange(hsv, ranges["lower"], ranges["upper"])
            combined_mask = cv2.bitwise_or(combined_mask, color_mask)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Hough circle detection
        blurred = cv2.GaussianBlur(combined_mask, (9, 9), 2)
        
        for params in self.hough_params:
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, **params)
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :2]:  # Take up to 2 best circles
                    x, y, r = circle
                    
                    # Calculate confidence based on mask overlap
                    circle_mask = np.zeros_like(combined_mask)
                    cv2.circle(circle_mask, (x, y), r, 255, -1)
                    overlap = cv2.bitwise_and(combined_mask, circle_mask)
                    confidence = np.sum(overlap) / (np.pi * r * r * 255) if r > 0 else 0
                    
                    if confidence > 0.2:
                        detections.append(BallDetection(
                            position=(int(x), int(y)),
                            confidence=float(confidence),
                            method="traditional",
                            radius=int(r)
                        ))
        
        return detections

    def _detect_ball_hybrid(self, frame: np.ndarray) -> Optional[BallDetection]:
        """Hybrid detection combining YOLO and traditional methods"""
        all_detections = []
        
        # YOLO detection
        yolo_detections = self._detect_ball_yolo(frame)
        all_detections.extend(yolo_detections)
        
        # Traditional detection as fallback
        traditional_detections = self._detect_ball_traditional(frame)
        all_detections.extend(traditional_detections)
        
        if not all_detections:
            return None
        
        # Choose best detection based on confidence and method priority
        best_detection = None
        best_score = -1
        
        for detection in all_detections:
            # Method priority weights
            method_weights = {
                "yolo": 1.5,      # Prefer YOLO if confident
                "traditional": 1.0
            }
            
            score = detection.confidence * method_weights.get(detection.method, 1.0)
            
            # Bonus for consistency with history
            if self.ball_history:
                last_pos = self.ball_history[-1].position
                distance = math.sqrt(
                    (detection.position[0] - last_pos[0]) ** 2 +
                    (detection.position[1] - last_pos[1]) ** 2
                )
                consistency_bonus = max(0, 1 - distance / 100)
                score += consistency_bonus * 0.3
            
            if score > best_score:
                best_score = score
                best_detection = detection
        
        return best_detection

    def _get_foot_positions(self, pose_results, frame_shape) -> List[Tuple[int, int]]:
        """Extract foot positions from pose detection"""
        if not pose_results or not pose_results.pose_landmarks:
            return []
        
        foot_positions = []
        height, width = frame_shape[:2]
        
        foot_indices = [
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            self.mp_pose.PoseLandmark.LEFT_HEEL,
            self.mp_pose.PoseLandmark.RIGHT_HEEL,
            self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
            self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
        ]
        
        for idx in foot_indices:
            landmark = pose_results.pose_landmarks.landmark[idx]
            if landmark.visibility > 0.5:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                foot_positions.append((x, y))
        
        return foot_positions

    def _check_touch(self, ball_pos: Tuple[int, int], foot_positions: List[Tuple[int, int]], 
                     last_ball_pos: Optional[Tuple[int, int]], timestamp: float, 
                     last_touch_time: float) -> Tuple[bool, float]:
        """Check if ball touch occurred"""
        if timestamp - last_touch_time < self.debounce_time:
            return False, 0.0
        
        if not foot_positions:
            return False, 0.0
        
        min_distance = float('inf')
        for foot_pos in foot_positions:
            distance = np.sqrt((ball_pos[0] - foot_pos[0])**2 + (ball_pos[1] - foot_pos[1])**2)
            min_distance = min(min_distance, distance)
        
        if min_distance > self.touch_threshold_pixels:
            return False, 0.0
        
        # Check ball velocity
        velocity_check = True
        if last_ball_pos:
            velocity = np.sqrt(
                (ball_pos[0] - last_ball_pos[0])**2 + 
                (ball_pos[1] - last_ball_pos[1])**2
            )
            velocity_check = velocity > self.min_ball_velocity
        
        if not velocity_check:
            return False, 0.0
        
        # Calculate confidence
        distance_confidence = 1.0 - (min_distance / self.touch_threshold_pixels)
        
        if distance_confidence > 0.8:
            confidence = 0.9
        elif distance_confidence > 0.6:
            confidence = 0.7
        else:
            confidence = 0.5
        
        return True, confidence

    async def analyze_video(self, video_path: str, video_id: str) -> Dict:
        """Main video analysis pipeline with YOLO integration"""
        start_time = datetime.now()
        
        # Get video metadata
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        reader.close()
        
        metadata = {
            "duration": meta.get("duration", 0),
            "fps": meta.get("fps", 0),
            "size": list(meta.get("size", [0, 0])),
            "format": Path(video_path).suffix
        }
        
        if metadata["duration"] < 5 or metadata["duration"] > 300:
            raise ValueError(f"Video duration {metadata['duration']}s outside allowed range")
        
        logger.info(f"Processing video {video_id} with YOLO + traditional methods")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")
        
        # Initialize tracking
        touch_events: List[TouchEvent] = []
        last_touch_time = -1
        last_ball_pos = None
        frame_count = 0
        processed_frames = 0
        
        # Create output directory
        frames_dir = Path(__file__).parent / "uploads" / "frames" / video_id
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Resize frame
                height, width = frame.shape[:2]
                if width > self.target_width:
                    scale = self.target_width / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                timestamp = frame_count / original_fps
                
                # Pose detection
                pose_results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                foot_positions = self._get_foot_positions(pose_results, frame.shape)
                
                # Hybrid ball detection
                ball_detection = self._detect_ball_hybrid(frame)
                
                # Check for touches
                if ball_detection and foot_positions:
                    touch_detected, confidence = self._check_touch(
                        ball_detection.position, foot_positions, last_ball_pos, timestamp, last_touch_time
                    )
                    
                    if touch_detected:
                        touch_event = TouchEvent(
                            timestamp=timestamp,
                            frame_number=frame_count,
                            confidence=confidence,
                            position=ball_detection.position,
                            detection_method=ball_detection.method
                        )
                        touch_events.append(touch_event)
                        last_touch_time = timestamp
                        
                        # Save debug frame
                        debug_frame = frame.copy()
                        if ball_detection:
                            cv2.circle(debug_frame, ball_detection.position, 20, (0, 255, 0), 3)
                            cv2.putText(debug_frame, f"{ball_detection.method} {ball_detection.confidence:.2f}", 
                                       (ball_detection.position[0] - 40, ball_detection.position[1] - 25),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        for foot_pos in foot_positions:
                            cv2.circle(debug_frame, foot_pos, 15, (255, 0, 0), 3)
                        
                        debug_frame_path = frames_dir / f"touch_{len(touch_events)}.jpg"
                        cv2.imwrite(str(debug_frame_path), debug_frame)
                
                # Update ball history
                if ball_detection:
                    self.ball_history.append(ball_detection)
                    last_ball_pos = ball_detection.position
                
                processed_frames += 1
                frame_count += 1
                
                if processed_frames % 50 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Processing progress: {progress:.1f}%")
        
        finally:
            cap.release()
            self.pose.close()
        
        # Calculate results
        processing_time = (datetime.now() - start_time).total_seconds()
        total_touches = len(touch_events)
        touches_per_minute = (total_touches / metadata["duration"]) * 60 if metadata["duration"] > 0 else 0
        
        # Calculate confidence
        if touch_events:
            avg_confidence = sum(t.confidence for t in touch_events) / len(touch_events)
        else:
            avg_confidence = 0.0
        
        results = {
            "total_ball_touches": total_touches,
            "video_duration": metadata["duration"],
            "touches_per_minute": round(touches_per_minute, 1),
            "confidence_score": round(avg_confidence, 2),
            "processing_time": round(processing_time, 1),
            "touch_events": [t.to_dict() for t in touch_events],
            "metadata": metadata,
            "model_used": "YOLO + Traditional Hybrid",
            "detection_summary": {
                "yolo_available": self.yolo_model is not None,
                "total_detections": len(self.ball_history),
                "detection_methods": {}
            }
        }
        
        # Count detection methods used
        method_counts = {}
        for detection in self.ball_history:
            method = detection.method
            method_counts[method] = method_counts.get(method, 0) + 1
        
        results["detection_summary"]["detection_methods"] = method_counts
        
        logger.info(f"Hybrid analysis complete: {total_touches} touches detected")
        return results

# Test function
async def test_hybrid_processor():
    processor = YOLOVideoProcessor()
    
    # Find test video
    uploads_dir = Path("uploads/raw")
    video_files = list(uploads_dir.glob("*.mp4"))
    
    if not video_files:
        print("❌ No test videos found")
        return
    
    video_path = str(video_files[0])
    video_id = "hybrid_test"
    
    print(f"🚀 Testing hybrid processor on: {video_path}")
    
    try:
        results = await processor.analyze_video(video_path, video_id)
        
        print(f"\n📊 Hybrid Analysis Results:")
        print(f"Total touches detected: {results['total_ball_touches']}")
        print(f"Confidence score: {results['confidence_score']}")
        print(f"Processing time: {results['processing_time']}s")
        print(f"Model used: {results['model_used']}")
        print(f"Detection methods: {results['detection_summary']['detection_methods']}")
        
        # Save results
        results_path = Path("hybrid_test_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_path}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_hybrid_processor())