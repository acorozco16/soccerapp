import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import asyncio
import logging
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import imageio

logger = logging.getLogger(__name__)


@dataclass
class TouchEvent:
    timestamp: float
    frame_number: int
    confidence: float
    position: Tuple[int, int]
    
    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "frame": self.frame_number,
            "confidence": self.confidence,
            "position": self.position
        }


class VideoProcessor:
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
        
        # Ball detection parameters
        self.ball_color_ranges = {
            # Orange ball (HSV ranges)
            "orange": {
                "lower": np.array([5, 50, 50]),
                "upper": np.array([15, 255, 255])
            },
            # White ball (HSV ranges)
            "white": {
                "lower": np.array([0, 0, 200]),
                "upper": np.array([180, 30, 255])
            }
        }
        
        # Touch detection parameters
        self.touch_threshold_pixels = 50  # Distance in pixels
        self.debounce_time = 0.5  # 500ms between touches
        self.min_ball_velocity = 5  # Minimum pixel movement
        
        # Processing parameters
        self.frame_skip = 3  # Process every 3rd frame
        self.target_width = 1280  # 720p width
        self.max_fps = 30  # Cap FPS for processing
        
    async def analyze_video(self, video_path: str, video_id: str) -> Dict:
        """Main video analysis pipeline"""
        start_time = datetime.now()
        
        # Get video metadata
        metadata = self._get_video_metadata(video_path)
        if metadata["duration"] < 10 or metadata["duration"] > 300:
            raise ValueError(f"Video duration {metadata['duration']}s outside allowed range (10-300s)")
        
        # Process video
        logger.info(f"Processing video {video_id}: {metadata['duration']}s @ {metadata['fps']}fps")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")
        
        # Calculate processing parameters
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        process_fps = min(original_fps, self.max_fps)
        
        # Initialize tracking variables
        touch_events: List[TouchEvent] = []
        last_touch_time = -1
        last_ball_pos = None
        frame_count = 0
        processed_frames = 0
        
        # Create output directory for debug frames
        frames_dir = Path(__file__).parent.parent / "uploads" / "frames" / video_id
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        debug_frames = []
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames for efficiency
                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Resize frame if needed
                frame = self._resize_frame(frame)
                
                # Get current timestamp
                timestamp = frame_count / original_fps
                
                # Detect ball
                ball_pos, ball_confidence = self._detect_ball(frame)
                
                # Detect pose and foot positions
                pose_results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                foot_positions = self._get_foot_positions(pose_results, frame.shape)
                
                # Check for touches
                if ball_pos and foot_positions:
                    touch_detected, confidence = self._check_touch(
                        ball_pos, foot_positions, last_ball_pos, timestamp, last_touch_time
                    )
                    
                    if touch_detected:
                        touch_event = TouchEvent(
                            timestamp=timestamp,
                            frame_number=frame_count,
                            confidence=confidence,
                            position=ball_pos
                        )
                        touch_events.append(touch_event)
                        last_touch_time = timestamp
                        
                        # Save debug frame for this touch
                        debug_frame = self._create_debug_frame(
                            frame.copy(), ball_pos, foot_positions, pose_results
                        )
                        debug_frame_path = frames_dir / f"touch_{len(touch_events)}.jpg"
                        cv2.imwrite(str(debug_frame_path), debug_frame)
                        debug_frames.append(f"touch_{len(touch_events)}.jpg")
                
                # Update ball position
                if ball_pos:
                    last_ball_pos = ball_pos
                
                # Save first and last frame for reference
                if processed_frames == 0:
                    debug_frame = self._create_debug_frame(
                        frame.copy(), ball_pos, foot_positions, pose_results
                    )
                    cv2.imwrite(str(frames_dir / "first_frame.jpg"), debug_frame)
                    debug_frames.insert(0, "first_frame.jpg")
                
                processed_frames += 1
                frame_count += 1
                
                # Progress logging
                if processed_frames % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Processing progress: {progress:.1f}%")
            
            # Save last frame
            if frame is not None:
                debug_frame = self._create_debug_frame(
                    frame.copy(), ball_pos, foot_positions, pose_results
                )
                cv2.imwrite(str(frames_dir / "last_frame.jpg"), debug_frame)
                if "last_frame.jpg" not in debug_frames:
                    debug_frames.append("last_frame.jpg")
            
        finally:
            cap.release()
            self.pose.close()
        
        # Calculate results
        processing_time = (datetime.now() - start_time).total_seconds()
        total_touches = len(touch_events)
        touches_per_minute = (total_touches / metadata["duration"]) * 60 if metadata["duration"] > 0 else 0
        
        # Calculate confidence score
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
            "debug_frames": debug_frames[:5],  # Limit to 5 frames
            "metadata": metadata
        }
        
        logger.info(f"Analysis complete: {total_touches} touches detected with {avg_confidence:.2f} confidence")
        return results
    
    def _get_video_metadata(self, video_path: str) -> Dict:
        """Extract video metadata"""
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        reader.close()
        
        return {
            "duration": meta.get("duration", 0),
            "fps": meta.get("fps", 0),
            "size": list(meta.get("size", [0, 0])),
            "format": Path(video_path).suffix
        }
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target width while maintaining aspect ratio"""
        height, width = frame.shape[:2]
        if width > self.target_width:
            scale = self.target_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height))
        return frame
    
    def _detect_ball(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int]], float]:
        """Detect soccer ball using color filtering and Hough circles"""
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create combined mask for orange and white balls
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for color_name, ranges in self.ball_color_ranges.items():
            color_mask = cv2.inRange(hsv, ranges["lower"], ranges["upper"])
            mask = cv2.bitwise_or(mask, color_mask)
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply Gaussian blur for better circle detection
        blurred = cv2.GaussianBlur(mask, (9, 9), 2)
        
        # Detect circles using Hough transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=50
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Return the first (most prominent) circle
            x, y, r = circles[0, 0]
            
            # Calculate confidence based on how well the circle fits the mask
            circle_mask = np.zeros_like(mask)
            cv2.circle(circle_mask, (x, y), r, 255, -1)
            overlap = cv2.bitwise_and(mask, circle_mask)
            confidence = np.sum(overlap) / (np.pi * r * r * 255) if r > 0 else 0
            
            return (int(x), int(y)), float(confidence)
        
        # Fallback: use contour detection if no circles found
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find the largest circular contour
            best_contour = None
            best_circularity = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100:  # Skip small contours
                    continue
                
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > best_circularity and circularity > 0.5:
                        best_contour = contour
                        best_circularity = circularity
            
            if best_contour is not None:
                M = cv2.moments(best_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy), best_circularity * 0.7  # Lower confidence for contour detection
        
        return None, 0.0
    
    def _get_foot_positions(self, pose_results, frame_shape) -> List[Tuple[int, int]]:
        """Extract foot positions from pose detection"""
        if not pose_results or not pose_results.pose_landmarks:
            return []
        
        foot_positions = []
        height, width = frame_shape[:2]
        
        # MediaPipe foot landmark indices
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
            if landmark.visibility > 0.5:  # Only use visible landmarks
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                foot_positions.append((x, y))
        
        return foot_positions
    
    def _check_touch(self, ball_pos: Tuple[int, int], foot_positions: List[Tuple[int, int]], 
                     last_ball_pos: Optional[Tuple[int, int]], timestamp: float, 
                     last_touch_time: float) -> Tuple[bool, float]:
        """Check if ball touch occurred"""
        # Check debounce time
        if timestamp - last_touch_time < self.debounce_time:
            return False, 0.0
        
        # Calculate minimum distance to any foot
        if not foot_positions:
            return False, 0.0
        
        min_distance = float('inf')
        for foot_pos in foot_positions:
            distance = np.sqrt((ball_pos[0] - foot_pos[0])**2 + (ball_pos[1] - foot_pos[1])**2)
            min_distance = min(min_distance, distance)
        
        # Check if ball is close enough to foot
        if min_distance > self.touch_threshold_pixels:
            return False, 0.0
        
        # Check ball velocity if we have previous position
        velocity_check = True
        if last_ball_pos:
            velocity = np.sqrt(
                (ball_pos[0] - last_ball_pos[0])**2 + 
                (ball_pos[1] - last_ball_pos[1])**2
            )
            velocity_check = velocity > self.min_ball_velocity
        
        if not velocity_check:
            return False, 0.0
        
        # Calculate confidence based on distance
        distance_confidence = 1.0 - (min_distance / self.touch_threshold_pixels)
        
        # Determine confidence level
        if distance_confidence > 0.8:
            confidence = 0.9  # Definite touch
        elif distance_confidence > 0.6:
            confidence = 0.7  # Probable touch
        else:
            confidence = 0.5  # Possible touch
        
        return True, confidence
    
    def _create_debug_frame(self, frame: np.ndarray, ball_pos: Optional[Tuple[int, int]], 
                           foot_positions: List[Tuple[int, int]], pose_results) -> np.ndarray:
        """Create annotated debug frame"""
        # Draw pose skeleton
        if pose_results and pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
        
        # Draw ball position
        if ball_pos:
            cv2.circle(frame, ball_pos, 20, (0, 0, 255), 3)
            cv2.putText(frame, "Ball", (ball_pos[0] - 20, ball_pos[1] - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw foot positions
        for foot_pos in foot_positions:
            cv2.circle(frame, foot_pos, 15, (255, 0, 0), 3)
        
        # Add timestamp
        cv2.putText(frame, f"Soccer Touch Analysis", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame