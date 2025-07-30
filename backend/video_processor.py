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
from bytetrack_tracker import BYTETracker, Detection

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


class VideoQuality(NamedTuple):
    brightness: float
    contrast: float
    blur_score: float
    shake_score: float
    overall_score: float
    needs_review: bool
    issues: List[str]


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
        
        # Load YOLO v8 model (prioritize v8, fallback to custom trained)
        try:
            # Try YOLO v8 first for better performance
            self.yolo_model = YOLO('yolov8n.pt')  # Nano for speed, can upgrade to yolov8s.pt
            self.yolo_confidence_threshold = 0.1  # Lower for ByteTrack two-stage matching
            print(f"✅ Loaded YOLO v8 model")
            logger.info(f"Loaded YOLO v8 model with confidence threshold {self.yolo_confidence_threshold}")
            
            # Also try custom trained model as backup
            custom_model_path = Path(__file__).parent.parent / "models/soccer_ball_trained.pt"
            if custom_model_path.exists():
                self.custom_yolo_model = YOLO(str(custom_model_path))
                print(f"✅ Also loaded custom trained model as backup: {custom_model_path}")
            else:
                self.custom_yolo_model = None
                
        except Exception as e:
            print(f"❌ Failed to load YOLO v8, trying custom model: {e}")
            try:
                model_path = Path(__file__).parent.parent / "models/soccer_ball_trained.pt"
                if model_path.exists():
                    self.yolo_model = YOLO(str(model_path))
                    self.yolo_confidence_threshold = 0.05
                    self.custom_yolo_model = None
                    print(f"✅ Loaded custom trained YOLO model: {model_path}")
                    logger.info(f"Loaded custom trained YOLO model with confidence threshold {self.yolo_confidence_threshold}")
                else:
                    print(f"⚠️ No YOLO models found, falling back to traditional detection")
                    logger.warning("No YOLO models found, using traditional detection")
                    self.yolo_model = None
                    self.custom_yolo_model = None
            except Exception as e2:
                print(f"❌ Failed to load any YOLO model: {e2}")
                logger.error(f"Failed to load any YOLO model: {e2}")
                self.yolo_model = None
                self.custom_yolo_model = None
        
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
            "white_dirty": {
                "lower": np.array([0, 0, 150]),
                "upper": np.array([180, 50, 230])
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
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.last_known_position = None
        self.frames_without_detection = 0
        
        # Phase 2: Trajectory prediction for missed balls
        self.trajectory_history = deque(maxlen=8)  # Store recent positions with timestamps
        self.prediction_confidence_threshold = 0.5  # Increased from 0.3 to be more competitive
        
        # ByteTrack integration - Optimized for soccer ball tracking
        self.bytetrack_tracker = BYTETracker(
            frame_rate=30,
            track_thresh=0.4,       # Lower threshold - soccer balls can have variable confidence
            track_buffer=60,        # Longer buffer for fast-moving balls
            match_thresh=0.7,       # Slightly lower for better association
            high_thresh=0.5,        # Lower high threshold to capture more detections
            low_thresh=0.05         # Very low - catch even weak detections
        )
        
        # Quality assessment thresholds
        self.quality_thresholds = {
            "min_brightness": 40,
            "max_brightness": 220,
            "min_contrast": 30,
            "max_blur_score": 100,
            "max_shake_score": 50
        }

    def _detect_ball_yolo_v8(self, frame: np.ndarray) -> List[Detection]:
        """YOLO v8 ball detection with low confidence threshold for ByteTrack"""
        detections = []
        
        if self.yolo_model is None:
            return detections
        
        try:
            # Run YOLO inference with optimized settings for ByteTrack
            results = self.yolo_model(frame, conf=0.01, iou=0.3, verbose=False)  # Ultra-low conf, lower IoU
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Extract box coordinates and confidence
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Filter for sports ball class (class 32 in COCO)
                        class_id = int(box.cls[0])
                        if class_id == 32:  # Sports ball in COCO dataset
                            detections.append(Detection(
                                bbox=(float(x1), float(y1), float(x2), float(y2)),
                                confidence=conf,
                                class_id=class_id
                            ))
        
        except Exception as e:
            logger.warning(f"YOLO v8 detection failed: {e}")
        
        # Fallback to custom model if available
        if not detections and self.custom_yolo_model is not None:
            try:
                results = self.custom_yolo_model(frame, conf=0.05, verbose=False)
                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detections.append(Detection(
                                bbox=(float(x1), float(y1), float(x2), float(y2)),
                                confidence=conf,
                                class_id=0  # Custom model class
                            ))
            except Exception as e:
                logger.warning(f"Custom YOLO detection failed: {e}")
        
        return detections

    def _detect_ball_yolo_legacy(self, frame: np.ndarray) -> List[BallDetection]:
        """Legacy YOLO detection for backwards compatibility"""
        yolo_detections = self._detect_ball_yolo_v8(frame)
        
        # Convert to legacy format
        legacy_detections = []
        for det in yolo_detections:
            x1, y1, x2, y2 = det.bbox
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            radius = int(max(x2 - x1, y2 - y1) / 2)
            
            legacy_detections.append(BallDetection(
                position=(center_x, center_y),
                confidence=det.confidence,
                method="yolo_v8",
                radius=radius
            ))
        
        return legacy_detections

    def _detect_ball_traditional(self, frame: np.ndarray) -> List[BallDetection]:
        """Traditional HSV + Hough circle detection as fallback"""
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
        
        for i, params in enumerate(self.hough_params):
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
                            method=f"hough_{i+1}",
                            radius=int(r)
                        ))
        
        return detections

    def _choose_best_detection(self, detections: List[BallDetection], foot_positions: List[Tuple[int, int]]) -> Optional[BallDetection]:
        """Choose best detection with preference for YOLO v2"""
        if not detections:
            return None
        
        if len(detections) == 1:
            return detections[0]
        
        # Score detections with method preference
        best_detection = None
        best_score = -1
        
        for detection in detections:
            score = detection.confidence
            
            # Method priority weights (prefer our improved YOLO model)
            method_weights = {
                "yolo_v2": 10.0,                # Very strong preference for YOLO model
                "hough_1": 1.0,
                "hough_2": 0.9,
                "contour": 0.8,
                "motion": 0.7,
                "trajectory_prediction": 2.0,   # Phase 2: Higher priority for physics prediction
                "prediction": 0.5
            }
            
            score *= method_weights.get(detection.method, 1.0)
            
            # Bonus for consistency with history
            if self.ball_history:
                last_pos = self.ball_history[-1].position
                distance = math.sqrt(
                    (detection.position[0] - last_pos[0]) ** 2 +
                    (detection.position[1] - last_pos[1]) ** 2
                )
                consistency_bonus = max(0, 1 - distance / 100)
                score += consistency_bonus * 0.3
            
            # Bonus for proximity to players
            if foot_positions:
                min_dist_to_player = min(
                    math.sqrt((detection.position[0] - fp[0]) ** 2 + (detection.position[1] - fp[1]) ** 2)
                    for fp in foot_positions
                )
                proximity_bonus = max(0, 1 - min_dist_to_player / 200)
                score += proximity_bonus * 0.2
            
            if score > best_score:
                best_score = score
                best_detection = detection
        
        return best_detection

    def _predict_ball_position(self, timestamp: float) -> Optional[BallDetection]:
        """Phase 2: Predict ball position using trajectory physics when YOLO fails"""
        if len(self.trajectory_history) < 3:
            return None
        
        # Get last 3 positions for physics calculation
        recent_positions = list(self.trajectory_history)[-3:]
        
        # Calculate time intervals
        dt1 = recent_positions[1]['timestamp'] - recent_positions[0]['timestamp']
        dt2 = recent_positions[2]['timestamp'] - recent_positions[1]['timestamp']
        dt_predict = timestamp - recent_positions[2]['timestamp']
        
        if dt1 <= 0 or dt2 <= 0 or dt_predict <= 0 or dt_predict > 0.2:  # Max 0.2s prediction
            return None
        
        # Physics-based prediction using position, velocity, and acceleration
        pos1 = recent_positions[0]['position']
        pos2 = recent_positions[1]['position'] 
        pos3 = recent_positions[2]['position']
        
        # Calculate velocities
        vel1_x = (pos2[0] - pos1[0]) / dt1
        vel1_y = (pos2[1] - pos1[1]) / dt1
        vel2_x = (pos3[0] - pos2[0]) / dt2
        vel2_y = (pos3[1] - pos2[1]) / dt2
        
        # Calculate acceleration
        accel_x = (vel2_x - vel1_x) / dt2
        accel_y = (vel2_y - vel1_y) / dt2
        
        # Add gravity effect (pixels/second²)
        gravity_effect = 200 * dt_predict * dt_predict  # Approximate gravity in pixel space
        
        # Predict position using kinematic equations
        predicted_x = pos3[0] + vel2_x * dt_predict + 0.5 * accel_x * dt_predict * dt_predict
        predicted_y = pos3[1] + vel2_y * dt_predict + 0.5 * accel_y * dt_predict * dt_predict + gravity_effect
        
        # Validate prediction is reasonable (within frame bounds with margin)
        if predicted_x < -100 or predicted_x > 1380 or predicted_y < -100 or predicted_y > 900:
            return None
        
        # Calculate confidence based on trajectory consistency
        trajectory_consistency = self._calculate_trajectory_consistency(recent_positions)
        confidence = min(self.prediction_confidence_threshold + trajectory_consistency * 0.4, 0.9)  # Increased max confidence
        
        return BallDetection(
            position=(int(predicted_x), int(predicted_y)),
            confidence=confidence,
            method="trajectory_prediction",
            radius=15  # Estimated radius
        )
    
    def _calculate_trajectory_consistency(self, positions: List[Dict]) -> float:
        """Calculate how consistent the ball trajectory is for prediction confidence"""
        if len(positions) < 3:
            return 0.0
        
        # Check if positions form a smooth trajectory
        distances = []
        for i in range(1, len(positions)):
            dist = math.sqrt(
                (positions[i]['position'][0] - positions[i-1]['position'][0]) ** 2 +
                (positions[i]['position'][1] - positions[i-1]['position'][1]) ** 2
            )
            distances.append(dist)
        
        # More consistent distances = higher confidence
        avg_distance = sum(distances) / len(distances)
        variance = sum((d - avg_distance) ** 2 for d in distances) / len(distances)
        
        # Convert variance to confidence (lower variance = higher confidence)
        consistency = max(0, 1 - variance / (avg_distance ** 2)) if avg_distance > 0 else 0
        return min(consistency, 1.0)

    def _detect_ball(self, frame: np.ndarray, frame_number: int, foot_positions: List[Tuple[int, int]], timestamp: float = 0.0) -> Optional[BallDetection]:
        """Enhanced ball detection with YOLO v2 + traditional fallback + trajectory prediction"""
        all_detections = []
        
        # Primary: YOLO v8 detection (legacy format)
        yolo_detections = self._detect_ball_yolo_legacy(frame)
        all_detections.extend(yolo_detections)
        
        # Always run both YOLO and traditional detection, let best selection decide
        traditional_detections = self._detect_ball_traditional(frame)
        all_detections.extend(traditional_detections)
        
        # Phase 2: Try trajectory prediction more aggressively
        if not all_detections or max(d.confidence for d in all_detections) < 0.6:
            predicted_detection = self._predict_ball_position(timestamp)
            if predicted_detection:
                all_detections.append(predicted_detection)
        
        # Choose best detection
        best_detection = self._choose_best_detection(all_detections, foot_positions)
        
        # Update tracking and trajectory history
        if best_detection:
            self.frames_without_detection = 0
            self.ball_history.append(best_detection)
            self.last_known_position = best_detection.position
            
            # Update trajectory history for physics prediction
            self.trajectory_history.append({
                'timestamp': timestamp,
                'position': best_detection.position,
                'confidence': best_detection.confidence,
                'method': best_detection.method
            })
        else:
            self.frames_without_detection += 1
        
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
    
    def _infer_trajectory_touch(self, timestamp: float, foot_positions: List[Tuple[int, int]]) -> Optional[TouchEvent]:
        """Phase 2b: Infer touches from ball trajectory changes near feet"""
        if len(self.trajectory_history) < 4 or not foot_positions:
            return None
        
        # Get recent trajectory points
        recent_points = list(self.trajectory_history)[-4:]
        
        # Calculate trajectory change (direction change indicates contact)
        direction_changes = []
        for i in range(1, len(recent_points)-1):
            prev_pos = recent_points[i-1]['position']
            curr_pos = recent_points[i]['position']
            next_pos = recent_points[i+1]['position']
            
            # Calculate direction vectors
            vec1 = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
            vec2 = (next_pos[0] - curr_pos[0], next_pos[1] - curr_pos[1])
            
            # Calculate angle change (dot product)
            if (vec1[0] != 0 or vec1[1] != 0) and (vec2[0] != 0 or vec2[1] != 0):
                dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
                mag1 = (vec1[0]**2 + vec1[1]**2)**0.5
                mag2 = (vec2[0]**2 + vec2[1]**2)**0.5
                
                if mag1 > 0 and mag2 > 0:
                    cos_angle = dot_product / (mag1 * mag2)
                    angle_change = abs(1 - cos_angle)  # 0 = no change, 2 = complete reversal
                    
                    # Check if ball was near foot during direction change
                    min_foot_distance = min(
                        ((curr_pos[0] - fp[0])**2 + (curr_pos[1] - fp[1])**2)**0.5
                        for fp in foot_positions
                    )
                    
                    if angle_change > 0.3 and min_foot_distance < 100:  # More sensitive thresholds
                        logger.info(f"🎯 Trajectory touch inferred: angle_change={angle_change:.2f}, distance={min_foot_distance:.1f}")
                        return TouchEvent(
                            timestamp=recent_points[i]['timestamp'],
                            frame_number=int(recent_points[i]['timestamp'] * 60),  # Estimate frame
                            confidence=min(0.8, angle_change * 0.9 + 0.1),
                            position=curr_pos,
                            detection_method="trajectory_inference"
                        )
        
        return None

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
    
    def _detect_video_orientation(self, frame: np.ndarray) -> str:
        """Detect if video is upside down or rotated using pose detection"""
        try:
            # Run pose detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(rgb_frame)
            
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                
                # Get key landmarks (normalized coordinates 0-1)
                nose_y = landmarks[self.mp_pose.PoseLandmark.NOSE].y
                left_ankle_y = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y
                right_ankle_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y
                
                # Average foot position
                avg_foot_y = (left_ankle_y + right_ankle_y) / 2
                
                # Check if head is below feet (upside down)
                if nose_y > avg_foot_y + 0.1:  # 0.1 threshold for confidence
                    return "upside_down"
                
                # Check for sideways rotation using shoulders
                left_shoulder_x = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x
                right_shoulder_x = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x
                left_shoulder_y = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y
                right_shoulder_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
                
                # Check if shoulders are more vertical than horizontal (sideways)
                shoulder_height_diff = abs(left_shoulder_y - right_shoulder_y)
                shoulder_width_diff = abs(left_shoulder_x - right_shoulder_x)
                
                if shoulder_height_diff > shoulder_width_diff * 1.5:
                    # Determine which way to rotate
                    if left_shoulder_y < right_shoulder_y:
                        return "rotated_left"
                    else:
                        return "rotated_right"
            
            return "normal"
            
        except Exception as e:
            logger.warning(f"Orientation detection failed: {e}")
            return "normal"
    
    def _standardize_orientation(self, frame: np.ndarray, orientation: str) -> np.ndarray:
        """Rotate frame to standard upright orientation"""
        if orientation == "upside_down":
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif orientation == "rotated_left":
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == "rotated_right":
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return frame
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target width while maintaining aspect ratio"""
        height, width = frame.shape[:2]
        if width > self.target_width:
            scale = self.target_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height))
        return frame
    
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
    
    def _assess_video_quality(self, frame: np.ndarray) -> VideoQuality:
        """Assess video quality and identify issues"""
        issues = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Brightness assessment
        brightness = np.mean(gray)
        if brightness < self.quality_thresholds["min_brightness"]:
            issues.append("Low brightness - video may be too dark")
        elif brightness > self.quality_thresholds["max_brightness"]:
            issues.append("High brightness - video may be overexposed")
        
        # Contrast assessment
        contrast = np.std(gray)
        if contrast < self.quality_thresholds["min_contrast"]:
            issues.append("Low contrast - details may be hard to distinguish")
        
        # Blur assessment
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < self.quality_thresholds["max_blur_score"]:
            issues.append("Video appears blurry or out of focus")
        
        # Shake detection
        shake_score = np.std(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3))
        
        # Overall quality score
        brightness_score = 1 - abs(brightness - 128) / 128
        contrast_score = min(contrast / 100, 1.0)
        blur_score_norm = min(blur_score / 200, 1.0)
        shake_score_norm = max(0, 1 - shake_score / 200)
        
        overall_score = (brightness_score + contrast_score + blur_score_norm + shake_score_norm) / 4
        needs_review = overall_score < 0.6 or len(issues) >= 2
        
        return VideoQuality(
            brightness=brightness,
            contrast=contrast,
            blur_score=blur_score,
            shake_score=shake_score,
            overall_score=overall_score,
            needs_review=needs_review,
            issues=issues
        )

    def _smooth_trajectory(self, touch_events: List[TouchEvent]) -> List[TouchEvent]:
        """Apply trajectory smoothing to remove detection noise"""
        if len(touch_events) <= 2:
            return touch_events
        
        smoothed_events = []
        window_size = 3
        
        for i, event in enumerate(touch_events):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(touch_events), i + window_size // 2 + 1)
            neighbors = touch_events[start_idx:end_idx]
            
            # Check for outliers
            distances = []
            for neighbor in neighbors:
                if neighbor != event:
                    distance = math.sqrt(
                        (event.position[0] - neighbor.position[0]) ** 2 +
                        (event.position[1] - neighbor.position[1]) ** 2
                    )
                    distances.append(distance)
            
            # Remove outliers
            if distances and min(distances) > 100:
                logger.info(f"Removing outlier touch at {event.timestamp:.1f}s")
                continue
            
            # Temporal smoothing
            if i > 0:
                time_diff = event.timestamp - touch_events[i-1].timestamp
                if time_diff < 0.3:
                    prev_pos = touch_events[i-1].position
                    smoothed_x = int((event.position[0] + prev_pos[0]) / 2)
                    smoothed_y = int((event.position[1] + prev_pos[1]) / 2)
                    
                    smoothed_event = TouchEvent(
                        timestamp=event.timestamp,
                        frame_number=event.frame_number,
                        confidence=max(event.confidence, touch_events[i-1].confidence),
                        position=(smoothed_x, smoothed_y),
                        detection_method=event.detection_method
                    )
                    smoothed_events.append(smoothed_event)
                else:
                    smoothed_events.append(event)
            else:
                smoothed_events.append(event)
        
        return smoothed_events

    def _calculate_touch_range(self, detected_touches: int, confidence: float, quality_assessment: Optional[VideoQuality]) -> Dict:
        """Calculate touch count range based on detection confidence and video quality"""
        
        # Base uncertainty factors
        base_uncertainty = 2  # ±2 touches base uncertainty
        
        # Adjust uncertainty based on confidence score
        if confidence >= 0.8:
            confidence_factor = 0.5  # High confidence: smaller range
        elif confidence >= 0.6:
            confidence_factor = 1.0  # Medium confidence: normal range
        else:
            confidence_factor = 1.5  # Low confidence: larger range
        
        # Adjust uncertainty based on video quality
        quality_factor = 1.0
        if quality_assessment:
            if quality_assessment.overall_score >= 0.8:
                quality_factor = 0.8  # High quality: smaller range
            elif quality_assessment.overall_score >= 0.6:
                quality_factor = 1.0  # Good quality: normal range
            else:
                quality_factor = 1.3  # Poor quality: larger range
        
        # Calculate range bounds
        uncertainty = int(base_uncertainty * confidence_factor * quality_factor)
        uncertainty = max(1, min(uncertainty, 5))  # Clamp between 1-5
        
        # Calculate range ensuring it doesn't go below 0
        range_min = max(0, detected_touches - uncertainty)
        range_max = detected_touches + uncertainty
        
        # Special handling for very low counts
        if detected_touches <= 3:
            range_min = max(0, detected_touches - 1)
            range_max = detected_touches + 2
        
        return {
            "min": range_min,
            "max": range_max,
            "display": f"{range_min}-{range_max} touches",
            "detected_count": detected_touches,
            "confidence_level": "high" if confidence >= 0.7 else "medium" if confidence >= 0.5 else "low",
            "explanation": f"Detected {detected_touches} touches with {confidence:.0%} confidence"
        }

    def _get_detection_summary(self) -> Dict:
        """Get summary of detection methods used"""
        if not self.ball_history:
            return {"total_detections": 0, "methods": {}}
        
        method_counts = {}
        for detection in self.ball_history:
            method = detection.method
            method_counts[method] = method_counts.get(method, 0) + 1
        
        return {
            "total_detections": len(self.ball_history),
            "methods": method_counts,
            "success_rate": len(self.ball_history) / max(1, len(self.ball_history) + self.frames_without_detection),
            "model_used": "YOLO v2 + Traditional" if self.yolo_model else "Traditional Only"
        }

    def _create_debug_frame(self, frame: np.ndarray, ball_pos: Optional[Tuple[int, int]], 
                           foot_positions: List[Tuple[int, int]], pose_results, 
                           ball_detection: Optional[BallDetection] = None) -> np.ndarray:
        """Create annotated debug frame"""
        # Draw pose skeleton
        if pose_results and pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
        
        # Draw ball position with method info
        if ball_pos:
            # Method-specific colors
            method_colors = {
                "yolo_v2": (0, 255, 0),             # Green for improved YOLO
                "hough_1": (0, 0, 255),             # Red
                "hough_2": (0, 100, 255),           # Orange-red
                "contour": (0, 255, 255),           # Yellow
                "motion": (255, 0, 255),            # Magenta
                "trajectory_prediction": (255, 165, 0),  # Orange for physics prediction
                "prediction": (128, 128, 128)       # Gray
            }
            
            color = method_colors.get(
                ball_detection.method if ball_detection else "unknown", 
                (0, 0, 255)
            )
            
            # Draw circle
            cv2.circle(frame, ball_pos, 20, color, 3)
            
            # Add method and confidence info
            method_text = ball_detection.method if ball_detection else "unknown"
            confidence_text = f"{ball_detection.confidence:.2f}" if ball_detection else "0.00"
            
            cv2.putText(frame, f"Ball ({method_text})", 
                       (ball_pos[0] - 40, ball_pos[1] - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(frame, f"Conf: {confidence_text}", 
                       (ball_pos[0] - 30, ball_pos[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw foot positions
        for foot_pos in foot_positions:
            cv2.circle(frame, foot_pos, 15, (255, 0, 0), 3)
        
        # Enhanced header
        cv2.putText(frame, f"Enhanced Soccer Analysis (YOLO v2)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame

    async def analyze_video(self, video_path: str, video_id: str) -> Dict:
        """Main video analysis pipeline with improved YOLO model"""
        start_time = datetime.now()
        
        # Get video metadata
        metadata = self._get_video_metadata(video_path)
        if metadata["duration"] < 10 or metadata["duration"] > 300:
            raise ValueError(f"Video duration {metadata['duration']}s outside allowed range (10-300s)")
        
        # Process video
        logger.info(f"Processing video {video_id} with improved model: {metadata['duration']}s @ {metadata['fps']}fps")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")
        
        # Initialize tracking variables
        touch_events: List[TouchEvent] = []
        last_touch_time = -1
        last_ball_pos = None
        frame_count = 0
        processed_frames = 0
        
        # Smart sampling: Track high-resolution processing
        self._high_res_frames_remaining = 0
        
        # Video orientation tracking
        video_orientation = None
        orientation_detected = False
        
        # Create output directory for debug frames
        frames_dir = Path(__file__).parent.parent / "uploads" / "frames" / video_id
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        debug_frames = []
        
        # Calculate processing parameters
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # SMART FRAME SAMPLING - Process all frames when ball might be near feet
                should_process_frame = False
                
                # Always process every Nth frame (normal sampling)
                if frame_count % self.frame_skip == 0:
                    should_process_frame = True
                
                # CRITICAL: Also process if we're in high-resolution mode
                # (This will be set when ball is detected near feet)
                elif hasattr(self, '_high_res_frames_remaining') and self._high_res_frames_remaining > 0:
                    should_process_frame = True
                    self._high_res_frames_remaining -= 1
                
                # Skip this frame if neither condition is met
                if not should_process_frame:
                    frame_count += 1
                    continue
                
                # Detect video orientation on first few frames
                if not orientation_detected and processed_frames < 3:
                    detected_orientation = self._detect_video_orientation(frame)
                    if detected_orientation != "normal":
                        video_orientation = detected_orientation
                        logger.info(f"Video orientation detected: {video_orientation}")
                        print(f"🔄 Video orientation: {video_orientation} - will auto-correct")
                    elif processed_frames >= 2:
                        video_orientation = "normal"
                        orientation_detected = True
                        logger.info("Video orientation: normal")
                
                # Standardize frame orientation
                if video_orientation and video_orientation != "normal":
                    frame = self._standardize_orientation(frame, video_orientation)
                
                # Resize frame if needed
                frame = self._resize_frame(frame)
                
                # Get current timestamp
                timestamp = frame_count / original_fps
                
                # Detect pose and foot positions
                pose_results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                foot_positions = self._get_foot_positions(pose_results, frame.shape)
                
                # PHASE 1: ByteTrack Detection + Tracking
                yolo_detections = self._detect_ball_yolo_v8(frame)
                tracks = self.bytetrack_tracker.update(yolo_detections)
                
                # Convert best track to legacy ball detection format with enhanced logic
                ball_detection = None
                if tracks:
                    # Multi-track logic: prioritize longest tracks and foot proximity
                    if foot_positions:
                        # Find track closest to feet (likely the ball being juggled)
                        best_track = None
                        min_foot_distance = float('inf')
                        
                        for track in tracks:
                            x1, y1, x2, y2 = track.bbox
                            track_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                            
                            # Find distance to nearest foot
                            for foot_x, foot_y in foot_positions:
                                distance = ((track_center[0] - foot_x)**2 + (track_center[1] - foot_y)**2)**0.5
                                if distance < min_foot_distance:
                                    min_foot_distance = distance
                                    best_track = track
                        
                        # If no track near feet, use highest confidence
                        if best_track is None:
                            best_track = max(tracks, key=lambda t: t.confidence)
                    else:
                        # No pose detected, use highest confidence track
                        best_track = max(tracks, key=lambda t: t.confidence)
                    
                    x1, y1, x2, y2 = best_track.bbox
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    radius = int(max(x2 - x1, y2 - y1) / 2)
                    
                    ball_detection = BallDetection(
                        position=(center_x, center_y),
                        confidence=best_track.confidence,
                        method="bytetrack_yolo_v8",
                        radius=radius
                    )
                
                # Fallback to legacy detection if no tracks
                if ball_detection is None:
                    ball_detection = self._detect_ball(frame, frame_count, foot_positions, timestamp)
                
                # ENHANCED SMART SAMPLING: More aggressive for ByteTrack
                if ball_detection and foot_positions:
                    ball_x, ball_y = ball_detection.position
                    for foot_x, foot_y in foot_positions:
                        distance = ((ball_x - foot_x)**2 + (ball_y - foot_y)**2)**0.5
                        # If ball within 200 pixels of any foot, process next 45 frames
                        if distance < 200:  # Increased range
                            self._high_res_frames_remaining = 45  # ~1.5 seconds at 30fps
                            break
                
                # ADDITIONAL: If we have multiple tracks, stay in high-res mode
                if tracks and len(tracks) > 1:
                    self._high_res_frames_remaining = max(self._high_res_frames_remaining, 15)
                
                # Check for touches (traditional method)
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
                        
                        # Save debug frame for this touch
                        debug_frame = self._create_debug_frame(
                            frame.copy(), ball_detection.position, foot_positions, pose_results, ball_detection
                        )
                        debug_frame_path = frames_dir / f"touch_{len(touch_events)}.jpg"
                        cv2.imwrite(str(debug_frame_path), debug_frame)
                        debug_frames.append(f"touch_{len(touch_events)}.jpg")
                
                # Phase 2b: Check for trajectory-based touches (catches missed touches)
                if processed_frames % 5 == 0:  # Check every 5 frames for more sensitivity
                    trajectory_touch = self._infer_trajectory_touch(timestamp, foot_positions)
                    if trajectory_touch and timestamp - last_touch_time > self.debounce_time:
                        touch_events.append(trajectory_touch)
                        last_touch_time = trajectory_touch.timestamp
                        
                        # Create debug frame for trajectory-inferred touch
                        debug_frame = self._create_debug_frame(
                            frame.copy(), trajectory_touch.position, foot_positions, pose_results
                        )
                        debug_frame_path = frames_dir / f"touch_{len(touch_events)}_trajectory.jpg"
                        cv2.imwrite(str(debug_frame_path), debug_frame)
                        debug_frames.append(f"touch_{len(touch_events)}_trajectory.jpg")
                
                # Update ball position
                if ball_detection:
                    last_ball_pos = ball_detection.position
                
                # Save reference frames
                if processed_frames == 0:
                    debug_frame = self._create_debug_frame(
                        frame.copy(), 
                        ball_detection.position if ball_detection else None, 
                        foot_positions, 
                        pose_results,
                        ball_detection
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
                    frame.copy(), 
                    ball_detection.position if ball_detection else None, 
                    foot_positions, 
                    pose_results,
                    ball_detection
                )
                cv2.imwrite(str(frames_dir / "last_frame.jpg"), debug_frame)
                if "last_frame.jpg" not in debug_frames:
                    debug_frames.append("last_frame.jpg")
            
        finally:
            cap.release()
            self.pose.close()
        
        # Apply trajectory smoothing
        smoothed_touch_events = self._smooth_trajectory(touch_events)
        
        # Assess video quality
        quality_assessment = self._assess_video_quality(frame) if frame is not None else None
        
        # Calculate results
        processing_time = (datetime.now() - start_time).total_seconds()
        total_touches = len(smoothed_touch_events)
        touches_per_minute = (total_touches / metadata["duration"]) * 60 if metadata["duration"] > 0 else 0
        
        # Calculate confidence score with method weighting
        if smoothed_touch_events:
            method_weights = {
                "yolo_v2": 1.2,                # Higher weight for improved model
                "hough_1": 1.0,
                "hough_2": 0.9,
                "contour": 0.8,
                "motion": 0.7,
                "trajectory_prediction": 0.8,  # Phase 2: Good weight for physics prediction
                "trajectory_inference": 0.9,   # Phase 2b: High weight for direction-change touches
                "prediction": 0.5
            }
            
            weighted_confidence = 0
            total_weight = 0
            
            for touch in smoothed_touch_events:
                weight = method_weights.get(touch.detection_method, 0.6)
                weighted_confidence += touch.confidence * weight
                total_weight += weight
            
            avg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0
        else:
            avg_confidence = 0.0
        
        # Calculate touch count range based on confidence and video quality
        touch_range = self._calculate_touch_range(total_touches, avg_confidence, quality_assessment)
        
        results = {
            "total_ball_touches": total_touches,
            "touch_range": touch_range,
            "video_duration": metadata["duration"],
            "touches_per_minute": round(touches_per_minute, 1),
            "confidence_score": round(avg_confidence, 2),
            "processing_time": round(processing_time, 1),
            "touch_events": [t.to_dict() for t in smoothed_touch_events],
            "debug_frames": debug_frames[:5],
            "metadata": metadata,
            "video_orientation": {
                "detected": video_orientation or "normal",
                "corrected": video_orientation != "normal" if video_orientation else False
            },
            "quality_assessment": {
                "overall_score": round(quality_assessment.overall_score, 2) if quality_assessment else 0.8,
                "needs_review": quality_assessment.needs_review if quality_assessment else False,
                "issues": quality_assessment.issues if quality_assessment else [],
                "brightness": round(quality_assessment.brightness, 1) if quality_assessment else 128,
                "contrast": round(quality_assessment.contrast, 1) if quality_assessment else 50
            },
            "detection_summary": self._get_detection_summary()
        }
        
        model_info = "YOLO v2 + Traditional" if self.yolo_model else "Traditional Only"
        logger.info(f"Analysis complete ({model_info}): {total_touches} touches detected with {avg_confidence:.2f} confidence")
        return results
