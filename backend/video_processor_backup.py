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
        
        # Enhanced ball detection parameters with adaptive thresholds
        self.ball_color_ranges = {
            # Clean orange ball
            "orange_clean": {
                "lower": np.array([5, 100, 100]),
                "upper": np.array([15, 255, 255])
            },
            # Muddy/dirty orange ball
            "orange_muddy": {
                "lower": np.array([8, 50, 80]),
                "upper": np.array([25, 200, 180])
            },
            # White ball
            "white_clean": {
                "lower": np.array([0, 0, 200]),
                "upper": np.array([180, 30, 255])
            },
            # Dirty white ball
            "white_dirty": {
                "lower": np.array([0, 0, 150]),
                "upper": np.array([180, 50, 230])
            },
            # Wet/reflective ball
            "reflective": {
                "lower": np.array([0, 0, 180]),
                "upper": np.array([180, 80, 255])
            }
        }
        
        # Multi-parameter sets for Hough circles
        self.hough_params = [
            # Standard detection
            {"dp": 1, "minDist": 50, "param1": 50, "param2": 30, "minRadius": 8, "maxRadius": 60},
            # Sensitive detection for faint circles
            {"dp": 1, "minDist": 40, "param1": 30, "param2": 20, "minRadius": 10, "maxRadius": 50},
            # Robust detection for clear circles
            {"dp": 2, "minDist": 60, "param1": 70, "param2": 40, "minRadius": 12, "maxRadius": 45},
        ]
        
        # Touch detection parameters
        self.touch_threshold_pixels = 50
        self.debounce_time = 0.5
        self.min_ball_velocity = 3  # Reduced for better sensitivity
        
        # Processing parameters
        self.frame_skip = 3
        self.target_width = 1280
        self.max_fps = 30
        
        # Tracking state
        self.ball_history = deque(maxlen=15)  # Store last 15 detections
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.last_known_position = None
        self.frames_without_detection = 0
        
        # Quality assessment thresholds
        self.quality_thresholds = {
            "min_brightness": 40,
            "max_brightness": 220,
            "min_contrast": 30,
            "max_blur_score": 100,
            "max_shake_score": 50
        }
        
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
                
                # Detect pose and foot positions first
                pose_results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                foot_positions = self._get_foot_positions(pose_results, frame.shape)
                
                # Enhanced ball detection
                ball_detection = self._detect_ball(frame, frame_count, foot_positions)
                
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
                        
                        # Save debug frame for this touch
                        debug_frame = self._create_debug_frame(
                            frame.copy(), ball_detection.position, foot_positions, pose_results, ball_detection
                        )
                        debug_frame_path = frames_dir / f"touch_{len(touch_events)}.jpg"
                        cv2.imwrite(str(debug_frame_path), debug_frame)
                        debug_frames.append(f"touch_{len(touch_events)}.jpg")
                
                # Update ball position
                if ball_detection:
                    last_ball_pos = ball_detection.position
                
                # Save first and last frame for reference
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
        
        # Apply trajectory smoothing to remove noise
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
                "hough_1": 1.0,
                "hough_2": 0.9,
                "contour": 0.8,
                "motion": 0.7,
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
        
        results = {
            "total_ball_touches": total_touches,
            "video_duration": metadata["duration"],
            "touches_per_minute": round(touches_per_minute, 1),
            "confidence_score": round(avg_confidence, 2),
            "processing_time": round(processing_time, 1),
            "touch_events": [t.to_dict() for t in smoothed_touch_events],
            "debug_frames": debug_frames[:5],  # Limit to 5 frames
            "metadata": metadata,
            "quality_assessment": {
                "overall_score": round(quality_assessment.overall_score, 2) if quality_assessment else 0.8,
                "needs_review": quality_assessment.needs_review if quality_assessment else False,
                "issues": quality_assessment.issues if quality_assessment else [],
                "brightness": round(quality_assessment.brightness, 1) if quality_assessment else 128,
                "contrast": round(quality_assessment.contrast, 1) if quality_assessment else 50
            },
            "detection_summary": self._get_detection_summary()
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
    
    def _assess_video_quality(self, frame: np.ndarray) -> VideoQuality:
        """Assess video quality and identify issues"""
        issues = []
        
        # Convert to grayscale for analysis
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
        
        # Blur assessment using Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < self.quality_thresholds["max_blur_score"]:
            issues.append("Video appears blurry or out of focus")
        
        # Shake detection (simplified - would need frame comparison in practice)
        shake_score = np.std(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3))
        
        # Overall quality score (0-1)
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

    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply automatic exposure/contrast adjustment"""
        # Convert to LAB color space for better luminance control
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced

    def _detect_ball_primary(self, frame: np.ndarray) -> List[BallDetection]:
        """Primary detection methods: HSV + Hough + Contours"""
        detections = []
        
        # Enhance frame if needed
        enhanced_frame = self._enhance_frame(frame)
        hsv = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2HSV)
        
        # Method 1: HSV Color Filtering with multiple ranges
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        best_color_match = None
        best_color_score = 0
        
        for color_name, ranges in self.ball_color_ranges.items():
            color_mask = cv2.inRange(hsv, ranges["lower"], ranges["upper"])
            mask_score = np.sum(color_mask) / 255
            
            if mask_score > best_color_score:
                best_color_score = mask_score
                best_color_match = color_name
            
            combined_mask = cv2.bitwise_or(combined_mask, color_mask)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Method 2: Hough Circle Detection with multiple parameter sets
        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(combined_mask, (9, 9), 2)
        
        for params in self.hough_params:
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                **params
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :3]:  # Take up to 3 best circles
                    x, y, r = circle
                    
                    # Calculate confidence based on mask overlap
                    circle_mask = np.zeros_like(combined_mask)
                    cv2.circle(circle_mask, (x, y), r, 255, -1)
                    overlap = cv2.bitwise_and(combined_mask, circle_mask)
                    confidence = np.sum(overlap) / (np.pi * r * r * 255) if r > 0 else 0
                    
                    if confidence > 0.3:  # Minimum confidence threshold
                        detections.append(BallDetection(
                            position=(int(x), int(y)),
                            confidence=float(confidence),
                            method=f"hough_{params['dp']}",
                            radius=int(r)
                        ))
        
        # Method 3: Contour Detection for irregular shapes
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50 or area > 5000:  # Size filtering
                continue
                
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > 0.4:  # Reasonably circular
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    confidence = circularity * 0.8  # Scale down contour confidence
                    detections.append(BallDetection(
                        position=(cx, cy),
                        confidence=float(confidence),
                        method="contour",
                        area=area
                    ))
        
        return detections

    def _detect_ball_motion(self, frame: np.ndarray) -> List[BallDetection]:
        """Motion-based detection for fast-moving scenarios"""
        detections = []
        
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Find motion contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 30 < area < 2000:  # Reasonable ball size in motion
                # Check if contour is roughly circular
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.3:  # Less strict for motion detection
                        M = cv2.moments(contour)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            confidence = circularity * 0.6  # Lower confidence for motion
                            detections.append(BallDetection(
                                position=(cx, cy),
                                confidence=float(confidence),
                                method="motion",
                                area=area
                            ))
        
        return detections

    def _predict_ball_position(self, frame_number: int) -> Optional[BallDetection]:
        """Predict ball position using physics when detection fails"""
        if len(self.ball_history) < 3:
            return None
        
        # Get recent positions
        recent_positions = list(self.ball_history)[-3:]
        
        # Simple linear extrapolation based on velocity
        if len(recent_positions) >= 2:
            last_pos = recent_positions[-1].position
            prev_pos = recent_positions[-2].position
            
            # Calculate velocity
            dx = last_pos[0] - prev_pos[0]
            dy = last_pos[1] - prev_pos[1]
            
            # Apply simple physics (gravity effect)
            frames_elapsed = frame_number - getattr(recent_positions[-1], 'frame_number', 0)
            predicted_x = last_pos[0] + dx * frames_elapsed
            predicted_y = last_pos[1] + dy * frames_elapsed + 0.5 * frames_elapsed  # Gravity
            
            # Only predict if reasonable
            if 0 <= predicted_x <= 1280 and 0 <= predicted_y <= 720:
                confidence = max(0.1, 0.5 - frames_elapsed * 0.1)  # Decreasing confidence
                return BallDetection(
                    position=(int(predicted_x), int(predicted_y)),
                    confidence=confidence,
                    method="prediction"
                )
        
        return None

    def _validate_detection(self, detection: BallDetection, frame_number: int) -> bool:
        """Physics-based validation of detections"""
        if not self.ball_history:
            return True
        
        last_detection = self.ball_history[-1]
        
        # Calculate distance moved
        distance = math.sqrt(
            (detection.position[0] - last_detection.position[0]) ** 2 +
            (detection.position[1] - last_detection.position[1]) ** 2
        )
        
        # Maximum reasonable speed (pixels per frame)
        max_speed = 50  # Adjust based on typical ball speeds
        
        if distance > max_speed:
            logger.warning(f"Detection rejected: ball moved {distance:.1f} pixels (max: {max_speed})")
            return False
        
        return True

    def _choose_best_detection(self, detections: List[BallDetection], foot_positions: List[Tuple[int, int]]) -> Optional[BallDetection]:
        """Choose best detection from multiple candidates"""
        if not detections:
            return None
        
        if len(detections) == 1:
            return detections[0]
        
        # Score detections based on multiple factors
        best_detection = None
        best_score = -1
        
        for detection in detections:
            score = detection.confidence
            
            # Bonus for consistency with history
            if self.ball_history:
                last_pos = self.ball_history[-1].position
                distance = math.sqrt(
                    (detection.position[0] - last_pos[0]) ** 2 +
                    (detection.position[1] - last_pos[1]) ** 2
                )
                consistency_bonus = max(0, 1 - distance / 100)  # Closer = better
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

    def _detect_ball(self, frame: np.ndarray, frame_number: int, foot_positions: List[Tuple[int, int]]) -> Optional[BallDetection]:
        """Enhanced multi-layered ball detection"""
        # Primary detection methods
        primary_detections = self._detect_ball_primary(frame)
        
        # Motion-based detection
        motion_detections = self._detect_ball_motion(frame)
        
        # Combine all detections
        all_detections = primary_detections + motion_detections
        
        # Filter valid detections
        valid_detections = [d for d in all_detections if self._validate_detection(d, frame_number)]
        
        # Choose best detection
        best_detection = self._choose_best_detection(valid_detections, foot_positions)
        
        # If no detection found, try prediction
        if best_detection is None:
            self.frames_without_detection += 1
            if self.frames_without_detection <= 5:  # Allow up to 5 frames without detection
                predicted = self._predict_ball_position(frame_number)
                if predicted:
                    best_detection = predicted
        else:
            self.frames_without_detection = 0
        
        # Update history
        if best_detection:
            self.ball_history.append(best_detection)
            self.last_known_position = best_detection.position
        
        return best_detection
    
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
    
    def _smooth_trajectory(self, touch_events: List[TouchEvent]) -> List[TouchEvent]:
        """Apply trajectory smoothing to remove detection noise"""
        if len(touch_events) <= 2:
            return touch_events
        
        smoothed_events = []
        window_size = 3
        
        for i, event in enumerate(touch_events):
            # Get neighboring events for smoothing
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(touch_events), i + window_size // 2 + 1)
            neighbors = touch_events[start_idx:end_idx]
            
            # Check for outliers using distance from neighbors
            distances = []
            for neighbor in neighbors:
                if neighbor != event:
                    distance = math.sqrt(
                        (event.position[0] - neighbor.position[0]) ** 2 +
                        (event.position[1] - neighbor.position[1]) ** 2
                    )
                    distances.append(distance)
            
            # Remove if it's too far from all neighbors (outlier)
            if distances and min(distances) > 100:  # 100 pixel threshold
                logger.info(f"Removing outlier touch at {event.timestamp:.1f}s")
                continue
            
            # Apply temporal smoothing for very close touches
            if i > 0:
                time_diff = event.timestamp - touch_events[i-1].timestamp
                if time_diff < 0.3:  # Very close in time
                    # Average with previous position
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
            "success_rate": len(self.ball_history) / max(1, len(self.ball_history) + self.frames_without_detection)
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
            # Choose color based on detection method
            method_colors = {
                "hough_1": (0, 0, 255),    # Red
                "hough_2": (0, 100, 255),  # Orange-red
                "contour": (0, 255, 255),  # Yellow
                "motion": (255, 0, 255),   # Magenta
                "prediction": (128, 128, 128)  # Gray
            }
            
            color = method_colors.get(
                ball_detection.method if ball_detection else "unknown", 
                (0, 0, 255)
            )
            
            # Draw circle with method-specific color
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
        
        # Add enhanced header with detection info
        cv2.putText(frame, f"Enhanced Soccer Analysis", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add legend for detection methods
        legend_y = 60
        for method, color in {
            "Hough": (0, 0, 255),
            "Contour": (0, 255, 255), 
            "Motion": (255, 0, 255),
            "Predict": (128, 128, 128)
        }.items():
            cv2.putText(frame, method, (10, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            legend_y += 20
        
        return frame