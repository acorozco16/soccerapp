#!/usr/bin/env python3
"""
CRITICAL: MediaPipe Pose Detector Stability Fixes
This implements multiple strategies to fix the intermittent MediaPipe failures
Apply these changes to video_processor.py on DigitalOcean server
"""

import threading
import time
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class PoseDetectorPool:
    """Singleton pool for MediaPipe pose detectors to prevent initialization conflicts"""
    
    _instance = None
    _lock = threading.Lock()
    _pose_detector = None
    _mp_pose = None
    _initialization_lock = threading.Lock()
    _last_used = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._initialize_mediapipe()
    
    def _initialize_mediapipe(self):
        """Safe MediaPipe initialization with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                import mediapipe as mp
                self._mp_pose = mp.solutions.pose
                
                # Create pose detector with optimal settings for stability
                self._pose_detector = self._mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,  # Balance between accuracy and performance
                    enable_segmentation=False,  # Disable to reduce memory usage
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self._last_used = time.time()
                logger.info(f"MediaPipe pose detector initialized successfully (attempt {attempt + 1})")
                return
                
            except Exception as e:
                logger.error(f"MediaPipe initialization attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to initialize MediaPipe after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    @contextmanager
    def get_pose_detector(self):
        """Thread-safe context manager for pose detector access"""
        with self._initialization_lock:
            try:
                # Check if detector needs reinitialization (idle for >5 minutes)
                if (self._pose_detector is None or 
                    (self._last_used and time.time() - self._last_used > 300)):
                    logger.info("Reinitializing pose detector due to timeout or None state")
                    self._cleanup_detector()
                    self._initialize_mediapipe()
                
                self._last_used = time.time()
                yield self._pose_detector, self._mp_pose
                
            except Exception as e:
                logger.error(f"Error in pose detector context: {e}")
                # Try to reinitialize on error
                try:
                    self._cleanup_detector()
                    self._initialize_mediapipe()
                    yield self._pose_detector, self._mp_pose
                except Exception as retry_error:
                    logger.error(f"Pose detector recovery failed: {retry_error}")
                    yield None, None
    
    def _cleanup_detector(self):
        """Safe cleanup of pose detector"""
        if self._pose_detector is not None:
            try:
                self._pose_detector.close()
            except Exception as e:
                logger.warning(f"Error closing pose detector: {e}")
            finally:
                self._pose_detector = None

# Global pose detector pool instance
pose_pool = PoseDetectorPool()

class StableVideoProcessor:
    """Enhanced VideoProcessor with MediaPipe stability fixes"""
    
    def __init__(self):
        self.pose_pool = pose_pool
    
    def _detect_video_orientation_stable(self, frame: np.ndarray) -> str:
        """Ultra-stable orientation detection with multiple fallback strategies"""
        
        # Strategy 1: Try MediaPipe pose detection
        try:
            with self.pose_pool.get_pose_detector() as (pose_detector, mp_pose):
                if pose_detector is None or mp_pose is None:
                    logger.warning("Pose detector unavailable, using fallback orientation detection")
                    return self._detect_orientation_fallback(frame)
                
                # Convert frame safely
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    logger.warning(f"Frame conversion error: {e}")
                    return "normal"
                
                # Process with timeout simulation (MediaPipe doesn't support real timeouts)
                try:
                    pose_results = pose_detector.process(rgb_frame)
                except Exception as e:
                    logger.warning(f"MediaPipe processing error: {e}")
                    return self._detect_orientation_fallback(frame)
                
                # Check results safely
                if (pose_results and 
                    hasattr(pose_results, 'pose_landmarks') and 
                    pose_results.pose_landmarks and 
                    hasattr(pose_results.pose_landmarks, 'landmark') and
                    pose_results.pose_landmarks.landmark):
                    
                    landmarks = pose_results.pose_landmarks.landmark
                    
                    # Verify we have enough landmarks
                    if len(landmarks) < 33:
                        logger.debug("Insufficient pose landmarks, using fallback")
                        return self._detect_orientation_fallback(frame)
                    
                    try:
                        # Safe landmark access with bounds checking
                        nose_y = landmarks[0].y if len(landmarks) > 0 else None
                        left_ankle_y = landmarks[27].y if len(landmarks) > 27 else None
                        right_ankle_y = landmarks[28].y if len(landmarks) > 28 else None
                        
                        if None in [nose_y, left_ankle_y, right_ankle_y]:
                            return "normal"
                        
                        # Check visibility
                        nose_vis = landmarks[0].visibility if hasattr(landmarks[0], 'visibility') else 1.0
                        left_ankle_vis = landmarks[27].visibility if hasattr(landmarks[27], 'visibility') else 1.0
                        right_ankle_vis = landmarks[28].visibility if hasattr(landmarks[28], 'visibility') else 1.0
                        
                        if min(nose_vis, left_ankle_vis, right_ankle_vis) < 0.5:
                            return "normal"
                        
                        # Calculate orientation
                        avg_foot_y = (left_ankle_y + right_ankle_y) / 2
                        
                        if nose_y > avg_foot_y + 0.15:  # Increased threshold for confidence
                            return "upside_down"
                        
                        # Check for sideways rotation
                        left_shoulder_y = landmarks[11].y if len(landmarks) > 11 else None
                        right_shoulder_y = landmarks[12].y if len(landmarks) > 12 else None
                        left_shoulder_x = landmarks[11].x if len(landmarks) > 11 else None
                        right_shoulder_x = landmarks[12].x if len(landmarks) > 12 else None
                        
                        if None not in [left_shoulder_y, right_shoulder_y, left_shoulder_x, right_shoulder_x]:
                            shoulder_height_diff = abs(left_shoulder_y - right_shoulder_y)
                            shoulder_width_diff = abs(left_shoulder_x - right_shoulder_x)
                            
                            if shoulder_height_diff > shoulder_width_diff * 2.0:
                                return "rotated_left" if left_shoulder_y < right_shoulder_y else "rotated_right"
                        
                        return "normal"
                        
                    except (IndexError, AttributeError, TypeError) as e:
                        logger.debug(f"Landmark processing error: {e}")
                        return "normal"
                else:
                    logger.debug("No pose landmarks detected")
                    return self._detect_orientation_fallback(frame)
                    
        except Exception as e:
            logger.warning(f"MediaPipe orientation detection failed: {e}")
            return self._detect_orientation_fallback(frame)
    
    def _detect_orientation_fallback(self, frame: np.ndarray) -> str:
        """Pose-free orientation detection using image analysis"""
        try:
            height, width = frame.shape[:2]
            
            # Strategy 1: Detect if video is clearly rotated based on aspect ratio
            if height > width * 1.5:
                return "rotated_left"  # Portrait mode, likely needs rotation
            
            # Strategy 2: Use edge detection to find ground/floor
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Look for horizontal lines (likely ground/field lines)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                horizontal_lines = 0
                for rho, theta in lines[0]:
                    angle = theta * 180 / np.pi
                    if 80 <= angle <= 100 or 260 <= angle <= 280:  # Horizontal-ish
                        horizontal_lines += 1
                
                # If we find horizontal lines in the top portion, video might be upside down
                if horizontal_lines > 2:
                    # Additional check: brightness distribution
                    top_brightness = np.mean(gray[:height//3, :])
                    bottom_brightness = np.mean(gray[2*height//3:, :])
                    
                    if top_brightness > bottom_brightness * 1.2:  # Sky typically brighter
                        return "upside_down"
            
            return "normal"
            
        except Exception as e:
            logger.warning(f"Fallback orientation detection failed: {e}")
            return "normal"
    
    def safe_analyze_video(self, video_path: str, drill_type: str) -> Dict[str, Any]:
        """Main video analysis with comprehensive error handling"""
        try:
            # Pre-flight checks
            self._verify_environment()
            
            # Use the stable orientation detection
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            try:
                # Get first frame for orientation
                ret, frame = cap.read()
                if not ret:
                    raise ValueError("Cannot read first frame from video")
                
                orientation = self._detect_video_orientation_stable(frame)
                logger.info(f"Detected video orientation: {orientation}")
                
                # Continue with rest of video analysis...
                # (This would be the existing ball tracking logic)
                
                return {
                    "success": True,
                    "orientation": orientation,
                    "message": "Video processed successfully with stable pose detection"
                }
                
            finally:
                cap.release()
                
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Video analysis failed, check logs for details"
            }
    
    def _verify_environment(self):
        """Verify MediaPipe environment is ready"""
        try:
            import mediapipe as mp
            import cv2
            logger.info(f"MediaPipe version: {mp.__version__}")
            logger.info(f"OpenCV version: {cv2.__version__}")
        except ImportError as e:
            raise RuntimeError(f"Missing required dependency: {e}")

# Usage in video_processor.py:
# Replace the existing _detect_video_orientation method with:
# self._detect_video_orientation_stable(frame)