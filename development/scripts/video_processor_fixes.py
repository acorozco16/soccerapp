# Video Processor Fixes for NoneType Errors
# Apply these changes to fix the orientation detection and close method issues

# FIX 1: Orientation Detection Function (around line 682)
# Replace the current _detect_video_orientation function with this safer version:

def _detect_video_orientation(self, frame: np.ndarray) -> str:
    """Detect if video is upside down or rotated using pose detection"""
    try:
        # Check if pose detector is available
        if not hasattr(self, 'pose') or self.pose is None:
            logger.warning("Pose detector not available for orientation detection")
            return "normal"
            
        # Run pose detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(rgb_frame)
        
        # Check if pose results and landmarks exist
        if pose_results and pose_results.pose_landmarks and pose_results.pose_landmarks.landmark:
            landmarks = pose_results.pose_landmarks.landmark
            
            # Verify we have the required landmarks
            required_landmarks = [
                self.mp_pose.PoseLandmark.NOSE,
                self.mp_pose.PoseLandmark.LEFT_ANKLE,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER
            ]
            
            # Check if all required landmarks are visible
            for landmark_idx in required_landmarks:
                if landmark_idx.value >= len(landmarks) or landmarks[landmark_idx.value].visibility < 0.5:
                    logger.debug(f"Required landmark {landmark_idx.name} not visible enough")
                    return "normal"
            
            # Get key landmarks (normalized coordinates 0-1)
            nose_y = landmarks[self.mp_pose.PoseLandmark.NOSE.value].y
            left_ankle_y = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y
            right_ankle_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
            
            # Average foot position
            avg_foot_y = (left_ankle_y + right_ankle_y) / 2
            
            # Check if head is below feet (upside down)
            if nose_y > avg_foot_y + 0.1:  # 0.1 threshold for confidence
                return "upside_down"
            
            # Check for sideways rotation using shoulders
            left_shoulder_x = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
            right_shoulder_x = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
            left_shoulder_y = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            right_shoulder_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            
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

# FIX 2: Safe Close Method in Finally Block
# Find the finally block (around line with "cap.release()" and "self.pose.close()") 
# Replace the self.pose.close() line with:

        finally:
            cap.release()
            # Safe close for pose detector
            if hasattr(self, 'pose') and self.pose is not None:
                try:
                    self.pose.close()
                except Exception as e:
                    logger.warning(f"Error closing pose detector: {e}")

# FIX 3: Safe Pose Initialization Check
# Add this method to VideoProcessor class to ensure pose is properly initialized:

def _ensure_pose_initialized(self):
    """Ensure pose detector is properly initialized"""
    if not hasattr(self, 'pose') or self.pose is None:
        try:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("Pose detector re-initialized")
        except Exception as e:
            logger.error(f"Failed to initialize pose detector: {e}")
            self.pose = None

# Call this method at the start of analyze_video before processing frames:
# self._ensure_pose_initialized()