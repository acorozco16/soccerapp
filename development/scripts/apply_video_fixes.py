#!/usr/bin/env python3
"""
Script to apply video processor fixes for NoneType errors
Run this on the DigitalOcean server
"""

import re

def apply_fixes():
    # Read the current video_processor.py
    with open('video_processor.py', 'r') as f:
        content = f.read()
    
    # FIX 1: Replace the orientation detection function
    old_orientation_func = r'def _detect_video_orientation\(self, frame: np\.ndarray\) -> str:.*?return "normal"'
    
    new_orientation_func = '''def _detect_video_orientation(self, frame: np.ndarray) -> str:
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
                
                # Verify we have enough landmarks
                if len(landmarks) < 33:  # MediaPipe has 33 pose landmarks
                    logger.debug("Insufficient pose landmarks detected")
                    return "normal"
                
                try:
                    # Get key landmarks (normalized coordinates 0-1)
                    nose_y = landmarks[0].y  # NOSE is index 0
                    left_ankle_y = landmarks[27].y  # LEFT_ANKLE is index 27
                    right_ankle_y = landmarks[28].y  # RIGHT_ANKLE is index 28
                    
                    # Average foot position
                    avg_foot_y = (left_ankle_y + right_ankle_y) / 2
                    
                    # Check if head is below feet (upside down)
                    if nose_y > avg_foot_y + 0.1:  # 0.1 threshold for confidence
                        return "upside_down"
                    
                    # Check for sideways rotation using shoulders
                    left_shoulder_x = landmarks[11].x  # LEFT_SHOULDER is index 11
                    right_shoulder_x = landmarks[12].x  # RIGHT_SHOULDER is index 12
                    left_shoulder_y = landmarks[11].y
                    right_shoulder_y = landmarks[12].y
                    
                    # Check if shoulders are more vertical than horizontal (sideways)
                    shoulder_height_diff = abs(left_shoulder_y - right_shoulder_y)
                    shoulder_width_diff = abs(left_shoulder_x - right_shoulder_x)
                    
                    if shoulder_height_diff > shoulder_width_diff * 1.5:
                        # Determine which way to rotate
                        if left_shoulder_y < right_shoulder_y:
                            return "rotated_left"
                        else:
                            return "rotated_right"
                            
                except (IndexError, AttributeError) as e:
                    logger.debug(f"Landmark access error: {e}")
                    return "normal"
            
            return "normal"
            
        except Exception as e:
            logger.warning(f"Orientation detection failed: {e}")
            return "normal"'''
    
    # Apply fix 1
    content = re.sub(old_orientation_func, new_orientation_func, content, flags=re.DOTALL)
    
    # FIX 2: Replace the unsafe close in finally block
    old_finally = r'finally:\s+cap\.release\(\)\s+self\.pose\.close\(\)'
    
    new_finally = '''finally:
            cap.release()
            # Safe close for pose detector
            if hasattr(self, 'pose') and self.pose is not None:
                try:
                    self.pose.close()
                except Exception as e:
                    logger.warning(f"Error closing pose detector: {e}")'''
    
    # Apply fix 2
    content = re.sub(old_finally, new_finally, content)
    
    # FIX 3: Add safe pose initialization check at the start of analyze_video
    # Find the analyze_video method and add the check
    analyze_video_start = content.find('async def analyze_video(')
    if analyze_video_start != -1:
        # Find the first line after the method signature
        next_line = content.find('\n', analyze_video_start)
        if next_line != -1:
            # Find the start of the method body (after docstring)
            method_body_start = content.find('start_time = datetime.now()', next_line)
            if method_body_start != -1:
                # Insert the safety check before start_time
                safety_check = '''        # Ensure pose detector is initialized
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
                raise RuntimeError(f"Cannot initialize pose detector: {e}")
        
        '''
                content = content[:method_body_start] + safety_check + content[method_body_start:]
    
    # Write the fixed content back
    with open('video_processor.py', 'w') as f:
        f.write(content)
    
    print("✅ Applied all video processor fixes")
    print("Fixed issues:")
    print("  1. Orientation detection NoneType error")
    print("  2. Pose close() method error")
    print("  3. Added pose detector safety checks")

if __name__ == "__main__":
    apply_fixes()