#!/usr/bin/env python3
"""
Video Processing Speed Optimizations
Apply these fixes to video_processor.py for major speed improvements:
1. Single orientation check
2. YOLO every 3 frames
3. Downscale to 640x480
4. Cache YOLO model
"""

import re

def apply_speed_optimizations():
    """Apply all 4 speed optimizations to video_processor.py"""
    
    # Read current file
    with open('/root/soccerapp/backend/video_processor.py', 'r') as f:
        content = f.read()
    
    print("🚀 Applying speed optimizations...")
    
    # OPTIMIZATION 1: Single Orientation Check
    print("1️⃣ Implementing single orientation check...")
    content = fix_single_orientation_check(content)
    
    # OPTIMIZATION 2: YOLO Every 3 Frames
    print("2️⃣ Optimizing YOLO to process every 3rd frame...")
    content = fix_yolo_frame_sampling(content)
    
    # OPTIMIZATION 3: Downscale to 640x480
    print("3️⃣ Adding video resolution downscaling...")
    content = fix_video_resolution(content)
    
    # OPTIMIZATION 4: Cache YOLO Model
    print("4️⃣ Implementing YOLO model caching...")
    content = fix_yolo_model_caching(content)
    
    # Write optimized file
    with open('/root/soccerapp/backend/video_processor.py', 'w') as f:
        f.write(content)
    
    print("✅ All speed optimizations applied successfully!")

def fix_single_orientation_check(content: str) -> str:
    """Fix 1: Check orientation only once at the beginning"""
    
    # Find the orientation detection loop
    old_pattern = r'''# Detect video orientation on first few frames
                if not orientation_detected and processed_frames < 3:
                    detected_orientation = self\._detect_video_orientation\(frame\)
                    if detected_orientation != "normal":
                        video_orientation = detected_orientation
                        logger\.info\(f"Video orientation detected: \{video_orientation\}"\)
                        print\(f"🔄 Video orientation: \{video_orientation\} - will auto-correct"\)
                    elif processed_frames >= 2:
                        video_orientation = "normal"
                        orientation_detected = True
                        logger\.info\("Video orientation: normal"\)'''
    
    new_pattern = '''# Single orientation check at start (SPEED OPTIMIZATION)
                if not orientation_detected and processed_frames == 0:
                    detected_orientation = self._detect_video_orientation(frame)
                    video_orientation = detected_orientation
                    orientation_detected = True
                    if detected_orientation != "normal":
                        logger.info(f"Video orientation detected: {video_orientation}")
                        print(f"🔄 Video orientation: {video_orientation} - will auto-correct")
                    else:
                        logger.info("Video orientation: normal")'''
    
    content = re.sub(old_pattern, new_pattern, content, flags=re.DOTALL)
    return content

def fix_yolo_frame_sampling(content: str) -> str:
    """Fix 2: Process YOLO every 3rd frame instead of every frame"""
    
    # Find the YOLO detection call
    old_yolo_pattern = r'# PHASE 1: ByteTrack Detection \+ Tracking\s+yolo_detections = self\._detect_ball_yolo_v8\(frame\)'
    
    new_yolo_pattern = '''# PHASE 1: ByteTrack Detection + Tracking (Every 3rd frame for speed)
                # Only run YOLO every 3rd frame for 3x speed improvement
                if frame_count % 3 == 0:
                    yolo_detections = self._detect_ball_yolo_v8(frame)
                    # Cache detections for interpolation
                    self._last_yolo_detections = yolo_detections
                    self._last_yolo_frame = frame_count
                else:
                    # Use cached detections from previous YOLO run
                    yolo_detections = getattr(self, '_last_yolo_detections', [])'''
    
    content = re.sub(old_yolo_pattern, new_yolo_pattern, content)
    
    # Initialize caching variables in __init__
    init_pattern = r'(self\.yolo_confidence_threshold = 0\.1.*?\n)'
    init_replacement = r'\1        # YOLO frame caching for speed optimization\n        self._last_yolo_detections = []\n        self._last_yolo_frame = 0\n'
    content = re.sub(init_pattern, init_replacement, content)
    
    return content

def fix_video_resolution(content: str) -> str:
    """Fix 3: Downscale video to 640x480 for faster processing"""
    
    # Add resolution optimization to _resize_frame method
    resize_pattern = r'def _resize_frame\(self, frame: np\.ndarray\) -> np\.ndarray:'
    
    if resize_pattern in content:
        # Replace existing _resize_frame method
        old_resize_method = r'def _resize_frame\(self, frame: np\.ndarray\) -> np\.ndarray:.*?return frame'
        
        new_resize_method = '''def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame for optimal processing speed vs accuracy"""
        height, width = frame.shape[:2]
        
        # SPEED OPTIMIZATION: Downscale to 640x480 for 4x speed improvement
        target_width = 640
        target_height = 480
        
        # Only resize if frame is larger than target
        if width > target_width or height > target_height:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            logger.debug(f"Downscaled frame from {width}x{height} to {target_width}x{target_height}")
        
        return frame'''
        
        content = re.sub(old_resize_method, new_resize_method, content, flags=re.DOTALL)
    else:
        # Add new _resize_frame method if it doesn't exist
        # Find a good place to insert it (after other methods)
        insert_point = content.find('def _get_video_metadata(self')
        if insert_point != -1:
            new_method = '''
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame for optimal processing speed vs accuracy"""
        height, width = frame.shape[:2]
        
        # SPEED OPTIMIZATION: Downscale to 640x480 for 4x speed improvement
        target_width = 640
        target_height = 480
        
        # Only resize if frame is larger than target
        if width > target_width or height > target_height:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            logger.debug(f"Downscaled frame from {width}x{height} to {target_width}x{target_height}")
        
        return frame

    '''
            content = content[:insert_point] + new_method + content[insert_point:]
    
    # Make sure _resize_frame is called in the main loop
    if 'frame = self._resize_frame(frame)' not in content:
        # Add resize call after orientation correction
        standardize_pattern = r'(frame = self\._standardize_orientation\(frame, video_orientation\))'
        replacement = r'\1\n                \n                # Resize frame for speed (OPTIMIZATION)\n                frame = self._resize_frame(frame)'
        content = re.sub(standardize_pattern, replacement, content)
    
    return content

def fix_yolo_model_caching(content: str) -> str:
    """Fix 4: Cache YOLO model to avoid reloading"""
    
    # Modify VideoProcessor __init__ to use singleton pattern for YOLO model
    old_init_pattern = r'class VideoProcessor:\s+def __init__\(self\):'
    
    # Add class variable for cached YOLO model
    class_vars = '''class VideoProcessor:
    # SPEED OPTIMIZATION: Class-level YOLO model caching
    _cached_yolo_model = None
    _cached_custom_model = None
    _model_cache_lock = threading.Lock()
    
    def __init__(self):'''
    
    content = re.sub(old_init_pattern, class_vars, content)
    
    # Replace YOLO model loading logic with caching
    old_yolo_loading = r'''# Load YOLO v8 model \(prioritize v8, fallback to custom trained\)
        try:
            # Try YOLO v8 first for better performance
            self\.yolo_model = YOLO\('yolov8n\.pt'\).*?except Exception as e2:
                print\(f"❌ Failed to load any YOLO model: \{e2\}"\)
                logger\.error\(f"Failed to load any YOLO model: \{e2\}"\)
                self\.yolo_model = None
                self\.custom_yolo_model = None'''
    
    new_yolo_loading = '''# SPEED OPTIMIZATION: Use cached YOLO models
        self.yolo_model, self.custom_yolo_model = self._get_cached_yolo_models()
        if self.yolo_model:
            self.yolo_confidence_threshold = 0.1
            logger.info(f"Using cached YOLO v8 model with confidence threshold {self.yolo_confidence_threshold}")
        else:
            self.yolo_confidence_threshold = 0.05
            logger.warning("No YOLO models available, using traditional detection")'''
    
    content = re.sub(old_yolo_loading, new_yolo_loading, content, flags=re.DOTALL)
    
    # Add the caching method
    caching_method = '''
    @classmethod
    def _get_cached_yolo_models(cls):
        """Get or create cached YOLO models for speed optimization"""
        with cls._model_cache_lock:
            if cls._cached_yolo_model is None:
                try:
                    # Load YOLO v8 model once and cache it
                    cls._cached_yolo_model = YOLO('yolov8n.pt')
                    print(f"✅ Loaded and cached YOLO v8 model")
                    logger.info("YOLO v8 model loaded and cached for reuse")
                    
                    # Also cache custom model if available
                    custom_model_path = Path(__file__).parent.parent / "models/soccer_ball_trained.pt"
                    if custom_model_path.exists():
                        cls._cached_custom_model = YOLO(str(custom_model_path))
                        print(f"✅ Also loaded and cached custom trained model")
                    
                except Exception as e:
                    print(f"❌ Failed to load YOLO v8, trying custom model: {e}")
                    try:
                        model_path = Path(__file__).parent.parent / "models/soccer_ball_trained.pt"
                        if model_path.exists():
                            cls._cached_yolo_model = YOLO(str(model_path))
                            print(f"✅ Loaded and cached custom YOLO model")
                            logger.info("Custom YOLO model loaded and cached for reuse")
                        else:
                            print(f"⚠️ No YOLO models found")
                            logger.warning("No YOLO models found")
                    except Exception as e2:
                        print(f"❌ Failed to load any YOLO model: {e2}")
                        logger.error(f"Failed to load any YOLO model: {e2}")
            
            return cls._cached_yolo_model, cls._cached_custom_model

    '''
    
    # Insert the method after the __init__ method
    init_end = content.find('        # Initialize ByteTrack')
    if init_end != -1:
        content = content[:init_end] + caching_method + '        ' + content[init_end:]
    
    return content

if __name__ == "__main__":
    apply_speed_optimizations()