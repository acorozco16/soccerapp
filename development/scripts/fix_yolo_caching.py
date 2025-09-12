#!/usr/bin/env python3
"""
Quick fix for YOLO model caching issue
"""

def fix_yolo_caching():
    """Fix the missing _get_cached_yolo_models method"""
    
    # Read current file
    with open('/root/soccerapp/backend/video_processor.py', 'r') as f:
        content = f.read()
    
    # Check if method already exists
    if '_get_cached_yolo_models' in content:
        print("Method already exists, checking implementation...")
        return
    
    # Find a good insertion point (after __init__ method)
    insertion_point = content.find('        # Initialize ByteTrack')
    
    if insertion_point == -1:
        print("Could not find insertion point, trying alternative...")
        insertion_point = content.find('def _get_video_metadata(self')
    
    if insertion_point != -1:
        # Insert the missing method
        method_code = '''
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
        
        # Insert before the found point
        content = content[:insertion_point] + method_code + '\n        ' + content[insertion_point:]
        
        # Write back
        with open('/root/soccerapp/backend/video_processor.py', 'w') as f:
            f.write(content)
        
        print("✅ Added missing _get_cached_yolo_models method")
    else:
        print("❌ Could not find suitable insertion point")

if __name__ == "__main__":
    fix_yolo_caching()