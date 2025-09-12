"""
Unified Video Processor using Drill Framework
Orchestrates video analysis with drill-specific analyzers
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

# Import video processor with error handling
try:
    from video_processor import VideoProcessor
    VIDEO_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  VideoProcessor not available: {e}")
    VIDEO_PROCESSOR_AVAILABLE = False
    VideoProcessor = None
from drill_analyzer import DrillType, DrillResults, drill_registry

logger = logging.getLogger(__name__)


class UnifiedVideoProcessor:
    """Unified processor that routes to drill-specific analyzers"""
    
    def __init__(self):
        # Use existing video processor for core detection
        if not VIDEO_PROCESSOR_AVAILABLE:
            raise RuntimeError("VideoProcessor not available - missing dependencies")
        self.base_processor = VideoProcessor()
        self.logger = logger
        
    def analyze_drill(self, video_path: str, drill_type: str, video_id: str = None) -> Dict:
        """Analyze video for specific drill type"""
        
        start_time = datetime.now()
        
        # Validate drill type
        try:
            drill_enum = DrillType(drill_type)
        except ValueError:
            raise ValueError(f"Unknown drill type: {drill_type}")
            
        # Get analyzer for drill
        analyzer = drill_registry.get_analyzer(drill_enum)
        if not analyzer:
            raise ValueError(f"No analyzer available for drill: {drill_type}")
            
        self.logger.info(f"Starting {drill_type} analysis for video: {video_path}")
        
        try:
            # Extract video features using existing processor
            video_data = self._extract_video_features(video_path, video_id)
            
            # Run drill-specific analysis
            results = analyzer.analyze(video_data)
            
            # Add processing time
            results.processing_time = (datetime.now() - start_time).total_seconds()
            results.timestamp = datetime.now().isoformat()
            
            self.logger.info(f"Analysis complete: {results.count_detected} reps detected")
            
            return results.to_dict()
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise
            
    def _extract_video_features(self, video_path: str, video_id: str) -> Dict:
        """Extract common video features needed by all drills"""
        
        # Use existing video processor for ball and pose detection
        # This runs the full detection pipeline
        raw_results = self.base_processor.analyze_video(video_path, video_id)
        
        # Transform results into format expected by drill analyzers
        video_data = {
            "video_id": video_id,
            "duration": raw_results["video_duration"],
            "frame_height": 720,  # Standard height
            "frame_width": 1280,  # Standard width
            "fps": 30,  # Assumed FPS
            
            # Ball detections with frame info
            "ball_detections": self._transform_ball_detections(raw_results),
            
            # Foot positions by frame
            "foot_positions": self._extract_foot_positions(raw_results),
            
            # Touch events (for drills that can reuse)
            "touch_events": raw_results.get("touch_events", []),
            
            # Raw results for backward compatibility
            "raw_results": raw_results
        }
        
        return video_data
    
    def _transform_ball_detections(self, raw_results: Dict) -> List[Dict]:
        """Transform ball detection data for drill analyzers"""
        detections = []
        
        # Extract from touch events (has ball positions)
        for event in raw_results.get("touch_events", []):
            detections.append({
                "timestamp": event["timestamp"],
                "frame_number": event["frame"],
                "position": event["position"],
                "confidence": event["confidence"],
                "method": event["detection_method"]
            })
            
        # Also extract from bell touches if available
        bell_data = raw_results.get("bell_touches", {})
        for event in bell_data.get("bell_touch_events", []):
            detections.append({
                "timestamp": event["timestamp"],
                "frame_number": event["frame"],
                "position": event["position"],
                "confidence": event["confidence"],
                "foot_used": event.get("foot_used")
            })
            
        # Sort by timestamp
        detections.sort(key=lambda x: x["timestamp"])
        
        return detections
    
    def _extract_foot_positions(self, raw_results: Dict) -> Dict[int, List[Tuple[int, int]]]:
        """Extract foot positions by frame number"""
        foot_positions = {}
        
        # This is simplified - in reality we'd extract from pose detection
        # For now, simulate based on touch events
        for event in raw_results.get("touch_events", []):
            frame = event["frame"]
            # Simulate foot positions around ball
            ball_x, ball_y = event["position"]
            foot_positions[frame] = [
                (ball_x - 100, ball_y),  # Left foot
                (ball_x + 100, ball_y)   # Right foot
            ]
            
        return foot_positions
    
    def list_available_drills(self) -> List[Dict]:
        """List all available drills with their configurations"""
        return drill_registry.list_drills()
    
    def get_drill_info(self, drill_type: str) -> Optional[Dict]:
        """Get detailed information about a specific drill"""
        try:
            drill_enum = DrillType(drill_type)
            config = drill_registry.get_config(drill_enum)
            
            if config:
                return {
                    "type": config.drill_type.value,
                    "name": config.name,
                    "description": config.description,
                    "success_criteria": config.success_criteria,
                    "time_window": config.time_window,
                    "benchmark_range": f"{config.min_reps}-{config.max_reps}",
                    "per_foot": config.per_foot,
                    "pattern_based": config.pattern_based
                }
                
        except ValueError:
            pass
            
        return None