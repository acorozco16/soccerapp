"""
Video Feature Caching System
Prevents redundant expensive processing (ByteTrack + MediaPipe)
"""

import hashlib
import json
import os
import time
from typing import Dict, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VideoFeatureCache:
    """
    Cache for expensive video feature extraction
    
    Key insight: The expensive part is ByteTrack + MediaPipe detection
    Once we have ball_detections and foot_positions, different drill
    analyzers can reuse this data without re-processing the video.
    """
    
    def __init__(self, cache_dir: str = "cache/video_features", max_age_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_seconds = max_age_hours * 3600
        
        # In-memory cache for current session
        self.memory_cache: Dict[str, Dict] = {}
        
    def _get_video_hash(self, video_path: str) -> str:
        """Generate hash for video file"""
        # Use file path + modification time + size for hash
        # This catches if same filename has different content
        stat = os.stat(video_path)
        content = f"{video_path}:{stat.st_mtime}:{stat.st_size}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_path(self, video_hash: str) -> Path:
        """Get cache file path for video hash"""
        return self.cache_dir / f"{video_hash}.json"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is still valid (not too old)"""
        if not cache_path.exists():
            return False
        
        age = time.time() - cache_path.stat().st_mtime
        return age < self.max_age_seconds
    
    def get_cached_features(self, video_path: str) -> Optional[Dict]:
        """Get cached video features if available"""
        video_hash = self._get_video_hash(video_path)
        
        # Check in-memory cache first (fastest)
        if video_hash in self.memory_cache:
            logger.info(f"Cache HIT (memory): {video_path}")
            return self.memory_cache[video_hash]
        
        # Check disk cache
        cache_path = self._get_cache_path(video_hash)
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    features = json.load(f)
                
                # Load into memory cache for next time
                self.memory_cache[video_hash] = features
                logger.info(f"Cache HIT (disk): {video_path}")
                return features
                
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Cache file corrupted: {cache_path}, error: {e}")
                cache_path.unlink(missing_ok=True)
        
        logger.info(f"Cache MISS: {video_path}")
        return None
    
    def cache_features(self, video_path: str, features: Dict) -> None:
        """Cache video features to disk and memory"""
        video_hash = self._get_video_hash(video_path)
        
        # Save to memory cache
        self.memory_cache[video_hash] = features
        
        # Save to disk cache
        cache_path = self._get_cache_path(video_hash)
        try:
            with open(cache_path, 'w') as f:
                json.dump(features, f, indent=2)
            
            logger.info(f"Cached features: {video_path} -> {cache_path}")
            
        except IOError as e:
            logger.error(f"Failed to cache features: {e}")
    
    def clear_old_cache(self) -> int:
        """Remove old cache files and return count removed"""
        removed = 0
        current_time = time.time()
        
        for cache_file in self.cache_dir.glob("*.json"):
            age = current_time - cache_file.stat().st_mtime
            if age > self.max_age_seconds:
                cache_file.unlink()
                removed += 1
        
        logger.info(f"Removed {removed} old cache files")
        return removed
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "cache_dir": str(self.cache_dir),
            "total_files": len(cache_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "memory_cached": len(self.memory_cache),
            "max_age_hours": self.max_age_seconds / 3600
        }


# Enhanced UnifiedProcessor with caching
class CachedUnifiedVideoProcessor:
    """Enhanced UnifiedVideoProcessor with feature caching"""
    
    def __init__(self, cache_dir: str = "cache/video_features"):
        from video_processor import VideoProcessor
        from drill_analyzer import drill_registry
        
        self.base_processor = VideoProcessor()
        self.drill_registry = drill_registry
        self.cache = VideoFeatureCache(cache_dir)
        self.logger = logging.getLogger(__name__)
    
    def analyze_drill(self, video_path: str, drill_type: str, video_id: str = None) -> Dict:
        """Analyze video with caching to avoid redundant processing"""
        
        start_time = time.time()
        
        # Validate drill type
        from drill_analyzer import DrillType
        try:
            drill_enum = DrillType(drill_type)
        except ValueError:
            raise ValueError(f"Unknown drill type: {drill_type}")
        
        # Get analyzer
        analyzer = self.drill_registry.get_analyzer(drill_enum)
        if not analyzer:
            raise ValueError(f"No analyzer available for drill: {drill_type}")
        
        self.logger.info(f"Starting {drill_type} analysis for video: {video_path}")
        
        # Try to get cached features first
        video_data = self.cache.get_cached_features(video_path)
        
        if video_data is None:
            # Cache miss - extract features the expensive way
            self.logger.info("Extracting video features (expensive operation)...")
            feature_extraction_start = time.time()
            
            video_data = self._extract_video_features(video_path, video_id)
            
            # Cache the extracted features for future use
            self.cache.cache_features(video_path, video_data)
            
            feature_extraction_time = time.time() - feature_extraction_start
            self.logger.info(f"Feature extraction took {feature_extraction_time:.1f}s")
        else:
            self.logger.info("Using cached video features (fast!)")
        
        # Update video_id if provided (might be different per request)
        if video_id:
            video_data["video_id"] = video_id
        
        # Run drill-specific analysis (this is fast)
        analysis_start = time.time()
        results = analyzer.analyze(video_data)
        analysis_time = time.time() - analysis_start
        
        # Add timing information
        total_time = time.time() - start_time
        results.processing_time = total_time
        
        self.logger.info(
            f"Analysis complete: {results.count_detected} reps detected "
            f"(analysis: {analysis_time:.1f}s, total: {total_time:.1f}s)"
        )
        
        return results.to_dict()
    
    def _extract_video_features(self, video_path: str, video_id: str) -> Dict:
        """Extract expensive video features (ByteTrack + MediaPipe)"""
        # This is the expensive operation we want to cache
        raw_results = self.base_processor.analyze_video(video_path, video_id)
        
        # Transform to format expected by drill analyzers
        video_data = {
            "video_id": video_id,
            "duration": raw_results["video_duration"],
            "frame_height": 720,
            "frame_width": 1280,
            "fps": 30,
            "ball_detections": self._transform_ball_detections(raw_results),
            "foot_positions": self._extract_foot_positions(raw_results),
            "touch_events": raw_results.get("touch_events", []),
            "raw_results": raw_results,
            "extraction_timestamp": time.time()  # For cache validation
        }
        
        return video_data
    
    def _transform_ball_detections(self, raw_results: Dict) -> list:
        """Transform ball detection data for drill analyzers"""
        detections = []
        
        for event in raw_results.get("touch_events", []):
            detections.append({
                "timestamp": event["timestamp"],
                "frame_number": event["frame"],
                "position": event["position"],
                "confidence": event["confidence"],
                "method": event["detection_method"]
            })
        
        detections.sort(key=lambda x: x["timestamp"])
        return detections
    
    def _extract_foot_positions(self, raw_results: Dict) -> Dict[int, list]:
        """Extract foot positions by frame number"""
        foot_positions = {}
        
        for event in raw_results.get("touch_events", []):
            frame = event["frame"]
            ball_x, ball_y = event["position"]
            # Simulate foot positions around ball
            foot_positions[frame] = [
                (ball_x - 100, ball_y),  # Left foot
                (ball_x + 100, ball_y)   # Right foot
            ]
        
        return foot_positions
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        return self.cache.get_cache_stats()
    
    def clear_old_cache(self) -> int:
        """Clear old cache files"""
        return self.cache.clear_old_cache()


# Usage example:
"""
# Instead of:
processor = UnifiedVideoProcessor()
results1 = processor.analyze_drill("video.mp4", "bell_touches")     # 3 minutes
results2 = processor.analyze_drill("video.mp4", "sole_rolls")       # 3 minutes again!

# Use:
processor = CachedUnifiedVideoProcessor()
results1 = processor.analyze_drill("video.mp4", "bell_touches")     # 3 minutes (cache miss)
results2 = processor.analyze_drill("video.mp4", "sole_rolls")       # 5 seconds (cache hit!)

# Check cache performance:
stats = processor.get_cache_stats()
print(f"Cache has {stats['total_files']} files, {stats['total_size_mb']} MB")
"""