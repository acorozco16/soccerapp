"""
Juggling (Keep-ups) Drill Analyzer
Migrates existing juggling analysis to the new framework
Success: Maintain control for 30-60 seconds
"""

from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

from drill_analyzer import DrillAnalyzer, DrillConfig, DrillType, drill_registry

logger = logging.getLogger(__name__)


@dataclass
class JugglingTouch:
    """Represents a single juggling touch detection"""
    timestamp: float
    frame_number: int
    position: Tuple[int, int]
    confidence: float
    detection_method: str


class JugglingAnalyzer(DrillAnalyzer):
    """Analyzer for Juggling (Keep-ups) drill"""
    
    def __init__(self, config: DrillConfig):
        super().__init__(config)
        
    def detect_repetitions(self, video_data: Dict) -> List[Dict]:
        """Extract juggling touches from existing video processor results"""
        repetitions = []
        
        # Use existing touch events from video processor
        touch_events = video_data.get("touch_events", [])
        
        for touch in touch_events:
            # Convert existing touch event format to framework format
            if self._is_juggling_touch(touch):
                repetitions.append({
                    "timestamp": touch["timestamp"],
                    "frame_number": touch["frame"],
                    "position": touch["position"],
                    "confidence": touch["confidence"],
                    "detection_method": touch["detection_method"],
                    "type": "juggling_touch"
                })
        
        return repetitions
    
    def validate_movement(self, movement_data: Dict) -> bool:
        """Validate if movement matches juggling pattern"""
        ball_pos = movement_data.get("ball_position")
        frame_height = movement_data.get("frame_height", 720)
        
        if not ball_pos:
            return False
            
        # Juggling typically happens above foot level (upper portion of frame)
        ball_y = ball_pos[1]
        juggling_zone_max = frame_height * 0.7  # Above bottom 30%
        
        return ball_y < juggling_zone_max
    
    def _is_juggling_touch(self, touch_event: Dict) -> bool:
        """Determine if touch event is a juggling touch (vs bell touch)"""
        # Juggling touches are typically higher in the frame
        position = touch_event.get("position", (0, 0))
        ball_y = position[1]
        
        # Assume frame height of 720 if not specified
        frame_height = 720
        juggling_zone_max = frame_height * 0.7  # Above bottom 30%
        
        return ball_y < juggling_zone_max
    
    def calculate_range(self, count: int, confidence: float) -> Dict:
        """Calculate range for juggling touches (uses existing logic)"""
        # Use similar logic to existing _calculate_touch_range
        base_uncertainty = 2
        
        if confidence >= 0.8:
            confidence_factor = 0.5
        elif confidence >= 0.6:
            confidence_factor = 1.0
        else:
            confidence_factor = 1.5
            
        uncertainty = int(base_uncertainty * confidence_factor)
        uncertainty = max(1, min(uncertainty, 5))  # Juggling can be less precise
        
        # Special handling for very low counts
        if count <= 3:
            range_min = max(0, count - 1)
            range_max = count + 2
        else:
            range_min = max(0, count - uncertainty)
            range_max = count + uncertainty
        
        return {
            "min": range_min,
            "max": range_max,
            "display": f"{range_min}-{range_max} touches",
            "detected_count": count,
            "confidence_level": "high" if confidence >= 0.7 else "medium" if confidence >= 0.5 else "low"
        }
    
    def check_benchmark(self, count: int, duration: float) -> bool:
        """Check if juggling performance meets benchmark criteria"""
        # Juggling benchmark is more about consistency over time
        # Rather than specific count, check touches per minute
        if duration < 30:  # Need at least 30 seconds
            return False
            
        touches_per_minute = (count / duration) * 60
        
        # Successful juggling: at least 20 touches per minute for 30+ seconds
        # This translates to about 10+ touches in 30 seconds minimum
        min_touches_per_minute = 20
        
        return touches_per_minute >= min_touches_per_minute and duration >= 30
    
    def _get_unit_name(self, count: int) -> str:
        """Get unit name for juggling touches"""
        return "touches" if count != 1 else "touch"
    
    def analyze(self, video_data: Dict) -> "DrillResults":
        """Enhanced analyze method that uses existing video processor data"""
        from drill_analyzer import DrillResults
        
        try:
            # Extract data
            duration = video_data.get("duration", 0)
            video_id = video_data.get("video_id", "")
            
            # Use existing results if available (from raw_results)
            raw_results = video_data.get("raw_results", {})
            
            if raw_results and "total_ball_touches" in raw_results:
                # Use existing analysis results
                count = raw_results["total_ball_touches"]
                confidence = raw_results.get("confidence_score", 0.5)
                
                # Use existing touch range if available
                existing_range = raw_results.get("touch_range", {})
                if existing_range:
                    count_range = {
                        "min": existing_range.get("min", count - 2),
                        "max": existing_range.get("max", count + 2),
                        "display": existing_range.get("display", f"{count-2}-{count+2} touches"),
                        "detected_count": count,
                        "confidence_level": existing_range.get("confidence_level", "medium")
                    }
                else:
                    count_range = self.calculate_range(count, confidence)
                    
            else:
                # Fallback to framework detection
                repetitions = self.detect_repetitions(video_data)
                count = len(repetitions)
                confidence = self._calculate_confidence(repetitions)
                count_range = self.calculate_range(count, confidence)
            
            # Check benchmark
            benchmark_met = self.check_benchmark(count, duration)
            
            # Build results
            results = DrillResults(
                drill_type=self.config.drill_type,
                success_criteria=self.config.success_criteria,
                count_detected=count,
                count_range=count_range,
                duration=duration,
                benchmark_met=benchmark_met,
                confidence=confidence,
                video_id=video_id
            )
            
            # Add juggling-specific metrics
            if duration > 0:
                touches_per_minute = (count / duration) * 60
                results.per_foot_counts = {
                    "touches_per_minute": round(touches_per_minute, 1),
                    "consistency": self._rate_consistency(touches_per_minute),
                    "duration_quality": self._rate_duration_quality(duration)
                }
            
            self.logger.info(f"Juggling analysis complete: {count} touches in {duration:.1f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Juggling analysis failed: {e}")
            raise
    
    def _rate_consistency(self, touches_per_minute: float) -> str:
        """Rate juggling consistency based on touches per minute"""
        if touches_per_minute >= 40:
            return "excellent"
        elif touches_per_minute >= 30:
            return "good"
        elif touches_per_minute >= 20:
            return "fair"
        else:
            return "needs improvement"
    
    def _rate_duration_quality(self, duration: float) -> str:
        """Rate juggling duration quality"""
        if duration >= 60:
            return "excellent"
        elif duration >= 45:
            return "good"
        elif duration >= 30:
            return "fair"
        else:
            return "needs improvement"


# Register the analyzer
drill_registry.register_analyzer(DrillType.JUGGLING, JugglingAnalyzer)