"""
Inside-Outside Touches Drill Analyzer
Detects alternating inside and outside touches with same foot
Success: 12-18 reps per foot per set
"""

from typing import Dict, List, Optional, Tuple
import math
import logging
from dataclasses import dataclass

from drill_analyzer import DrillAnalyzer, DrillConfig, DrillType, drill_registry

logger = logging.getLogger(__name__)


@dataclass
class InsideOutsideTouch:
    """Represents a single inside-outside touch detection"""
    timestamp: float
    frame_number: int
    position: Tuple[int, int]
    foot_used: str  # "left" or "right"
    touch_type: str  # "inside" or "outside"
    confidence: float


class InsideOutsideAnalyzer(DrillAnalyzer):
    """Analyzer for Inside-Outside Touches drill"""
    
    def __init__(self, config: DrillConfig):
        super().__init__(config)
        self.touch_threshold = 45  # pixels (slightly larger than bell touches)
        self.last_touch_type = {}  # Track last touch type per foot
        self.touch_sequence = {}  # Track sequence per foot
        
    def detect_repetitions(self, video_data: Dict) -> List[Dict]:
        """Detect inside-outside touches in video data"""
        repetitions = []
        
        # Extract necessary data
        ball_detections = video_data.get("ball_detections", [])
        foot_positions = video_data.get("foot_positions", {})
        frame_height = video_data.get("frame_height", 720)
        
        # Process each frame
        for detection in ball_detections:
            if not self._is_valid_detection(detection, foot_positions, frame_height):
                continue
                
            # Determine which foot and touch type
            frame_feet = foot_positions.get(detection["frame_number"], [])
            foot_analysis = self._analyze_foot_touch(detection["position"], frame_feet)
            
            if foot_analysis and self._validate_alternating_pattern(
                foot_analysis["foot"], foot_analysis["touch_type"]
            ):
                repetitions.append({
                    "timestamp": detection["timestamp"],
                    "frame_number": detection["frame_number"],
                    "position": detection["position"],
                    "foot_used": foot_analysis["foot"],
                    "touch_type": foot_analysis["touch_type"],
                    "confidence": detection["confidence"],
                    "type": "inside_outside_touch"
                })
                
                # Update tracking
                foot = foot_analysis["foot"]
                touch_type = foot_analysis["touch_type"]
                self.last_touch_type[foot] = touch_type
                if foot not in self.touch_sequence:
                    self.touch_sequence[foot] = 0
                self.touch_sequence[foot] += 1
                
        return repetitions
    
    def validate_movement(self, movement_data: Dict) -> bool:
        """Validate if movement matches inside-outside pattern"""
        ball_pos = movement_data.get("ball_position")
        foot_positions = movement_data.get("foot_positions", [])
        frame_height = movement_data.get("frame_height", 720)
        
        if not ball_pos or len(foot_positions) < 2:
            return False
            
        # Ball should be at foot level (not too high like juggling)
        if not self._is_foot_level(ball_pos[1], frame_height):
            return False
            
        # Ball should be near one of the feet
        foot_analysis = self._analyze_foot_touch(ball_pos, foot_positions)
        return foot_analysis is not None
    
    def _is_valid_detection(self, detection: Dict, foot_positions: Dict, frame_height: int) -> bool:
        """Check if detection is valid for inside-outside touches"""
        frame_feet = foot_positions.get(detection["frame_number"], [])
        if len(frame_feet) < 2:
            return False
            
        # Check foot level (not ground level like bell touches, not high like juggling)
        ball_y = detection["position"][1]
        if not self._is_foot_level(ball_y, frame_height):
            return False
            
        # Must be near one of the feet
        foot_analysis = self._analyze_foot_touch(detection["position"], frame_feet)
        return foot_analysis is not None
    
    def _is_foot_level(self, ball_y: int, frame_height: int) -> bool:
        """Check if ball is at foot level (middle 50% of frame)"""
        # Inside-outside happens at foot level, not ground level
        foot_level_min = frame_height * 0.4  # 40% down from top
        foot_level_max = frame_height * 0.9  # 90% down from top
        return foot_level_min <= ball_y <= foot_level_max
    
    def _analyze_foot_touch(self, ball_pos: Tuple[int, int], 
                           foot_positions: List[Tuple[int, int]]) -> Optional[Dict]:
        """Analyze which foot is touching and whether it's inside or outside"""
        if len(foot_positions) < 2:
            return None
            
        ball_x, ball_y = ball_pos
        left_foot_x, left_foot_y = foot_positions[0]
        right_foot_x, right_foot_y = foot_positions[1]
        
        # Calculate distances to each foot
        left_distance = math.sqrt(
            (ball_x - left_foot_x)**2 + (ball_y - left_foot_y)**2
        )
        right_distance = math.sqrt(
            (ball_x - right_foot_x)**2 + (ball_y - right_foot_y)**2
        )
        
        # Determine which foot is touching
        touching_foot = None
        foot_x = None
        
        if left_distance < self.touch_threshold and left_distance < right_distance:
            touching_foot = "left"
            foot_x = left_foot_x
        elif right_distance < self.touch_threshold and right_distance < left_distance:
            touching_foot = "right"
            foot_x = right_foot_x
        
        if not touching_foot:
            return None
            
        # Determine inside vs outside touch based on ball position relative to foot
        # Inside touch: ball is towards the center of the body
        # Outside touch: ball is towards the outside of the body
        if touching_foot == "left":
            # For left foot: inside = ball to the right of foot, outside = ball to the left
            touch_type = "inside" if ball_x > foot_x - 20 else "outside"
        else:  # right foot
            # For right foot: inside = ball to the left of foot, outside = ball to the right
            touch_type = "inside" if ball_x < foot_x + 20 else "outside"
            
        return {
            "foot": touching_foot,
            "touch_type": touch_type,
            "distance": min(left_distance, right_distance)
        }
    
    def _validate_alternating_pattern(self, foot: str, touch_type: str) -> bool:
        """Validate that touches alternate between inside and outside for same foot"""
        # First touch for this foot is always valid
        if foot not in self.last_touch_type:
            return True
            
        # Must alternate between inside and outside for same foot
        return touch_type != self.last_touch_type[foot]
    
    def calculate_range(self, count: int, confidence: float) -> Dict:
        """Calculate range for inside-outside touches"""
        # Similar uncertainty to bell touches but slightly higher
        base_uncertainty = 2
        
        if confidence >= 0.8:
            confidence_factor = 0.6
        elif confidence >= 0.6:
            confidence_factor = 1.0
        else:
            confidence_factor = 1.4
            
        uncertainty = int(base_uncertainty * confidence_factor)
        uncertainty = max(1, min(uncertainty, 3))
        
        range_min = max(0, count - uncertainty)
        range_max = count + uncertainty
        
        return {
            "min": range_min,
            "max": range_max,
            "display": f"{range_min}-{range_max} inside-outside reps",
            "detected_count": count,
            "confidence_level": "high" if confidence >= 0.7 else "medium" if confidence >= 0.5 else "low"
        }
    
    def _get_unit_name(self, count: int) -> str:
        """Get unit name for inside-outside touches"""
        return "inside-outside reps" if count != 1 else "inside-outside rep"
    
    def _count_per_foot(self, repetitions: List[Dict]) -> Dict[str, int]:
        """Count inside-outside reps per foot with pattern analysis"""
        counts = {"left": 0, "right": 0}
        foot_patterns = {"left": [], "right": []}
        
        # Count reps and track patterns
        for rep in repetitions:
            foot = rep.get("foot_used", "").lower()
            touch_type = rep.get("touch_type", "")
            
            if foot in counts:
                counts[foot] += 1
                foot_patterns[foot].append(touch_type)
        
        # Analyze pattern quality for each foot
        for foot in ["left", "right"]:
            pattern = foot_patterns[foot]
            if len(pattern) > 1:
                # Check alternating pattern quality
                alternations = sum(1 for i in range(1, len(pattern)) 
                                 if pattern[i] != pattern[i-1])
                expected_alternations = len(pattern) - 1
                alternation_rate = alternations / expected_alternations if expected_alternations > 0 else 1.0
                counts[f"{foot}_pattern_quality"] = self._rate_pattern_quality(alternation_rate)
                counts[f"{foot}_alternation_rate"] = round(alternation_rate, 2)
        
        return counts
    
    def _rate_pattern_quality(self, alternation_rate: float) -> str:
        """Rate the quality of inside-outside alternating pattern"""
        if alternation_rate >= 0.9:
            return "excellent"
        elif alternation_rate >= 0.7:
            return "good"
        elif alternation_rate >= 0.5:
            return "fair"
        else:
            return "needs improvement"


# Register the analyzer
drill_registry.register_analyzer(DrillType.INSIDE_OUTSIDE, InsideOutsideAnalyzer)