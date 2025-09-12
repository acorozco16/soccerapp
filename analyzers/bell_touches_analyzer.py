"""
Bell Touches Drill Analyzer
Detects alternating touches between feet at ground level
Success: 18-24 touches in 30 seconds
"""

from typing import Dict, List, Optional, Tuple
import math
import logging
from dataclasses import dataclass

from drill_analyzer import DrillAnalyzer, DrillConfig, DrillType, drill_registry
from drill_decorator import register_drill_analyzer

logger = logging.getLogger(__name__)


@dataclass
class BellTouch:
    """Represents a single bell touch detection"""
    timestamp: float
    frame_number: int
    position: Tuple[int, int]
    foot_used: str  # "left" or "right"
    confidence: float


@register_drill_analyzer(DrillConfig(
    drill_type=DrillType.BELL_TOUCHES,
    name="Bell Touches",
    description="Tap ball between feet using inside of both feet",
    success_criteria="18-24 touches in 30 seconds",
    time_window=30.0,
    min_reps=18,
    max_reps=24,
    per_foot=True,
    pattern_based=False
))
class BellTouchesAnalyzer(DrillAnalyzer):
    """Analyzer for Bell Touches drill"""
    
    def __init__(self, config: DrillConfig):
        super().__init__(config)
        self.last_touch_foot = None
        self.touch_threshold = 40  # pixels
        self.ground_level_threshold = 0.75  # bottom 25% of frame
        
    def detect_repetitions(self, video_data: Dict) -> List[Dict]:
        """Detect bell touches in video data"""
        repetitions = []
        
        # Extract necessary data
        ball_detections = video_data.get("ball_detections", [])
        foot_positions = video_data.get("foot_positions", [])
        frame_height = video_data.get("frame_height", 720)
        
        # Process each frame
        for detection in ball_detections:
            if not self._is_valid_detection(detection, foot_positions, frame_height):
                continue
                
            # Check which foot is touching
            touching_foot = self._classify_touching_foot(
                detection["position"], 
                foot_positions.get(detection["frame_number"], [])
            )
            
            if touching_foot and self._validate_alternating_pattern(touching_foot):
                repetitions.append({
                    "timestamp": detection["timestamp"],
                    "frame_number": detection["frame_number"],
                    "position": detection["position"],
                    "foot_used": touching_foot,
                    "confidence": detection["confidence"],
                    "type": "bell_touch"
                })
                self.last_touch_foot = touching_foot
                
        return repetitions
    
    def validate_movement(self, movement_data: Dict) -> bool:
        """Validate if movement matches bell touches pattern"""
        ball_pos = movement_data.get("ball_position")
        foot_positions = movement_data.get("foot_positions", [])
        frame_height = movement_data.get("frame_height", 720)
        
        if not ball_pos or len(foot_positions) < 2:
            return False
            
        # Check if ball is at ground level
        if not self._is_ground_level(ball_pos[1], frame_height):
            return False
            
        # Check if ball is between feet
        if not self._is_between_feet(ball_pos[0], foot_positions):
            return False
            
        return True
    
    def _is_valid_detection(self, detection: Dict, foot_positions: Dict, frame_height: int) -> bool:
        """Check if detection is valid for bell touches"""
        # Get foot positions for this frame
        frame_feet = foot_positions.get(detection["frame_number"], [])
        if len(frame_feet) < 2:
            return False
            
        # Check ground level
        ball_y = detection["position"][1]
        if not self._is_ground_level(ball_y, frame_height):
            return False
            
        # Check if between feet
        ball_x = detection["position"][0]
        if not self._is_between_feet(ball_x, frame_feet):
            return False
            
        return True
    
    def _is_ground_level(self, ball_y: int, frame_height: int) -> bool:
        """Check if ball is at ground level (bottom 25% of frame)"""
        ground_threshold = frame_height * self.ground_level_threshold
        return ball_y >= ground_threshold
    
    def _is_between_feet(self, ball_x: int, foot_positions: List[Tuple[int, int]]) -> bool:
        """Check if ball is horizontally between feet"""
        if len(foot_positions) < 2:
            return False
            
        left_foot_x = foot_positions[0][0]
        right_foot_x = foot_positions[1][0]
        
        min_x = min(left_foot_x, right_foot_x) - 50  # Add margin
        max_x = max(left_foot_x, right_foot_x) + 50
        
        return min_x <= ball_x <= max_x
    
    def _classify_touching_foot(self, ball_pos: Tuple[int, int], 
                               foot_positions: List[Tuple[int, int]]) -> Optional[str]:
        """Determine which foot is touching the ball"""
        if len(foot_positions) < 2:
            return None
            
        ball_x, ball_y = ball_pos
        left_foot = foot_positions[0]
        right_foot = foot_positions[1]
        
        # Calculate distances
        left_distance = math.sqrt(
            (ball_x - left_foot[0])**2 + 
            (ball_y - left_foot[1])**2
        )
        right_distance = math.sqrt(
            (ball_x - right_foot[0])**2 + 
            (ball_y - right_foot[1])**2
        )
        
        # Check which foot is closer and within threshold
        if left_distance < self.touch_threshold and left_distance < right_distance:
            return "left"
        elif right_distance < self.touch_threshold and right_distance < left_distance:
            return "right"
            
        return None
    
    def _validate_alternating_pattern(self, touching_foot: str) -> bool:
        """Validate that touches alternate between feet"""
        # First touch is always valid
        if self.last_touch_foot is None:
            return True
            
        # Must alternate between feet
        return touching_foot != self.last_touch_foot
    
    def calculate_range(self, count: int, confidence: float) -> Dict:
        """Calculate range for bell touches (tighter than juggling)"""
        # Bell touches have more consistent detection
        base_uncertainty = 1  # Tighter than default
        
        if confidence >= 0.8:
            confidence_factor = 0.5
        elif confidence >= 0.6:
            confidence_factor = 0.8
        else:
            confidence_factor = 1.2
            
        uncertainty = int(base_uncertainty * confidence_factor)
        uncertainty = max(1, min(uncertainty, 2))  # Max uncertainty of 2
        
        range_min = max(0, count - uncertainty)
        range_max = count + uncertainty
        
        return {
            "min": range_min,
            "max": range_max,
            "display": f"{range_min}-{range_max} bell touches",
            "detected_count": count,
            "confidence_level": "high" if confidence >= 0.7 else "medium" if confidence >= 0.5 else "low"
        }
    
    def _get_unit_name(self, count: int) -> str:
        """Get unit name for bell touches"""
        return "bell touches" if count != 1 else "bell touch"
    
    def _count_per_foot(self, repetitions: List[Dict]) -> Dict[str, int]:
        """Count bell touches per foot with additional metrics"""
        counts = super()._count_per_foot(repetitions)
        
        # Add alternating pattern analysis
        alternations = 0
        errors = 0
        
        for i in range(1, len(repetitions)):
            if repetitions[i]["foot_used"] != repetitions[i-1]["foot_used"]:
                alternations += 1
            else:
                errors += 1
                
        # Add pattern quality to foot counts
        total = len(repetitions)
        if total > 0:
            alternation_rate = alternations / (total - 1) if total > 1 else 1.0
            counts["alternation_rate"] = round(alternation_rate, 2)
            counts["pattern_quality"] = self._rate_pattern_quality(alternation_rate)
            counts["errors"] = errors
            
        return counts
    
    def _rate_pattern_quality(self, alternation_rate: float) -> str:
        """Rate the quality of alternating pattern"""
        if alternation_rate >= 0.9:
            return "excellent"
        elif alternation_rate >= 0.7:
            return "good"
        elif alternation_rate >= 0.5:
            return "fair"
        else:
            return "needs improvement"


# No manual registration needed - decorator handles it!