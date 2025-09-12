"""
Sole Rolls Drill Analyzer
Detects rolling ball back and forth with sole of foot
Success: 8-14 smooth rolls in 20-30 seconds
"""

from typing import Dict, List, Optional, Tuple
import math
import logging
from dataclasses import dataclass

from drill_analyzer import DrillAnalyzer, DrillConfig, DrillType, drill_registry

logger = logging.getLogger(__name__)


@dataclass
class SoleRoll:
    """Represents a single sole roll detection"""
    timestamp: float
    frame_number: int
    position: Tuple[int, int]
    foot_used: str  # "left" or "right"
    direction: str  # "forward" or "backward"
    confidence: float


class SoleRollsAnalyzer(DrillAnalyzer):
    """Analyzer for Sole Rolls drill"""
    
    def __init__(self, config: DrillConfig):
        super().__init__(config)
        self.touch_threshold = 50  # pixels (larger for sole contact)
        self.min_roll_distance = 30  # minimum distance for a roll
        self.last_positions = {}  # Track last position per foot
        self.roll_direction = {}  # Track current roll direction per foot
        
    def detect_repetitions(self, video_data: Dict) -> List[Dict]:
        """Detect sole rolls in video data"""
        repetitions = []
        
        # Extract necessary data
        ball_detections = video_data.get("ball_detections", [])
        foot_positions = video_data.get("foot_positions", {})
        frame_height = video_data.get("frame_height", 720)
        
        # Process each frame
        for detection in ball_detections:
            if not self._is_valid_detection(detection, foot_positions, frame_height):
                continue
                
            # Analyze sole contact and rolling motion
            frame_feet = foot_positions.get(detection["frame_number"], [])
            sole_analysis = self._analyze_sole_roll(detection, frame_feet)
            
            if sole_analysis and self._validate_roll_pattern(sole_analysis):
                repetitions.append({
                    "timestamp": detection["timestamp"],
                    "frame_number": detection["frame_number"],
                    "position": detection["position"],
                    "foot_used": sole_analysis["foot"],
                    "direction": sole_analysis["direction"],
                    "roll_distance": sole_analysis["distance"],
                    "confidence": detection["confidence"],
                    "type": "sole_roll"
                })
                
                # Update tracking
                foot = sole_analysis["foot"]
                self.last_positions[foot] = detection["position"]
                self.roll_direction[foot] = sole_analysis["direction"]
                
        return repetitions
    
    def validate_movement(self, movement_data: Dict) -> bool:
        """Validate if movement matches sole rolls pattern"""
        ball_pos = movement_data.get("ball_position")
        foot_positions = movement_data.get("foot_positions", [])
        frame_height = movement_data.get("frame_height", 720)
        
        if not ball_pos or len(foot_positions) < 2:
            return False
            
        # Ball should be at ground level for sole rolls
        if not self._is_ground_level(ball_pos[1], frame_height):
            return False
            
        # Ball should be directly under/near one foot
        sole_analysis = self._analyze_sole_contact(ball_pos, foot_positions)
        return sole_analysis is not None
    
    def _is_valid_detection(self, detection: Dict, foot_positions: Dict, frame_height: int) -> bool:
        """Check if detection is valid for sole rolls"""
        frame_feet = foot_positions.get(detection["frame_number"], [])
        if len(frame_feet) < 2:
            return False
            
        # Check ground level (sole rolls happen on the ground)
        ball_y = detection["position"][1]
        if not self._is_ground_level(ball_y, frame_height):
            return False
            
        # Must be under one of the feet
        sole_analysis = self._analyze_sole_contact(detection["position"], frame_feet)
        return sole_analysis is not None
    
    def _is_ground_level(self, ball_y: int, frame_height: int) -> bool:
        """Check if ball is at ground level (bottom 30% of frame)"""
        ground_threshold = frame_height * 0.7  # Bottom 30%
        return ball_y >= ground_threshold
    
    def _analyze_sole_contact(self, ball_pos: Tuple[int, int], 
                             foot_positions: List[Tuple[int, int]]) -> Optional[Dict]:
        """Analyze if ball is under sole of foot"""
        if len(foot_positions) < 2:
            return None
            
        ball_x, ball_y = ball_pos
        left_foot_x, left_foot_y = foot_positions[0]
        right_foot_x, right_foot_y = foot_positions[1]
        
        # Calculate distances (emphasis on vertical alignment for sole contact)
        left_distance = math.sqrt(
            (ball_x - left_foot_x)**2 + (ball_y - left_foot_y)**2
        )
        right_distance = math.sqrt(
            (ball_x - right_foot_x)**2 + (ball_y - right_foot_y)**2
        )
        
        # For sole rolls, ball should be directly under the foot
        # Check horizontal alignment (ball should be close to foot x-position)
        left_horizontal_dist = abs(ball_x - left_foot_x)
        right_horizontal_dist = abs(ball_x - right_foot_x)
        
        touching_foot = None
        if (left_distance < self.touch_threshold and 
            left_horizontal_dist < 40 and  # Close horizontal alignment
            left_distance < right_distance):
            touching_foot = "left"
        elif (right_distance < self.touch_threshold and 
              right_horizontal_dist < 40 and  # Close horizontal alignment
              right_distance < left_distance):
            touching_foot = "right"
        
        if touching_foot:
            return {
                "foot": touching_foot,
                "distance": min(left_distance, right_distance),
                "horizontal_alignment": min(left_horizontal_dist, right_horizontal_dist)
            }
            
        return None
    
    def _analyze_sole_roll(self, detection: Dict, foot_positions: List[Tuple[int, int]]) -> Optional[Dict]:
        """Analyze sole roll motion and direction"""
        sole_contact = self._analyze_sole_contact(detection["position"], foot_positions)
        if not sole_contact:
            return None
            
        foot = sole_contact["foot"]
        current_pos = detection["position"]
        
        # Determine roll direction based on movement
        direction = "forward"  # Default
        roll_distance = 0
        
        if foot in self.last_positions:
            last_pos = self.last_positions[foot]
            # Calculate movement vector
            dx = current_pos[0] - last_pos[0]
            dy = current_pos[1] - last_pos[1]
            roll_distance = math.sqrt(dx**2 + dy**2)
            
            # Determine direction based on y-movement (forward/backward)
            if abs(dy) > abs(dx):  # More vertical than horizontal movement
                direction = "forward" if dy > 0 else "backward"
            else:  # More horizontal movement
                direction = "forward" if dx > 0 else "backward"
        
        return {
            "foot": foot,
            "direction": direction,
            "distance": roll_distance,
            "sole_contact": sole_contact
        }
    
    def _validate_roll_pattern(self, sole_analysis: Dict) -> bool:
        """Validate that roll shows sufficient movement"""
        # Must have minimum roll distance
        if sole_analysis["distance"] < self.min_roll_distance:
            return False
            
        # Good sole contact (close horizontal alignment)
        if sole_analysis["sole_contact"]["horizontal_alignment"] > 50:
            return False
            
        return True
    
    def calculate_range(self, count: int, confidence: float) -> Dict:
        """Calculate range for sole rolls"""
        # Sole rolls can be harder to detect consistently
        base_uncertainty = 2
        
        if confidence >= 0.8:
            confidence_factor = 0.7
        elif confidence >= 0.6:
            confidence_factor = 1.1
        else:
            confidence_factor = 1.5
            
        uncertainty = int(base_uncertainty * confidence_factor)
        uncertainty = max(1, min(uncertainty, 3))
        
        range_min = max(0, count - uncertainty)
        range_max = count + uncertainty
        
        return {
            "min": range_min,
            "max": range_max,
            "display": f"{range_min}-{range_max} sole rolls",
            "detected_count": count,
            "confidence_level": "high" if confidence >= 0.7 else "medium" if confidence >= 0.5 else "low"
        }
    
    def _get_unit_name(self, count: int) -> str:
        """Get unit name for sole rolls"""
        return "sole rolls" if count != 1 else "sole roll"
    
    def _count_per_foot(self, repetitions: List[Dict]) -> Dict[str, int]:
        """Count sole rolls per foot with quality analysis"""
        counts = {"left": 0, "right": 0}
        foot_distances = {"left": [], "right": []}
        
        # Count rolls and track quality metrics
        for rep in repetitions:
            foot = rep.get("foot_used", "").lower()
            distance = rep.get("roll_distance", 0)
            
            if foot in counts:
                counts[foot] += 1
                foot_distances[foot].append(distance)
        
        # Analyze roll quality for each foot
        for foot in ["left", "right"]:
            distances = foot_distances[foot]
            if distances:
                avg_distance = sum(distances) / len(distances)
                counts[f"{foot}_avg_distance"] = round(avg_distance, 1)
                counts[f"{foot}_quality"] = self._rate_roll_quality(avg_distance)
        
        return counts
    
    def _rate_roll_quality(self, avg_distance: float) -> str:
        """Rate the quality of sole rolls based on distance"""
        if avg_distance >= 60:
            return "excellent"
        elif avg_distance >= 45:
            return "good"
        elif avg_distance >= 30:
            return "fair"
        else:
            return "needs improvement"


# Register the analyzer
drill_registry.register_analyzer(DrillType.SOLE_ROLLS, SoleRollsAnalyzer)