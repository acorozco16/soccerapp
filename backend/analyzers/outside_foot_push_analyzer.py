"""
Outside Foot Push Drill Analyzer
Detects pushing ball with outside of foot repeatedly
Success: 15-22 touches in 30 seconds
"""

from typing import Dict, List, Optional, Tuple
import math
import logging
from dataclasses import dataclass

from drill_analyzer import DrillAnalyzer, DrillConfig, DrillType, drill_registry

logger = logging.getLogger(__name__)


@dataclass
class OutsideFootPush:
    """Represents a single outside foot push detection"""
    timestamp: float
    frame_number: int
    position: Tuple[int, int]
    foot_used: str  # "left" or "right"
    push_direction: str  # "left", "right", "forward"
    confidence: float


class OutsideFootPushAnalyzer(DrillAnalyzer):
    """Analyzer for Outside Foot Push drill"""
    
    def __init__(self, config: DrillConfig):
        super().__init__(config)
        self.touch_threshold = 45  # pixels
        self.min_push_distance = 25  # minimum distance for a push
        self.last_positions = {}  # Track last position per foot
        
    def detect_repetitions(self, video_data: Dict) -> List[Dict]:
        """Detect outside foot pushes in video data"""
        repetitions = []
        
        # Extract necessary data
        ball_detections = video_data.get("ball_detections", [])
        foot_positions = video_data.get("foot_positions", {})
        frame_height = video_data.get("frame_height", 720)
        
        # Process each frame
        for detection in ball_detections:
            if not self._is_valid_detection(detection, foot_positions, frame_height):
                continue
                
            # Analyze outside foot contact and push motion
            frame_feet = foot_positions.get(detection["frame_number"], [])
            push_analysis = self._analyze_outside_push(detection, frame_feet)
            
            if push_analysis and self._validate_push_pattern(push_analysis):
                repetitions.append({
                    "timestamp": detection["timestamp"],
                    "frame_number": detection["frame_number"],
                    "position": detection["position"],
                    "foot_used": push_analysis["foot"],
                    "push_direction": push_analysis["direction"],
                    "push_distance": push_analysis["distance"],
                    "confidence": detection["confidence"],
                    "type": "outside_foot_push"
                })
                
                # Update tracking
                foot = push_analysis["foot"]
                self.last_positions[foot] = detection["position"]
                
        return repetitions
    
    def validate_movement(self, movement_data: Dict) -> bool:
        """Validate if movement matches outside foot push pattern"""
        ball_pos = movement_data.get("ball_position")
        foot_positions = movement_data.get("foot_positions", [])
        frame_height = movement_data.get("frame_height", 720)
        
        if not ball_pos or len(foot_positions) < 2:
            return False
            
        # Ball should be at foot level (not ground, not high)
        if not self._is_foot_level(ball_pos[1], frame_height):
            return False
            
        # Ball should be on outside of one foot
        outside_analysis = self._analyze_outside_contact(ball_pos, foot_positions)
        return outside_analysis is not None
    
    def _is_valid_detection(self, detection: Dict, foot_positions: Dict, frame_height: int) -> bool:
        """Check if detection is valid for outside foot pushes"""
        frame_feet = foot_positions.get(detection["frame_number"], [])
        if len(frame_feet) < 2:
            return False
            
        # Check foot level
        ball_y = detection["position"][1]
        if not self._is_foot_level(ball_y, frame_height):
            return False
            
        # Must be on outside of one foot
        outside_analysis = self._analyze_outside_contact(detection["position"], frame_feet)
        return outside_analysis is not None
    
    def _is_foot_level(self, ball_y: int, frame_height: int) -> bool:
        """Check if ball is at foot level (middle-lower portion of frame)"""
        foot_level_min = frame_height * 0.5  # 50% down from top
        foot_level_max = frame_height * 0.9  # 90% down from top
        return foot_level_min <= ball_y <= foot_level_max
    
    def _analyze_outside_contact(self, ball_pos: Tuple[int, int], 
                                foot_positions: List[Tuple[int, int]]) -> Optional[Dict]:
        """Analyze if ball is on outside of foot"""
        if len(foot_positions) < 2:
            return None
            
        ball_x, ball_y = ball_pos
        left_foot_x, left_foot_y = foot_positions[0]
        right_foot_x, right_foot_y = foot_positions[1]
        
        # Calculate distances
        left_distance = math.sqrt(
            (ball_x - left_foot_x)**2 + (ball_y - left_foot_y)**2
        )
        right_distance = math.sqrt(
            (ball_x - right_foot_x)**2 + (ball_y - right_foot_y)**2
        )
        
        # Determine which foot and check if it's on the outside
        touching_foot = None
        is_outside = False
        
        if left_distance < self.touch_threshold and left_distance < right_distance:
            touching_foot = "left"
            # For left foot: outside = ball to the left of foot
            is_outside = ball_x < left_foot_x - 10
        elif right_distance < self.touch_threshold and right_distance < left_distance:
            touching_foot = "right"
            # For right foot: outside = ball to the right of foot
            is_outside = ball_x > right_foot_x + 10
        
        if touching_foot and is_outside:
            return {
                "foot": touching_foot,
                "distance": min(left_distance, right_distance),
                "is_outside": is_outside
            }
            
        return None
    
    def _analyze_outside_push(self, detection: Dict, foot_positions: List[Tuple[int, int]]) -> Optional[Dict]:
        """Analyze outside foot push motion and direction"""
        outside_contact = self._analyze_outside_contact(detection["position"], foot_positions)
        if not outside_contact:
            return None
            
        foot = outside_contact["foot"]
        current_pos = detection["position"]
        
        # Determine push direction based on movement
        direction = "forward"  # Default
        push_distance = 0
        
        if foot in self.last_positions:
            last_pos = self.last_positions[foot]
            # Calculate movement vector
            dx = current_pos[0] - last_pos[0]
            dy = current_pos[1] - last_pos[1]
            push_distance = math.sqrt(dx**2 + dy**2)
            
            # Determine push direction
            if abs(dx) > abs(dy):  # More horizontal movement
                if foot == "left":
                    direction = "left" if dx < 0 else "right"
                else:  # right foot
                    direction = "right" if dx > 0 else "left"
            else:  # More vertical movement
                direction = "forward" if dy > 0 else "backward"
        
        return {
            "foot": foot,
            "direction": direction,
            "distance": push_distance,
            "outside_contact": outside_contact
        }
    
    def _validate_push_pattern(self, push_analysis: Dict) -> bool:
        """Validate that push shows sufficient movement and proper outside contact"""
        # Must have minimum push distance
        if push_analysis["distance"] < self.min_push_distance:
            return False
            
        # Must be proper outside contact
        if not push_analysis["outside_contact"]["is_outside"]:
            return False
            
        return True
    
    def calculate_range(self, count: int, confidence: float) -> Dict:
        """Calculate range for outside foot pushes"""
        # Similar to bell touches but slightly more uncertainty
        base_uncertainty = 2
        
        if confidence >= 0.8:
            confidence_factor = 0.6
        elif confidence >= 0.6:
            confidence_factor = 1.0
        else:
            confidence_factor = 1.3
            
        uncertainty = int(base_uncertainty * confidence_factor)
        uncertainty = max(1, min(uncertainty, 3))
        
        range_min = max(0, count - uncertainty)
        range_max = count + uncertainty
        
        return {
            "min": range_min,
            "max": range_max,
            "display": f"{range_min}-{range_max} outside foot pushes",
            "detected_count": count,
            "confidence_level": "high" if confidence >= 0.7 else "medium" if confidence >= 0.5 else "low"
        }
    
    def _get_unit_name(self, count: int) -> str:
        """Get unit name for outside foot pushes"""
        return "outside foot pushes" if count != 1 else "outside foot push"
    
    def _count_per_foot(self, repetitions: List[Dict]) -> Dict[str, int]:
        """Count outside foot pushes per foot with direction analysis"""
        counts = {"left": 0, "right": 0}
        foot_directions = {"left": [], "right": []}
        
        # Count pushes and track directions
        for rep in repetitions:
            foot = rep.get("foot_used", "").lower()
            direction = rep.get("push_direction", "")
            
            if foot in counts:
                counts[foot] += 1
                foot_directions[foot].append(direction)
        
        # Analyze push directions for each foot
        for foot in ["left", "right"]:
            directions = foot_directions[foot]
            if directions:
                # Count direction variety
                unique_directions = len(set(directions))
                counts[f"{foot}_direction_variety"] = unique_directions
                
                # Most common direction
                direction_counts = {}
                for direction in directions:
                    direction_counts[direction] = direction_counts.get(direction, 0) + 1
                
                if direction_counts:
                    most_common = max(direction_counts.items(), key=lambda x: x[1])
                    counts[f"{foot}_primary_direction"] = most_common[0]
        
        return counts


# Register the analyzer
drill_registry.register_analyzer(DrillType.OUTSIDE_FOOT_PUSH, OutsideFootPushAnalyzer)