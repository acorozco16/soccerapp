"""
Croquetas Drill Analyzer
Detects side-to-side cutting movements with sole
Success: 8-15 smooth cuts in 15-30 seconds
"""

from typing import Dict, List, Optional, Tuple
import math
import logging
from dataclasses import dataclass

from drill_analyzer import DrillAnalyzer, DrillConfig, DrillType, drill_registry

logger = logging.getLogger(__name__)


@dataclass
class Croqueta:
    """Represents a single croqueta detection"""
    timestamp: float
    frame_number: int
    position: Tuple[int, int]
    foot_used: str  # "left" or "right"
    direction: str  # "left", "right"
    confidence: float


class CroquetasAnalyzer(DrillAnalyzer):
    """Analyzer for Croquetas drill"""
    
    def __init__(self, config: DrillConfig):
        super().__init__(config)
        self.touch_threshold = 50  # pixels (larger for sole contact)
        self.min_cut_distance = 35  # minimum distance for a croqueta
        self.last_positions = []  # Track recent positions for direction analysis
        self.max_position_history = 5  # Keep last 5 positions
        
    def detect_repetitions(self, video_data: Dict) -> List[Dict]:
        """Detect croquetas in video data"""
        repetitions = []
        
        # Extract necessary data
        ball_detections = video_data.get("ball_detections", [])
        foot_positions = video_data.get("foot_positions", {})
        frame_height = video_data.get("frame_height", 720)
        
        # Process each frame
        for detection in ball_detections:
            if not self._is_valid_detection(detection, foot_positions, frame_height):
                continue
                
            # Analyze croqueta motion
            frame_feet = foot_positions.get(detection["frame_number"], [])
            croqueta_analysis = self._analyze_croqueta(detection, frame_feet)
            
            if croqueta_analysis and self._validate_croqueta_pattern(croqueta_analysis):
                repetitions.append({
                    "timestamp": detection["timestamp"],
                    "frame_number": detection["frame_number"],
                    "position": detection["position"],
                    "foot_used": croqueta_analysis["foot"],
                    "direction": croqueta_analysis["direction"],
                    "cut_distance": croqueta_analysis["distance"],
                    "smoothness": croqueta_analysis["smoothness"],
                    "confidence": detection["confidence"],
                    "type": "croqueta"
                })
                
            # Update position history
            self._update_position_history(detection["position"])
                
        return repetitions
    
    def validate_movement(self, movement_data: Dict) -> bool:
        """Validate if movement matches croquetas pattern"""
        ball_pos = movement_data.get("ball_position")
        foot_positions = movement_data.get("foot_positions", [])
        frame_height = movement_data.get("frame_height", 720)
        
        if not ball_pos or len(foot_positions) < 2:
            return False
            
        # Ball should be at ground level for sole contact
        if not self._is_ground_level(ball_pos[1], frame_height):
            return False
            
        # Ball should be near feet (between or close to them)
        sole_contact = self._analyze_sole_contact(ball_pos, foot_positions)
        return sole_contact is not None
    
    def _is_valid_detection(self, detection: Dict, foot_positions: Dict, frame_height: int) -> bool:
        """Check if detection is valid for croquetas"""
        frame_feet = foot_positions.get(detection["frame_number"], [])
        if len(frame_feet) < 2:
            return False
            
        # Check ground level (croquetas are done with sole on ground)
        ball_y = detection["position"][1]
        if not self._is_ground_level(ball_y, frame_height):
            return False
            
        # Must have sole contact
        sole_contact = self._analyze_sole_contact(detection["position"], frame_feet)
        return sole_contact is not None
    
    def _is_ground_level(self, ball_y: int, frame_height: int) -> bool:
        """Check if ball is at ground level (bottom 25% of frame)"""
        ground_threshold = frame_height * 0.75
        return ball_y >= ground_threshold
    
    def _analyze_sole_contact(self, ball_pos: Tuple[int, int], 
                             foot_positions: List[Tuple[int, int]]) -> Optional[Dict]:
        """Analyze if ball has sole contact with either foot"""
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
        
        # Check for sole contact (ball should be close to foot)
        touching_foot = None
        if left_distance < self.touch_threshold:
            touching_foot = "left"
        elif right_distance < self.touch_threshold:
            touching_foot = "right"
        
        # Also check if ball is between feet (common in croquetas)
        between_feet = (
            min(left_foot_x, right_foot_x) - 30 <= ball_x <= max(left_foot_x, right_foot_x) + 30
        )
        
        if touching_foot or between_feet:
            return {
                "foot": touching_foot or "both",
                "distance": min(left_distance, right_distance),
                "between_feet": between_feet
            }
            
        return None
    
    def _analyze_croqueta(self, detection: Dict, foot_positions: List[Tuple[int, int]]) -> Optional[Dict]:
        """Analyze croqueta cutting motion"""
        sole_contact = self._analyze_sole_contact(detection["position"], foot_positions)
        if not sole_contact:
            return None
            
        current_pos = detection["position"]
        
        # Analyze movement direction and distance
        direction_analysis = self._analyze_cutting_direction(current_pos)
        if not direction_analysis:
            return None
            
        # Determine which foot is performing the croqueta
        foot = self._determine_cutting_foot(current_pos, foot_positions, direction_analysis["direction"])
        
        return {
            "foot": foot,
            "direction": direction_analysis["direction"],
            "distance": direction_analysis["distance"],
            "smoothness": direction_analysis["smoothness"],
            "sole_contact": sole_contact
        }
    
    def _analyze_cutting_direction(self, current_pos: Tuple[int, int]) -> Optional[Dict]:
        """Analyze cutting direction based on position history"""
        if len(self.last_positions) < 2:
            return None
            
        # Calculate movement vector from recent positions
        recent_pos = self.last_positions[-1]
        dx = current_pos[0] - recent_pos[0]
        dy = current_pos[1] - recent_pos[1]
        
        # Calculate distance and smoothness
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance < self.min_cut_distance:
            return None
            
        # Determine direction (croquetas are primarily horizontal)
        if abs(dx) > abs(dy):  # More horizontal than vertical movement
            direction = "right" if dx > 0 else "left"
        else:
            return None  # Too much vertical movement for croqueta
            
        # Calculate smoothness based on position consistency
        smoothness = self._calculate_movement_smoothness()
        
        return {
            "direction": direction,
            "distance": distance,
            "smoothness": smoothness
        }
    
    def _determine_cutting_foot(self, ball_pos: Tuple[int, int], 
                               foot_positions: List[Tuple[int, int]], 
                               direction: str) -> str:
        """Determine which foot is performing the croqueta based on direction"""
        if len(foot_positions) < 2:
            return "unknown"
            
        ball_x = ball_pos[0]
        left_foot_x, right_foot_x = foot_positions[0][0], foot_positions[1][0]
        
        # In croquetas, the foot doing the cutting is usually the one on the opposite side
        # of the direction (e.g., left foot cuts the ball to the right)
        if direction == "right":
            # Cutting right usually done with left foot
            return "left" if ball_x <= (left_foot_x + right_foot_x) / 2 else "right"
        else:  # direction == "left"
            # Cutting left usually done with right foot
            return "right" if ball_x >= (left_foot_x + right_foot_x) / 2 else "left"
    
    def _calculate_movement_smoothness(self) -> float:
        """Calculate smoothness of movement based on position history"""
        if len(self.last_positions) < 3:
            return 0.5  # Default moderate smoothness
            
        # Calculate variation in movement direction
        direction_changes = 0
        for i in range(2, len(self.last_positions)):
            pos1 = self.last_positions[i-2]
            pos2 = self.last_positions[i-1]
            pos3 = self.last_positions[i]
            
            # Calculate direction vectors
            vec1 = (pos2[0] - pos1[0], pos2[1] - pos1[1])
            vec2 = (pos3[0] - pos2[0], pos3[1] - pos2[1])
            
            # Check if direction changed significantly
            if vec1[0] * vec2[0] < 0 or vec1[1] * vec2[1] < 0:  # Direction flip
                direction_changes += 1
        
        # More direction changes = less smooth
        total_segments = len(self.last_positions) - 2
        if total_segments > 0:
            smoothness = 1.0 - (direction_changes / total_segments)
        else:
            smoothness = 0.5
            
        return max(0.0, min(1.0, smoothness))
    
    def _update_position_history(self, position: Tuple[int, int]):
        """Update position history for movement analysis"""
        self.last_positions.append(position)
        if len(self.last_positions) > self.max_position_history:
            self.last_positions.pop(0)
    
    def _validate_croqueta_pattern(self, croqueta_analysis: Dict) -> bool:
        """Validate that movement matches croqueta pattern"""
        # Must have sufficient cutting distance
        if croqueta_analysis["distance"] < self.min_cut_distance:
            return False
            
        # Must have reasonable smoothness
        if croqueta_analysis["smoothness"] < 0.3:
            return False
            
        return True
    
    def calculate_range(self, count: int, confidence: float) -> Dict:
        """Calculate range for croquetas"""
        # Croquetas can be harder to detect precisely
        base_uncertainty = 2
        
        if confidence >= 0.8:
            confidence_factor = 0.8
        elif confidence >= 0.6:
            confidence_factor = 1.2
        else:
            confidence_factor = 1.5
            
        uncertainty = int(base_uncertainty * confidence_factor)
        uncertainty = max(1, min(uncertainty, 4))
        
        range_min = max(0, count - uncertainty)
        range_max = count + uncertainty
        
        return {
            "min": range_min,
            "max": range_max,
            "display": f"{range_min}-{range_max} croquetas",
            "detected_count": count,
            "confidence_level": "high" if confidence >= 0.7 else "medium" if confidence >= 0.5 else "low"
        }
    
    def _get_unit_name(self, count: int) -> str:
        """Get unit name for croquetas"""
        return "croquetas" if count != 1 else "croqueta"
    
    def _count_per_foot(self, repetitions: List[Dict]) -> Dict[str, int]:
        """Count croquetas with direction and smoothness analysis"""
        counts = {"left": 0, "right": 0}
        direction_counts = {"left": 0, "right": 0}
        smoothness_scores = []
        
        # Count cuts and analyze patterns
        for rep in repetitions:
            foot = rep.get("foot_used", "").lower()
            direction = rep.get("direction", "")
            smoothness = rep.get("smoothness", 0)
            
            if foot in counts:
                counts[foot] += 1
                
            # Track direction preference
            if direction in direction_counts:
                direction_counts[direction] += 1
                
            smoothness_scores.append(smoothness)
        
        # Add overall analysis
        total = len(repetitions)
        if total > 0:
            avg_smoothness = sum(smoothness_scores) / len(smoothness_scores)
            counts["avg_smoothness"] = round(avg_smoothness, 2)
            counts["smoothness_quality"] = self._rate_smoothness_quality(avg_smoothness)
            
            # Direction balance
            left_pct = (direction_counts["left"] / total) * 100
            right_pct = (direction_counts["right"] / total) * 100
            counts["direction_balance"] = f"{left_pct:.0f}% left, {right_pct:.0f}% right"
        
        return counts
    
    def _rate_smoothness_quality(self, avg_smoothness: float) -> str:
        """Rate the smoothness quality of croquetas"""
        if avg_smoothness >= 0.8:
            return "excellent"
        elif avg_smoothness >= 0.6:
            return "good"
        elif avg_smoothness >= 0.4:
            return "fair"
        else:
            return "needs improvement"


# Register the analyzer
drill_registry.register_analyzer(DrillType.CROQUETAS, CroquetasAnalyzer)