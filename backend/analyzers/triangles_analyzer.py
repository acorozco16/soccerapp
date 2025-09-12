"""
Triangles Drill Analyzer
Detects triangle pattern movement with different foot surfaces
Success: 4-8 full patterns in 20-30 seconds
"""

from typing import Dict, List, Optional, Tuple
import math
import logging
from dataclasses import dataclass

from drill_analyzer import DrillAnalyzer, DrillConfig, DrillType, drill_registry

logger = logging.getLogger(__name__)


@dataclass
class TrianglePoint:
    """Represents a point in the triangle pattern"""
    timestamp: float
    frame_number: int
    position: Tuple[int, int]
    foot_used: str  # "left" or "right"
    surface: str  # "inside", "outside", "sole"
    confidence: float


class TrianglesAnalyzer(DrillAnalyzer):
    """Analyzer for Triangles drill"""
    
    def __init__(self, config: DrillConfig):
        super().__init__(config)
        self.touch_threshold = 45  # pixels
        self.min_triangle_side = 40  # minimum distance for triangle side
        self.max_triangle_side = 120  # maximum distance for triangle side
        self.pattern_points = []  # Track points in current pattern
        self.completed_triangles = []  # Track completed triangles
        self.max_pattern_time = 5.0  # Maximum time for one triangle pattern
        
    def detect_repetitions(self, video_data: Dict) -> List[Dict]:
        """Detect triangle patterns in video data"""
        repetitions = []
        
        # Extract necessary data
        ball_detections = video_data.get("ball_detections", [])
        foot_positions = video_data.get("foot_positions", {})
        frame_height = video_data.get("frame_height", 720)
        
        # Process each frame
        for detection in ball_detections:
            if not self._is_valid_detection(detection, foot_positions, frame_height):
                continue
                
            # Analyze triangle movement
            frame_feet = foot_positions.get(detection["frame_number"], [])
            triangle_point = self._analyze_triangle_point(detection, frame_feet)
            
            if triangle_point:
                self._process_triangle_pattern(triangle_point, repetitions)
                
        # Process any remaining incomplete patterns
        self._finalize_patterns(repetitions)
                
        return repetitions
    
    def validate_movement(self, movement_data: Dict) -> bool:
        """Validate if movement matches triangle pattern"""
        ball_pos = movement_data.get("ball_position")
        foot_positions = movement_data.get("foot_positions", [])
        frame_height = movement_data.get("frame_height", 720)
        
        if not ball_pos or len(foot_positions) < 2:
            return False
            
        # Ball should be at foot level
        if not self._is_foot_level(ball_pos[1], frame_height):
            return False
            
        # Ball should be near one foot
        foot_analysis = self._analyze_foot_contact(ball_pos, foot_positions)
        return foot_analysis is not None
    
    def _is_valid_detection(self, detection: Dict, foot_positions: Dict, frame_height: int) -> bool:
        """Check if detection is valid for triangles"""
        frame_feet = foot_positions.get(detection["frame_number"], [])
        if len(frame_feet) < 2:
            return False
            
        # Check foot level
        ball_y = detection["position"][1]
        if not self._is_foot_level(ball_y, frame_height):
            return False
            
        # Must be near one foot
        foot_analysis = self._analyze_foot_contact(detection["position"], frame_feet)
        return foot_analysis is not None
    
    def _is_foot_level(self, ball_y: int, frame_height: int) -> bool:
        """Check if ball is at foot level"""
        foot_level_min = frame_height * 0.4  # 40% down from top
        foot_level_max = frame_height * 0.9  # 90% down from top
        return foot_level_min <= ball_y <= foot_level_max
    
    def _analyze_foot_contact(self, ball_pos: Tuple[int, int], 
                             foot_positions: List[Tuple[int, int]]) -> Optional[Dict]:
        """Analyze foot contact and surface type"""
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
        
        # Determine which foot is touching
        touching_foot = None
        foot_pos = None
        
        if left_distance < self.touch_threshold and left_distance < right_distance:
            touching_foot = "left"
            foot_pos = (left_foot_x, left_foot_y)
        elif right_distance < self.touch_threshold and right_distance < left_distance:
            touching_foot = "right"
            foot_pos = (right_foot_x, right_foot_y)
        
        if not touching_foot:
            return None
            
        # Determine surface based on ball position relative to foot
        surface = self._determine_surface(ball_pos, foot_pos, touching_foot)
        
        return {
            "foot": touching_foot,
            "surface": surface,
            "distance": min(left_distance, right_distance),
            "foot_pos": foot_pos
        }
    
    def _determine_surface(self, ball_pos: Tuple[int, int], foot_pos: Tuple[int, int], foot: str) -> str:
        """Determine which part of foot is touching ball"""
        ball_x, ball_y = ball_pos
        foot_x, foot_y = foot_pos
        
        relative_x = ball_x - foot_x
        relative_y = ball_y - foot_y
        
        # Sole: ball is under/behind the foot
        if abs(relative_x) < 25 and relative_y >= -5:
            return "sole"
        
        # Inside/Outside based on foot and ball position
        if foot == "left":
            if relative_x > 15:  # Ball to right of left foot
                return "inside"
            elif relative_x < -15:  # Ball to left of left foot
                return "outside"
        else:  # right foot
            if relative_x < -15:  # Ball to left of right foot
                return "inside"
            elif relative_x > 15:  # Ball to right of right foot
                return "outside"
        
        return "inside"  # Default
    
    def _analyze_triangle_point(self, detection: Dict, foot_positions: List[Tuple[int, int]]) -> Optional[Dict]:
        """Analyze if this detection represents a triangle pattern point"""
        foot_contact = self._analyze_foot_contact(detection["position"], foot_positions)
        if not foot_contact:
            return None
            
        return {
            "timestamp": detection["timestamp"],
            "frame_number": detection["frame_number"],
            "position": detection["position"],
            "foot": foot_contact["foot"],
            "surface": foot_contact["surface"],
            "confidence": detection["confidence"]
        }
    
    def _process_triangle_pattern(self, triangle_point: Dict, repetitions: List[Dict]):
        """Process triangle point and detect complete patterns"""
        current_time = triangle_point["timestamp"]
        
        # Clean old points that are too old
        self.pattern_points = [
            p for p in self.pattern_points 
            if current_time - p["timestamp"] <= self.max_pattern_time
        ]
        
        # Add current point
        self.pattern_points.append(triangle_point)
        
        # Check if we have enough points for a triangle (at least 3)
        if len(self.pattern_points) >= 3:
            triangle = self._detect_triangle_completion()
            if triangle:
                repetitions.append(triangle)
                # Clear pattern points after completing triangle
                self.pattern_points = []
    
    def _detect_triangle_completion(self) -> Optional[Dict]:
        """Detect if current points form a complete triangle"""
        if len(self.pattern_points) < 3:
            return None
            
        # Take last 3-4 points to check for triangle
        recent_points = self.pattern_points[-4:] if len(self.pattern_points) >= 4 else self.pattern_points[-3:]
        
        # Check if points form a reasonable triangle shape
        triangle_analysis = self._analyze_triangle_geometry(recent_points)
        if not triangle_analysis["is_triangle"]:
            return None
            
        # Check surface variety (triangles should use different surfaces)
        surface_variety = self._analyze_surface_variety(recent_points)
        if surface_variety < 2:  # Need at least 2 different surfaces
            return None
            
        # Create triangle completion record
        first_point = recent_points[0]
        last_point = recent_points[-1]
        
        return {
            "timestamp": first_point["timestamp"],
            "frame_number": first_point["frame_number"],
            "position": first_point["position"],
            "foot_used": "both",  # Triangles typically use both feet
            "surfaces_used": list(set(p["surface"] for p in recent_points)),
            "pattern_duration": last_point["timestamp"] - first_point["timestamp"],
            "triangle_quality": triangle_analysis["quality"],
            "surface_variety": surface_variety,
            "confidence": sum(p["confidence"] for p in recent_points) / len(recent_points),
            "type": "triangle_pattern"
        }
    
    def _analyze_triangle_geometry(self, points: List[Dict]) -> Dict:
        """Analyze if points form a triangle shape"""
        if len(points) < 3:
            return {"is_triangle": False, "quality": 0}
            
        # Take first, middle, and last points to form triangle
        p1 = points[0]["position"]
        p2 = points[len(points)//2]["position"]
        p3 = points[-1]["position"]
        
        # Calculate side lengths
        side1 = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        side2 = math.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
        side3 = math.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2)
        
        # Check if sides are reasonable lengths
        sides = [side1, side2, side3]
        if any(side < self.min_triangle_side or side > self.max_triangle_side for side in sides):
            return {"is_triangle": False, "quality": 0}
            
        # Check triangle inequality
        if not (side1 + side2 > side3 and side2 + side3 > side1 and side1 + side3 > side2):
            return {"is_triangle": False, "quality": 0}
            
        # Calculate quality based on how equilateral the triangle is
        avg_side = sum(sides) / 3
        side_variance = sum((side - avg_side)**2 for side in sides) / 3
        quality = max(0, 1 - (side_variance / avg_side**2))
        
        return {"is_triangle": True, "quality": quality}
    
    def _analyze_surface_variety(self, points: List[Dict]) -> int:
        """Count unique surfaces used in the pattern"""
        surfaces = set(point["surface"] for point in points)
        return len(surfaces)
    
    def _finalize_patterns(self, repetitions: List[Dict]):
        """Process any remaining incomplete patterns"""
        # This could be used to handle patterns that were started but not completed
        # For now, we only count complete triangles
        pass
    
    def calculate_range(self, count: int, confidence: float) -> Dict:
        """Calculate range for triangle patterns"""
        # Triangle patterns are complex and may have higher uncertainty
        base_uncertainty = 1  # Lower base since patterns are more distinct
        
        if confidence >= 0.8:
            confidence_factor = 0.5
        elif confidence >= 0.6:
            confidence_factor = 0.8
        else:
            confidence_factor = 1.2
            
        uncertainty = int(base_uncertainty * confidence_factor)
        uncertainty = max(1, min(uncertainty, 2))
        
        range_min = max(0, count - uncertainty)
        range_max = count + uncertainty
        
        return {
            "min": range_min,
            "max": range_max,
            "display": f"{range_min}-{range_max} triangle patterns",
            "detected_count": count,
            "confidence_level": "high" if confidence >= 0.7 else "medium" if confidence >= 0.5 else "low"
        }
    
    def _get_unit_name(self, count: int) -> str:
        """Get unit name for triangle patterns"""
        return "triangle patterns" if count != 1 else "triangle pattern"
    
    def _count_patterns(self, repetitions: List[Dict]) -> int:
        """Count completed triangle patterns"""
        return len(repetitions)  # Each repetition is already a complete pattern
    
    def _count_per_foot(self, repetitions: List[Dict]) -> Dict[str, int]:
        """Analyze triangle patterns (not per-foot since triangles use both feet)"""
        counts = {"total_patterns": len(repetitions)}
        
        if repetitions:
            # Analyze surface variety across all patterns
            all_surfaces = []
            pattern_durations = []
            triangle_qualities = []
            
            for rep in repetitions:
                surfaces = rep.get("surfaces_used", [])
                duration = rep.get("pattern_duration", 0)
                quality = rep.get("triangle_quality", 0)
                
                all_surfaces.extend(surfaces)
                pattern_durations.append(duration)
                triangle_qualities.append(quality)
            
            # Surface usage analysis
            surface_counts = {}
            for surface in all_surfaces:
                surface_counts[surface] = surface_counts.get(surface, 0) + 1
            
            counts["surface_usage"] = surface_counts
            
            # Average metrics
            if pattern_durations:
                counts["avg_duration"] = round(sum(pattern_durations) / len(pattern_durations), 2)
            if triangle_qualities:
                avg_quality = sum(triangle_qualities) / len(triangle_qualities)
                counts["avg_quality"] = round(avg_quality, 2)
                counts["pattern_quality"] = self._rate_triangle_quality(avg_quality)
        
        return counts
    
    def _rate_triangle_quality(self, avg_quality: float) -> str:
        """Rate the quality of triangle patterns"""
        if avg_quality >= 0.8:
            return "excellent"
        elif avg_quality >= 0.6:
            return "good"
        elif avg_quality >= 0.4:
            return "fair"
        else:
            return "needs improvement"


# Register the analyzer
drill_registry.register_analyzer(DrillType.TRIANGLES, TrianglesAnalyzer)