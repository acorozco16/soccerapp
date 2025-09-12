"""
V Cuts (Pull-Push) Drill Analyzer
Detects pull ball back with sole, push forward with inside
Success: 6-10 cuts per foot in 20-30 seconds
"""

from typing import Dict, List, Optional, Tuple
import math
import logging
from dataclasses import dataclass

from drill_analyzer import DrillAnalyzer, DrillConfig, DrillType, drill_registry

logger = logging.getLogger(__name__)


@dataclass
class VCut:
    """Represents a single V cut detection (pull + push = 1 cut)"""
    timestamp: float
    frame_number: int
    position: Tuple[int, int]
    foot_used: str  # "left" or "right"
    phase: str  # "pull" or "push"
    confidence: float


class VCutsAnalyzer(DrillAnalyzer):
    """Analyzer for V Cuts (Pull-Push) drill"""
    
    def __init__(self, config: DrillConfig):
        super().__init__(config)
        self.touch_threshold = 45  # pixels
        self.min_pull_distance = 20  # minimum distance for pull phase
        self.min_push_distance = 20  # minimum distance for push phase
        self.last_positions = {}  # Track last position per foot
        self.cut_phase = {}  # Track current phase per foot
        self.pending_cuts = {}  # Track incomplete cuts
        
    def detect_repetitions(self, video_data: Dict) -> List[Dict]:
        """Detect V cuts in video data"""
        repetitions = []
        
        # Extract necessary data
        ball_detections = video_data.get("ball_detections", [])
        foot_positions = video_data.get("foot_positions", {})
        frame_height = video_data.get("frame_height", 720)
        
        # Process each frame
        for detection in ball_detections:
            if not self._is_valid_detection(detection, foot_positions, frame_height):
                continue
                
            # Analyze V cut motion
            frame_feet = foot_positions.get(detection["frame_number"], [])
            cut_analysis = self._analyze_v_cut(detection, frame_feet)
            
            if cut_analysis:
                foot = cut_analysis["foot"]
                phase = cut_analysis["phase"]
                
                # Track V cut completion
                v_cut = self._process_cut_phase(cut_analysis, detection)
                if v_cut:
                    repetitions.append(v_cut)
                
        return repetitions
    
    def validate_movement(self, movement_data: Dict) -> bool:
        """Validate if movement matches V cuts pattern"""
        ball_pos = movement_data.get("ball_position")
        foot_positions = movement_data.get("foot_positions", [])
        frame_height = movement_data.get("frame_height", 720)
        
        if not ball_pos or len(foot_positions) < 2:
            return False
            
        # Ball should be at foot level
        if not self._is_foot_level(ball_pos[1], frame_height):
            return False
            
        # Ball should be near one foot (for sole or inside contact)
        foot_analysis = self._analyze_foot_contact(ball_pos, foot_positions)
        return foot_analysis is not None
    
    def _is_valid_detection(self, detection: Dict, foot_positions: Dict, frame_height: int) -> bool:
        """Check if detection is valid for V cuts"""
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
        foot_level_min = frame_height * 0.5  # 50% down from top
        foot_level_max = frame_height * 0.9  # 90% down from top
        return foot_level_min <= ball_y <= foot_level_max
    
    def _analyze_foot_contact(self, ball_pos: Tuple[int, int], 
                             foot_positions: List[Tuple[int, int]]) -> Optional[Dict]:
        """Analyze which foot is in contact with ball"""
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
        if left_distance < self.touch_threshold and left_distance < right_distance:
            return {
                "foot": "left",
                "distance": left_distance,
                "foot_pos": (left_foot_x, left_foot_y)
            }
        elif right_distance < self.touch_threshold and right_distance < left_distance:
            return {
                "foot": "right", 
                "distance": right_distance,
                "foot_pos": (right_foot_x, right_foot_y)
            }
            
        return None
    
    def _analyze_v_cut(self, detection: Dict, foot_positions: List[Tuple[int, int]]) -> Optional[Dict]:
        """Analyze V cut motion and determine phase"""
        foot_contact = self._analyze_foot_contact(detection["position"], foot_positions)
        if not foot_contact:
            return None
            
        foot = foot_contact["foot"]
        current_pos = detection["position"]
        foot_pos = foot_contact["foot_pos"]
        
        # Determine cut phase and direction based on ball position relative to foot
        phase = self._determine_cut_phase(current_pos, foot_pos, foot)
        if not phase:
            return None
            
        # Calculate movement distance if we have previous position
        movement_distance = 0
        if foot in self.last_positions:
            last_pos = self.last_positions[foot]
            dx = current_pos[0] - last_pos[0]
            dy = current_pos[1] - last_pos[1]
            movement_distance = math.sqrt(dx**2 + dy**2)
        
        return {
            "foot": foot,
            "phase": phase,
            "distance": movement_distance,
            "foot_contact": foot_contact,
            "timestamp": detection["timestamp"],
            "frame_number": detection["frame_number"],
            "position": current_pos,
            "confidence": detection["confidence"]
        }
    
    def _determine_cut_phase(self, ball_pos: Tuple[int, int], foot_pos: Tuple[int, int], foot: str) -> Optional[str]:
        """Determine if this is pull or push phase of V cut"""
        ball_x, ball_y = ball_pos
        foot_x, foot_y = foot_pos
        
        # Check ball position relative to foot
        relative_y = ball_y - foot_y
        relative_x = ball_x - foot_x
        
        # Pull phase: ball is behind/under the foot (sole contact)
        if abs(relative_x) < 30 and relative_y >= -10:  # Ball close to foot horizontally, at or behind
            return "pull"
        
        # Push phase: ball is in front of foot (inside contact)
        if relative_y < -15:  # Ball clearly in front of foot
            if foot == "left" and relative_x > -20:  # Ball to inside of left foot
                return "push"
            elif foot == "right" and relative_x < 20:  # Ball to inside of right foot
                return "push"
        
        return None
    
    def _process_cut_phase(self, cut_analysis: Dict, detection: Dict) -> Optional[Dict]:
        """Process cut phase and detect complete V cuts"""
        foot = cut_analysis["foot"]
        phase = cut_analysis["phase"]
        
        # Update position tracking
        self.last_positions[foot] = cut_analysis["position"]
        
        # Initialize foot tracking if needed
        if foot not in self.pending_cuts:
            self.pending_cuts[foot] = None
            self.cut_phase[foot] = None
        
        # V cut logic: pull -> push = complete cut
        if phase == "pull":
            # Start of potential V cut
            if cut_analysis["distance"] >= self.min_pull_distance:
                self.pending_cuts[foot] = {
                    "pull_timestamp": cut_analysis["timestamp"],
                    "pull_frame": cut_analysis["frame_number"],
                    "pull_position": cut_analysis["position"]
                }
                self.cut_phase[foot] = "pull"
            return None
            
        elif phase == "push" and self.cut_phase[foot] == "pull":
            # Complete V cut (pull -> push)
            if cut_analysis["distance"] >= self.min_push_distance and self.pending_cuts[foot]:
                pending = self.pending_cuts[foot]
                
                # Create complete V cut record
                v_cut = {
                    "timestamp": pending["pull_timestamp"],  # Use pull timestamp as start
                    "frame_number": pending["pull_frame"],
                    "position": pending["pull_position"],
                    "foot_used": foot,
                    "phase": "complete",  # Mark as complete V cut
                    "pull_timestamp": pending["pull_timestamp"],
                    "push_timestamp": cut_analysis["timestamp"],
                    "confidence": cut_analysis["confidence"],
                    "type": "v_cut"
                }
                
                # Reset tracking for this foot
                self.pending_cuts[foot] = None
                self.cut_phase[foot] = None
                
                return v_cut
        
        return None
    
    def calculate_range(self, count: int, confidence: float) -> Dict:
        """Calculate range for V cuts"""
        # V cuts require complex motion, higher uncertainty
        base_uncertainty = 2
        
        if confidence >= 0.8:
            confidence_factor = 0.8
        elif confidence >= 0.6:
            confidence_factor = 1.2
        else:
            confidence_factor = 1.6
            
        uncertainty = int(base_uncertainty * confidence_factor)
        uncertainty = max(1, min(uncertainty, 3))
        
        range_min = max(0, count - uncertainty)
        range_max = count + uncertainty
        
        return {
            "min": range_min,
            "max": range_max,
            "display": f"{range_min}-{range_max} V cuts",
            "detected_count": count,
            "confidence_level": "high" if confidence >= 0.7 else "medium" if confidence >= 0.5 else "low"
        }
    
    def _get_unit_name(self, count: int) -> str:
        """Get unit name for V cuts"""
        return "V cuts" if count != 1 else "V cut"
    
    def _count_per_foot(self, repetitions: List[Dict]) -> Dict[str, int]:
        """Count V cuts per foot with timing analysis"""
        counts = {"left": 0, "right": 0}
        foot_timings = {"left": [], "right": []}
        
        # Count cuts and track timing
        for rep in repetitions:
            foot = rep.get("foot_used", "").lower()
            
            if foot in counts:
                counts[foot] += 1
                
                # Calculate cut duration (pull to push)
                pull_time = rep.get("pull_timestamp", 0)
                push_time = rep.get("push_timestamp", 0)
                if push_time > pull_time:
                    cut_duration = push_time - pull_time
                    foot_timings[foot].append(cut_duration)
        
        # Analyze cut timing for each foot
        for foot in ["left", "right"]:
            timings = foot_timings[foot]
            if timings:
                avg_duration = sum(timings) / len(timings)
                counts[f"{foot}_avg_duration"] = round(avg_duration, 2)
                counts[f"{foot}_timing_quality"] = self._rate_timing_quality(avg_duration)
        
        return counts
    
    def _rate_timing_quality(self, avg_duration: float) -> str:
        """Rate the timing quality of V cuts"""
        if avg_duration <= 0.8:
            return "excellent"  # Quick, crisp cuts
        elif avg_duration <= 1.2:
            return "good"
        elif avg_duration <= 1.8:
            return "fair"
        else:
            return "needs improvement"  # Too slow


# Register the analyzer
drill_registry.register_analyzer(DrillType.V_CUTS, VCutsAnalyzer)