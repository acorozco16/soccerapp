"""
Ball Mastery Drill Analysis Framework
Foundation for all 8 drill types with consistent architecture
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DrillType(Enum):
    """Enum for all supported drill types"""
    JUGGLING = "juggling"
    BELL_TOUCHES = "bell_touches"
    INSIDE_OUTSIDE = "inside_outside"
    SOLE_ROLLS = "sole_rolls"
    OUTSIDE_FOOT_PUSH = "outside_foot_push"
    V_CUTS = "v_cuts"
    CROQUETAS = "croquetas"
    TRIANGLES = "triangles"


@dataclass
class DrillConfig:
    """Configuration for each drill type"""
    drill_type: DrillType
    name: str
    description: str
    success_criteria: str
    time_window: Optional[float]  # seconds (e.g., 30 for "in 30 seconds")
    min_reps: int
    max_reps: int
    per_foot: bool  # True if we track each foot separately
    pattern_based: bool  # True for drills like triangles
    

@dataclass
class DrillResults:
    """Unified results structure for all drills"""
    drill_type: DrillType
    success_criteria: str
    
    # Core metrics
    count_detected: int
    count_range: Dict[str, Any]  # min, max, display
    duration: float
    benchmark_met: bool
    confidence: float
    
    # Optional metrics (drill-specific)
    per_foot_counts: Optional[Dict[str, int]] = None
    pattern_count: Optional[int] = None
    
    # Metadata
    video_id: str = ""
    timestamp: str = ""
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "drill_type": self.drill_type.value,
            "success_criteria": self.success_criteria,
            "results": {
                "count_detected": self.count_detected,
                "count_range": self.count_range,
                "duration": round(self.duration, 1),
                "benchmark_met": self.benchmark_met,
                "confidence": round(self.confidence, 2)
            },
            "per_foot_counts": self.per_foot_counts,
            "pattern_count": self.pattern_count,
            "metadata": {
                "video_id": self.video_id,
                "timestamp": self.timestamp,
                "processing_time": round(self.processing_time, 1)
            }
        }


class DrillAnalyzer(ABC):
    """Base class for all drill analyzers"""
    
    def __init__(self, config: DrillConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.drill_type.value}")
        
    @abstractmethod
    def detect_repetitions(self, video_data: Dict) -> List[Dict]:
        """Detect drill-specific repetitions in video data"""
        pass
        
    @abstractmethod
    def validate_movement(self, movement_data: Dict) -> bool:
        """Validate if movement matches drill pattern"""
        pass
    
    def calculate_range(self, count: int, confidence: float) -> Dict:
        """Calculate count range based on confidence"""
        # Base uncertainty calculation (can be overridden)
        base_uncertainty = 2
        
        if confidence >= 0.8:
            confidence_factor = 0.5
        elif confidence >= 0.6:
            confidence_factor = 1.0
        else:
            confidence_factor = 1.5
            
        uncertainty = int(base_uncertainty * confidence_factor)
        uncertainty = max(1, min(uncertainty, 3))
        
        range_min = max(0, count - uncertainty)
        range_max = count + uncertainty
        
        # Special handling for very low counts
        if count <= 3:
            range_min = max(0, count - 1)
            range_max = count + 2
            
        return {
            "min": range_min,
            "max": range_max,
            "display": f"{range_min}-{range_max} {self._get_unit_name(count)}",
            "detected_count": count,
            "confidence_level": "high" if confidence >= 0.7 else "medium" if confidence >= 0.5 else "low"
        }
    
    def check_benchmark(self, count: int, duration: float) -> bool:
        """Check if performance meets benchmark criteria"""
        # Handle time-based criteria
        if self.config.time_window:
            # Normalize to time window (e.g., touches per 30s)
            normalized_count = (count / duration) * self.config.time_window
            return self.config.min_reps <= normalized_count <= self.config.max_reps
        else:
            # Simple count-based criteria
            return self.config.min_reps <= count <= self.config.max_reps
    
    def analyze(self, video_data: Dict) -> DrillResults:
        """Main analysis method - orchestrates drill analysis"""
        try:
            # Extract common data
            duration = video_data.get("duration", 0)
            video_id = video_data.get("video_id", "")
            
            # Detect repetitions
            repetitions = self.detect_repetitions(video_data)
            
            # Calculate metrics
            count = len(repetitions)
            confidence = self._calculate_confidence(repetitions)
            count_range = self.calculate_range(count, confidence)
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
            
            # Add drill-specific data
            if self.config.per_foot:
                results.per_foot_counts = self._count_per_foot(repetitions)
                
            if self.config.pattern_based:
                results.pattern_count = self._count_patterns(repetitions)
                
            self.logger.info(f"Analysis complete: {count} {self._get_unit_name(count)} detected")
            return results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise
            
    def _calculate_confidence(self, repetitions: List[Dict]) -> float:
        """Calculate average confidence from detections"""
        if not repetitions:
            return 0.0
        return sum(r.get("confidence", 0.5) for r in repetitions) / len(repetitions)
    
    def _count_per_foot(self, repetitions: List[Dict]) -> Dict[str, int]:
        """Count repetitions per foot"""
        counts = {"left": 0, "right": 0}
        for rep in repetitions:
            foot = rep.get("foot_used", "").lower()
            if foot in counts:
                counts[foot] += 1
        return counts
    
    def _count_patterns(self, repetitions: List[Dict]) -> int:
        """Count completed patterns (for pattern-based drills)"""
        # Override in pattern-based drills
        return 0
    
    def _get_unit_name(self, count: int) -> str:
        """Get the unit name for display (singular/plural)"""
        # Override for drill-specific naming
        return "reps" if count != 1 else "rep"


class DrillRegistry:
    """Registry for all available drills and their configurations"""
    
    def __init__(self):
        self.drills: Dict[DrillType, DrillConfig] = {}
        self.analyzers: Dict[DrillType, type] = {}
        self._register_default_drills()
        
    def _register_default_drills(self):
        """Register all 8 ball mastery drills"""
        
        # Juggling (already implemented separately)
        self.register_drill(DrillConfig(
            drill_type=DrillType.JUGGLING,
            name="Juggling (Keep-ups)",
            description="Keep the ball in the air using feet, thighs, and head",
            success_criteria="Maintain control for 30-60 seconds",
            time_window=None,
            min_reps=10,
            max_reps=100,
            per_foot=False,
            pattern_based=False
        ))
        
        # Bell Touches
        self.register_drill(DrillConfig(
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
        
        # Inside-Outside Touches
        self.register_drill(DrillConfig(
            drill_type=DrillType.INSIDE_OUTSIDE,
            name="Inside-Outside Touches",
            description="Alternating inside and outside touches with same foot",
            success_criteria="12-18 reps per foot per set",
            time_window=None,
            min_reps=12,
            max_reps=18,
            per_foot=True,
            pattern_based=False
        ))
        
        # Sole Rolls
        self.register_drill(DrillConfig(
            drill_type=DrillType.SOLE_ROLLS,
            name="Sole Rolls",
            description="Rolling ball back and forth with sole of foot",
            success_criteria="8-14 smooth rolls in 20-30 seconds",
            time_window=25.0,  # Average of 20-30
            min_reps=8,
            max_reps=14,
            per_foot=True,
            pattern_based=False
        ))
        
        # Outside Foot Push
        self.register_drill(DrillConfig(
            drill_type=DrillType.OUTSIDE_FOOT_PUSH,
            name="Outside Foot Push",
            description="Pushing ball with outside of foot repeatedly",
            success_criteria="15-22 touches in 30 seconds",
            time_window=30.0,
            min_reps=15,
            max_reps=22,
            per_foot=True,
            pattern_based=False
        ))
        
        # V Cuts
        self.register_drill(DrillConfig(
            drill_type=DrillType.V_CUTS,
            name="V Cuts (Pull-Push)",
            description="Pull ball back with sole, push forward with inside",
            success_criteria="6-10 cuts per foot in 20-30 seconds",
            time_window=25.0,
            min_reps=6,
            max_reps=10,
            per_foot=True,
            pattern_based=False
        ))
        
        # Croquetas
        self.register_drill(DrillConfig(
            drill_type=DrillType.CROQUETAS,
            name="Croquetas",
            description="Side-to-side cutting movements with sole",
            success_criteria="8-15 smooth cuts in 15-30 seconds",
            time_window=22.5,  # Average of 15-30
            min_reps=8,
            max_reps=15,
            per_foot=False,
            pattern_based=False
        ))
        
        # Triangles
        self.register_drill(DrillConfig(
            drill_type=DrillType.TRIANGLES,
            name="Triangles",
            description="Move ball in triangle pattern with different surfaces",
            success_criteria="4-8 full patterns in 20-30 seconds",
            time_window=25.0,
            min_reps=4,
            max_reps=8,
            per_foot=False,
            pattern_based=True
        ))
        
    def register_drill(self, config: DrillConfig):
        """Register a drill configuration"""
        self.drills[config.drill_type] = config
        
    def register_analyzer(self, drill_type: DrillType, analyzer_class: type):
        """Register an analyzer class for a drill type"""
        self.analyzers[drill_type] = analyzer_class
        
    def get_config(self, drill_type: DrillType) -> Optional[DrillConfig]:
        """Get configuration for a drill type"""
        return self.drills.get(drill_type)
        
    def get_analyzer(self, drill_type: DrillType) -> Optional[DrillAnalyzer]:
        """Get analyzer instance for a drill type"""
        config = self.get_config(drill_type)
        analyzer_class = self.analyzers.get(drill_type)
        
        if config and analyzer_class:
            return analyzer_class(config)
        return None
        
    def list_drills(self) -> List[Dict]:
        """List all available drills"""
        return [
            {
                "type": drill_type.value,
                "name": config.name,
                "description": config.description,
                "success_criteria": config.success_criteria
            }
            for drill_type, config in self.drills.items()
        ]


# Global registry instance
drill_registry = DrillRegistry()