"""
Drill Analysis Error Taxonomy
Specific error types for better error handling and user experience
"""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels for frontend handling"""
    RETRYABLE = "retryable"      # User can retry with same video
    USER_ACTION = "user_action"   # User needs to fix something
    SYSTEM_ERROR = "system_error" # Internal system issue


class DrillAnalysisError(Exception):
    """Base class for all drill analysis errors"""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.SYSTEM_ERROR, 
                 error_code: str = "UNKNOWN", user_message: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.error_code = error_code
        self.user_message = user_message or message
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to API response format"""
        return {
            "error": True,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "message": self.message,
            "user_message": self.user_message,
            "metadata": self.metadata
        }


class VideoQualityError(DrillAnalysisError):
    """Video quality issues that prevent analysis"""
    
    def __init__(self, issue: str, detection_confidence: Optional[float] = None,
                 camera_angle: Optional[float] = None):
        metadata = {}
        if detection_confidence is not None:
            metadata["detection_confidence"] = detection_confidence
        if camera_angle is not None:
            metadata["camera_angle"] = camera_angle
        
        user_messages = {
            "lighting": "Video is too dark or poorly lit. Try recording in better lighting conditions.",
            "angle": "Camera angle is too steep. Hold phone more horizontally when recording.",
            "distance": "Camera is too far from the action. Move closer to capture foot and ball movements clearly.",
            "blur": "Video is too blurry or shaky. Use a steady hand or tripod when recording.",
            "resolution": "Video resolution is too low. Record in at least 720p HD quality."
        }
        
        user_message = user_messages.get(issue, f"Video quality issue: {issue}")
        
        super().__init__(
            message=f"Video quality insufficient: {issue}",
            severity=ErrorSeverity.USER_ACTION,
            error_code=f"VIDEO_QUALITY_{issue.upper()}",
            user_message=user_message,
            metadata=metadata
        )


class InsufficientDataError(DrillAnalysisError):
    """Not enough data in video for analysis"""
    
    def __init__(self, issue: str, required: Optional[Any] = None, 
                 detected: Optional[Any] = None):
        metadata = {}
        if required is not None:
            metadata["required"] = required
        if detected is not None:
            metadata["detected"] = detected
        
        user_messages = {
            "duration": f"Video is too short. Need at least {required} seconds for this drill.",
            "ball_detection": "Ball not visible in video. Make sure the ball is clearly visible throughout the drill.",
            "foot_detection": "Feet not visible in video. Make sure your feet and the ball are in frame.",
            "movement": "Not enough movement detected. Perform the drill with clear, deliberate movements."
        }
        
        user_message = user_messages.get(issue, f"Insufficient data: {issue}")
        
        super().__init__(
            message=f"Insufficient data for analysis: {issue}",
            severity=ErrorSeverity.USER_ACTION,
            error_code=f"INSUFFICIENT_{issue.upper()}",
            user_message=user_message,
            metadata=metadata
        )


class DrillValidationError(DrillAnalysisError):
    """Movement doesn't match expected drill pattern"""
    
    def __init__(self, drill_type: str, issue: str, expected_pattern: str):
        user_message = f"Movement doesn't match {drill_type} pattern. {expected_pattern}"
        
        super().__init__(
            message=f"{drill_type} validation failed: {issue}",
            severity=ErrorSeverity.USER_ACTION,
            error_code=f"DRILL_VALIDATION_{drill_type.upper()}",
            user_message=user_message,
            metadata={"drill_type": drill_type, "expected_pattern": expected_pattern}
        )


class ProcessingTimeoutError(DrillAnalysisError):
    """Analysis took too long"""
    
    def __init__(self, duration: float, timeout: float):
        super().__init__(
            message=f"Analysis timeout: {duration:.1f}s exceeded {timeout:.1f}s limit",
            severity=ErrorSeverity.RETRYABLE,
            error_code="PROCESSING_TIMEOUT",
            user_message="Analysis is taking longer than expected. Please try again.",
            metadata={"duration": duration, "timeout": timeout}
        )


class ModelLoadError(DrillAnalysisError):
    """AI model loading issues"""
    
    def __init__(self, model_name: str, details: str):
        super().__init__(
            message=f"Failed to load {model_name} model: {details}",
            severity=ErrorSeverity.SYSTEM_ERROR,
            error_code="MODEL_LOAD_ERROR",
            user_message="System temporarily unavailable. Please try again in a few minutes.",
            metadata={"model_name": model_name, "details": details}
        )


class ConfigurationError(DrillAnalysisError):
    """Drill configuration issues"""
    
    def __init__(self, drill_type: str, issue: str):
        super().__init__(
            message=f"Configuration error for {drill_type}: {issue}",
            severity=ErrorSeverity.SYSTEM_ERROR,
            error_code="CONFIGURATION_ERROR",
            user_message="System configuration issue. Please contact support.",
            metadata={"drill_type": drill_type, "issue": issue}
        )


# Error detection helpers
def check_video_quality(video_data: Dict) -> None:
    """Check video quality and raise appropriate errors"""
    
    # Check detection confidence
    ball_detections = video_data.get("ball_detections", [])
    if ball_detections:
        avg_confidence = sum(d.get("confidence", 0) for d in ball_detections) / len(ball_detections)
        if avg_confidence < 0.3:
            raise VideoQualityError("lighting", detection_confidence=avg_confidence)
    
    # Check duration
    duration = video_data.get("duration", 0)
    if duration < 10:
        raise InsufficientDataError("duration", required=15, detected=duration)
    
    # Check if ball detected at all
    if not ball_detections:
        raise InsufficientDataError("ball_detection")


def check_drill_requirements(drill_type: str, video_data: Dict) -> None:
    """Check drill-specific requirements"""
    
    duration = video_data.get("duration", 0)
    
    # Time-based requirements
    min_durations = {
        "bell_touches": 15,
        "sole_rolls": 15,
        "outside_foot_push": 15,
        "croquetas": 10,
        "triangles": 15,
        "v_cuts": 15
    }
    
    if drill_type in min_durations and duration < min_durations[drill_type]:
        raise InsufficientDataError("duration", 
                                   required=min_durations[drill_type], 
                                   detected=duration)


# Usage examples for your analyzers:
"""
# In your analyzer.analyze() method:
def analyze(self, video_data: Dict) -> DrillResults:
    try:
        # Check video quality first
        check_video_quality(video_data)
        check_drill_requirements(self.config.drill_type.value, video_data)
        
        # Your existing analysis logic...
        repetitions = self.detect_repetitions(video_data)
        
        # Validate drill pattern
        if not repetitions and video_data.get("duration", 0) > 20:
            raise DrillValidationError(
                drill_type=self.config.name,
                issue="no valid movements detected",
                expected_pattern=self.config.success_criteria
            )
        
        # Continue with analysis...
        
    except DrillAnalysisError:
        raise  # Re-raise our specific errors
    except Exception as e:
        # Convert unexpected errors to system errors
        raise DrillAnalysisError(
            message=f"Unexpected analysis error: {str(e)}",
            severity=ErrorSeverity.SYSTEM_ERROR,
            error_code="UNEXPECTED_ERROR",
            user_message="Something went wrong during analysis. Please try again."
        )
"""