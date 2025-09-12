"""
Self-registering drill analyzer decorator
Eliminates dual registration and keeps everything in one place
"""

from functools import wraps
from typing import Type, Optional
from drill_analyzer import DrillAnalyzer, DrillConfig, DrillType, drill_registry


def register_drill_analyzer(config: DrillConfig):
    """
    Decorator that automatically registers both config and analyzer
    Usage:
    
    @register_drill_analyzer(DrillConfig(
        drill_type=DrillType.BELL_TOUCHES,
        name="Bell Touches",
        success_criteria="18-24 touches in 30 seconds",
        # ... rest of config
    ))
    class BellTouchesAnalyzer(DrillAnalyzer):
        pass
    """
    def decorator(analyzer_class: Type[DrillAnalyzer]):
        # Validate that analyzer class is properly structured
        if not issubclass(analyzer_class, DrillAnalyzer):
            raise TypeError(f"{analyzer_class.__name__} must inherit from DrillAnalyzer")
        
        # Register both config and analyzer atomically
        drill_registry.register_drill(config)
        drill_registry.register_analyzer(config.drill_type, analyzer_class)
        
        # Add config as class attribute for easy access
        analyzer_class.config = config
        
        print(f"✅ Registered {config.drill_type.value}: {analyzer_class.__name__}")
        
        return analyzer_class
    
    return decorator


def validate_analyzer_registration():
    """
    Validate that all registered drills have both config and analyzer
    Call this during startup to catch registration issues early
    """
    errors = []
    
    for drill_type in DrillType:
        config = drill_registry.get_config(drill_type)
        analyzer = drill_registry.get_analyzer(drill_type)
        
        if config and not analyzer:
            errors.append(f"❌ {drill_type.value}: Has config but no analyzer")
        elif analyzer and not config:
            errors.append(f"❌ {drill_type.value}: Has analyzer but no config")
        elif not config and not analyzer:
            errors.append(f"⚠️  {drill_type.value}: Neither config nor analyzer registered")
    
    if errors:
        error_msg = "Drill registration validation failed:\n" + "\n".join(errors)
        raise RuntimeError(error_msg)
    
    print(f"✅ All {len(list(DrillType))} drills properly registered")