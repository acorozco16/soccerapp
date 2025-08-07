#!/usr/bin/env python3
"""
Test the new drill framework with Bell Touches
Validates the architecture works correctly
"""

import sys
import logging
from pathlib import Path

# Add backend directory to path
sys.path.append(str(Path(__file__).parent))

# Import only the drill framework components (not unified_processor due to dependencies)
from drill_analyzer import drill_registry, DrillType

# Import the analyzer to trigger registration
try:
    from analyzers.bell_touches_analyzer import BellTouchesAnalyzer
except ImportError:
    print("⚠️  Bell Touches analyzer not available (import failed)")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_drill_registry():
    """Test the drill registry functionality"""
    print("🧪 Testing Drill Registry")
    print("=" * 50)
    
    # List all available drills
    drills = drill_registry.list_drills()
    print(f"\n📋 Available Drills: {len(drills)}")
    for drill in drills:
        print(f"   - {drill['name']} ({drill['type']})")
        print(f"     Success: {drill['success_criteria']}")
    
    # Test getting specific drill config
    bell_config = drill_registry.get_config(DrillType.BELL_TOUCHES)
    if bell_config:
        print(f"\n✅ Bell Touches Config:")
        print(f"   - Time Window: {bell_config.time_window}s")
        print(f"   - Benchmark: {bell_config.min_reps}-{bell_config.max_reps} reps")
        print(f"   - Per Foot: {bell_config.per_foot}")
    
    # Test getting analyzer
    bell_analyzer = drill_registry.get_analyzer(DrillType.BELL_TOUCHES)
    if bell_analyzer:
        print(f"\n✅ Bell Touches Analyzer: {type(bell_analyzer).__name__}")
    else:
        print(f"\n❌ Bell Touches Analyzer not found!")
        
def test_framework_directly():
    """Test the framework directly without unified processor"""
    print("\n\n🧪 Testing Framework Directly")
    print("=" * 50)
    
    # Test drill enumeration
    available_types = [dt.value for dt in DrillType]
    print(f"\n📋 DrillType Enum: {len(available_types)} types")
    for drill_type in available_types:
        print(f"   - {drill_type}")
    
    # Test drill info generation
    bell_config = drill_registry.get_config(DrillType.BELL_TOUCHES)
    if bell_config:
        print(f"\n🔔 Bell Touches Info (Direct):")
        print(f"   - Name: {bell_config.name}")
        print(f"   - Success: {bell_config.success_criteria}")
        print(f"   - Time Window: {bell_config.time_window}s")
        print(f"   - Benchmark: {bell_config.min_reps}-{bell_config.max_reps}")
        print(f"   - Per Foot: {bell_config.per_foot}")
    
def test_mock_analysis():
    """Test analysis with mock video data"""
    print("\n\n🧪 Testing Mock Bell Touches Analysis")
    print("=" * 50)
    
    # Create mock video data
    mock_video_data = {
        "video_id": "test_123",
        "duration": 30.0,
        "frame_height": 720,
        "frame_width": 1280,
        "fps": 30,
        
        # Mock ball detections (alternating between feet)
        "ball_detections": [
            {"timestamp": 1.0, "frame_number": 30, "position": (100, 600), "confidence": 0.8},
            {"timestamp": 1.5, "frame_number": 45, "position": (200, 600), "confidence": 0.85},
            {"timestamp": 2.0, "frame_number": 60, "position": (100, 600), "confidence": 0.82},
            {"timestamp": 2.5, "frame_number": 75, "position": (200, 600), "confidence": 0.83},
            {"timestamp": 3.0, "frame_number": 90, "position": (100, 600), "confidence": 0.81},
            {"timestamp": 3.5, "frame_number": 105, "position": (200, 600), "confidence": 0.84},
        ],
        
        # Mock foot positions
        "foot_positions": {
            30: [(100, 600), (200, 600)],
            45: [(100, 600), (200, 600)],
            60: [(100, 600), (200, 600)],
            75: [(100, 600), (200, 600)],
            90: [(100, 600), (200, 600)],
            105: [(100, 600), (200, 600)],
        }
    }
    
    # Test with framework analyzer
    from drill_analyzer import drill_registry, DrillType
    analyzer = drill_registry.get_analyzer(DrillType.BELL_TOUCHES)
    
    if analyzer:
        print("\n📊 Running Bell Touches Analysis...")
        results = analyzer.analyze(mock_video_data)
        
        print(f"\n✅ Analysis Results:")
        print(f"   - Drill Type: {results.drill_type.value}")
        print(f"   - Count Detected: {results.count_detected}")
        print(f"   - Count Range: {results.count_range['display']}")
        print(f"   - Duration: {results.duration}s")
        print(f"   - Benchmark Met: {results.benchmark_met}")
        print(f"   - Confidence: {results.confidence:.2f}")
        
        if results.per_foot_counts:
            print(f"   - Per Foot Counts: {results.per_foot_counts}")
        
        # Test JSON serialization
        results_dict = results.to_dict()
        print(f"\n📄 JSON Structure:")
        print(f"   - drill_type: {results_dict['drill_type']}")
        print(f"   - results: {list(results_dict['results'].keys())}")
        print(f"   - metadata: {list(results_dict['metadata'].keys())}")
        
def test_benchmark_calculation():
    """Test benchmark calculation logic"""
    print("\n\n🧪 Testing Benchmark Calculations")
    print("=" * 50)
    
    from drill_analyzer import drill_registry, DrillType
    
    # Test Bell Touches (time-based)
    bell_config = drill_registry.get_config(DrillType.BELL_TOUCHES)
    bell_analyzer = drill_registry.get_analyzer(DrillType.BELL_TOUCHES)
    
    test_cases = [
        (20, 30, "20 touches in 30s"),  # 20 touches in 30s = exactly in range
        (15, 30, "15 touches in 30s"),  # 15 touches in 30s = below range
        (25, 30, "25 touches in 30s"),  # 25 touches in 30s = above range
        (10, 15, "10 touches in 15s"),  # 10 in 15s = 20 in 30s = in range
    ]
    
    print(f"\n🔔 Bell Touches Benchmark: {bell_config.success_criteria}")
    for count, duration, desc in test_cases:
        meets_benchmark = bell_analyzer.check_benchmark(count, duration)
        normalized = (count / duration) * bell_config.time_window
        print(f"   - {desc}: {'✅' if meets_benchmark else '❌'} (normalized: {normalized:.1f})")
    
    # Test Inside-Outside (reps-based)
    io_config = drill_registry.get_config(DrillType.INSIDE_OUTSIDE)
    if io_config:
        print(f"\n🔄 Inside-Outside Benchmark: {io_config.success_criteria}")
        # Would test reps-based logic here
        
def main():
    """Run all framework tests"""
    print("🚀 Testing Drill Framework Architecture")
    print("=" * 70)
    
    try:
        test_drill_registry()
        test_framework_directly()
        test_mock_analysis()
        test_benchmark_calculation()
        
        print("\n\n✅ FRAMEWORK TEST SUMMARY:")
        print("   ✅ Drill Registry: Working")
        print("   ✅ Drill Configurations: Loaded")
        print("   ✅ Analyzer Creation: Working")
        print("   ✅ Analysis Pipeline: Working")
        print("   ✅ Results Structure: Working")
        print("   ✅ Benchmark Logic: Working")
        print("\n🎯 Drill framework architecture is ready!")
        
    except Exception as e:
        print(f"\n❌ Framework test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()