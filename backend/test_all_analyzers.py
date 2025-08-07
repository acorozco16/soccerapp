#!/usr/bin/env python3
"""
Comprehensive test for all drill analyzers
Tests the complete framework with all 8 drill types
"""

import sys
import logging
from pathlib import Path

# Add backend directory to path
sys.path.append(str(Path(__file__).parent))

from drill_analyzer import drill_registry, DrillType

# Import all analyzers to trigger registration
try:
    from analyzers.bell_touches_analyzer import BellTouchesAnalyzer
    from analyzers.inside_outside_analyzer import InsideOutsideAnalyzer
    from analyzers.sole_rolls_analyzer import SoleRollsAnalyzer
    from analyzers.outside_foot_push_analyzer import OutsideFootPushAnalyzer
    from analyzers.v_cuts_analyzer import VCutsAnalyzer
    from analyzers.croquetas_analyzer import CroquetasAnalyzer
    from analyzers.triangles_analyzer import TrianglesAnalyzer
    print("✅ All analyzers imported successfully")
except ImportError as e:
    print(f"⚠️  Some analyzers not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_all_drill_registrations():
    """Test that all drills are properly registered"""
    print("🧪 Testing All Drill Registrations")
    print("=" * 60)
    
    expected_drills = [
        DrillType.JUGGLING,
        DrillType.BELL_TOUCHES,
        DrillType.INSIDE_OUTSIDE,
        DrillType.SOLE_ROLLS,
        DrillType.OUTSIDE_FOOT_PUSH,
        DrillType.V_CUTS,
        DrillType.CROQUETAS,
        DrillType.TRIANGLES
    ]
    
    print(f"\n📋 Expected Drills: {len(expected_drills)}")
    
    # Test drill configs
    missing_configs = []
    for drill_type in expected_drills:
        config = drill_registry.get_config(drill_type)
        if config:
            print(f"   ✅ {drill_type.value}: {config.name}")
        else:
            print(f"   ❌ {drill_type.value}: Missing config")
            missing_configs.append(drill_type)
    
    # Test analyzer registrations
    missing_analyzers = []
    for drill_type in expected_drills:
        analyzer = drill_registry.get_analyzer(drill_type)
        if analyzer:
            print(f"   🔧 {drill_type.value}: {type(analyzer).__name__}")
        else:
            print(f"   ❌ {drill_type.value}: Missing analyzer")
            missing_analyzers.append(drill_type)
    
    return missing_configs, missing_analyzers

def create_mock_video_data(drill_type: DrillType) -> dict:
    """Create mock video data tailored for each drill type"""
    base_data = {
        "video_id": f"test_{drill_type.value}",
        "duration": 30.0,
        "frame_height": 720,
        "frame_width": 1280,
        "fps": 30,
        "foot_positions": {}
    }
    
    # Drill-specific mock data
    if drill_type == DrillType.BELL_TOUCHES:
        # Alternating touches between feet at ground level
        base_data["ball_detections"] = [
            {"timestamp": 1.0, "frame_number": 30, "position": (100, 600), "confidence": 0.8},
            {"timestamp": 1.5, "frame_number": 45, "position": (200, 600), "confidence": 0.85},
            {"timestamp": 2.0, "frame_number": 60, "position": (100, 600), "confidence": 0.82},
            {"timestamp": 2.5, "frame_number": 75, "position": (200, 600), "confidence": 0.83},
        ]
        base_data["foot_positions"] = {30: [(100, 600), (200, 600)], 45: [(100, 600), (200, 600)], 
                                       60: [(100, 600), (200, 600)], 75: [(100, 600), (200, 600)]}
    
    elif drill_type == DrillType.INSIDE_OUTSIDE:
        # Alternating inside/outside touches with same foot
        base_data["ball_detections"] = [
            {"timestamp": 1.0, "frame_number": 30, "position": (120, 500), "confidence": 0.8},  # Inside left
            {"timestamp": 1.5, "frame_number": 45, "position": (80, 500), "confidence": 0.85},   # Outside left
            {"timestamp": 2.0, "frame_number": 60, "position": (125, 500), "confidence": 0.82},  # Inside left
        ]
        base_data["foot_positions"] = {30: [(100, 500), (200, 500)], 45: [(100, 500), (200, 500)], 
                                       60: [(100, 500), (200, 500)]}
    
    elif drill_type == DrillType.SOLE_ROLLS:
        # Rolling motions at ground level
        base_data["ball_detections"] = [
            {"timestamp": 1.0, "frame_number": 30, "position": (100, 580), "confidence": 0.8},
            {"timestamp": 1.2, "frame_number": 36, "position": (130, 580), "confidence": 0.85},
            {"timestamp": 1.4, "frame_number": 42, "position": (100, 580), "confidence": 0.82},
        ]
        base_data["foot_positions"] = {30: [(100, 580), (200, 580)], 36: [(100, 580), (200, 580)], 
                                       42: [(100, 580), (200, 580)]}
    
    elif drill_type == DrillType.OUTSIDE_FOOT_PUSH:
        # Outside foot pushes
        base_data["ball_detections"] = [
            {"timestamp": 1.0, "frame_number": 30, "position": (80, 500), "confidence": 0.8},   # Outside left
            {"timestamp": 1.5, "frame_number": 45, "position": (220, 500), "confidence": 0.85}, # Outside right
            {"timestamp": 2.0, "frame_number": 60, "position": (75, 500), "confidence": 0.82},  # Outside left
        ]
        base_data["foot_positions"] = {30: [(100, 500), (200, 500)], 45: [(100, 500), (200, 500)], 
                                       60: [(100, 500), (200, 500)]}
    
    elif drill_type == DrillType.V_CUTS:
        # Pull-push movements
        base_data["ball_detections"] = [
            {"timestamp": 1.0, "frame_number": 30, "position": (100, 520), "confidence": 0.8},  # Pull
            {"timestamp": 1.3, "frame_number": 39, "position": (120, 480), "confidence": 0.85}, # Push
            {"timestamp": 2.0, "frame_number": 60, "position": (200, 520), "confidence": 0.82}, # Pull right
        ]
        base_data["foot_positions"] = {30: [(100, 500), (200, 500)], 39: [(100, 500), (200, 500)], 
                                       60: [(100, 500), (200, 500)]}
    
    elif drill_type == DrillType.CROQUETAS:
        # Side-to-side cutting movements
        base_data["ball_detections"] = [
            {"timestamp": 1.0, "frame_number": 30, "position": (120, 580), "confidence": 0.8},
            {"timestamp": 1.4, "frame_number": 42, "position": (180, 580), "confidence": 0.85}, # Cut right
            {"timestamp": 1.8, "frame_number": 54, "position": (140, 580), "confidence": 0.82}, # Cut left
        ]
        base_data["foot_positions"] = {30: [(100, 580), (200, 580)], 42: [(100, 580), (200, 580)], 
                                       54: [(100, 580), (200, 580)]}
    
    elif drill_type == DrillType.TRIANGLES:
        # Triangle pattern movements
        base_data["ball_detections"] = [
            {"timestamp": 1.0, "frame_number": 30, "position": (100, 500), "confidence": 0.8},  # Point 1
            {"timestamp": 1.8, "frame_number": 54, "position": (160, 460), "confidence": 0.85}, # Point 2
            {"timestamp": 2.6, "frame_number": 78, "position": (140, 540), "confidence": 0.82}, # Point 3
            {"timestamp": 3.2, "frame_number": 96, "position": (100, 500), "confidence": 0.83}, # Back to start
        ]
        base_data["foot_positions"] = {30: [(100, 500), (200, 500)], 54: [(100, 500), (200, 500)], 
                                       78: [(100, 500), (200, 500)], 96: [(100, 500), (200, 500)]}
    
    else:  # JUGGLING - simple default
        base_data["ball_detections"] = [
            {"timestamp": 1.0, "frame_number": 30, "position": (150, 300), "confidence": 0.8},
            {"timestamp": 1.5, "frame_number": 45, "position": (150, 200), "confidence": 0.85},
            {"timestamp": 2.0, "frame_number": 60, "position": (150, 300), "confidence": 0.82},
        ]
        base_data["foot_positions"] = {30: [(100, 500), (200, 500)], 45: [(100, 500), (200, 500)], 
                                       60: [(100, 500), (200, 500)]}
    
    return base_data

def test_all_analyzers():
    """Test all analyzers with mock data"""
    print("\n\n🧪 Testing All Analyzers with Mock Data")
    print("=" * 60)
    
    drill_types = [
        DrillType.BELL_TOUCHES,
        DrillType.INSIDE_OUTSIDE,
        DrillType.SOLE_ROLLS,
        DrillType.OUTSIDE_FOOT_PUSH,
        DrillType.V_CUTS,
        DrillType.CROQUETAS,
        DrillType.TRIANGLES
    ]
    
    results_summary = {}
    
    for drill_type in drill_types:
        print(f"\n🔧 Testing {drill_type.value.upper()} Analyzer")
        print("-" * 40)
        
        try:
            # Get analyzer
            analyzer = drill_registry.get_analyzer(drill_type)
            if not analyzer:
                print(f"   ❌ No analyzer found for {drill_type.value}")
                results_summary[drill_type.value] = "FAILED - No analyzer"
                continue
            
            # Create mock data
            mock_data = create_mock_video_data(drill_type)
            
            # Run analysis
            results = analyzer.analyze(mock_data)
            
            # Display results
            print(f"   📊 Count Detected: {results.count_detected}")
            print(f"   📊 Range: {results.count_range['display']}")
            print(f"   📊 Benchmark Met: {results.benchmark_met}")
            print(f"   📊 Confidence: {results.confidence:.2f}")
            
            if results.per_foot_counts:
                print(f"   👣 Per Foot: {results.per_foot_counts}")
            
            if results.pattern_count is not None:
                print(f"   🔄 Patterns: {results.pattern_count}")
            
            # Test JSON serialization
            json_results = results.to_dict()
            print(f"   📄 JSON Keys: {list(json_results.keys())}")
            
            results_summary[drill_type.value] = "PASSED"
            
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
            results_summary[drill_type.value] = f"FAILED - {str(e)[:50]}..."
            import traceback
            traceback.print_exc()
    
    return results_summary

def test_benchmark_logic():
    """Test benchmark calculations for all drills"""
    print("\n\n🧪 Testing Benchmark Logic")
    print("=" * 60)
    
    test_cases = [
        (DrillType.BELL_TOUCHES, [(20, 30, True), (15, 30, False), (25, 30, False)]),
        (DrillType.INSIDE_OUTSIDE, [(15, 60, True), (10, 60, False), (20, 60, False)]),
        (DrillType.SOLE_ROLLS, [(12, 25, True), (6, 25, False), (16, 25, False)]),
        (DrillType.OUTSIDE_FOOT_PUSH, [(18, 30, True), (12, 30, False), (25, 30, False)]),
        (DrillType.V_CUTS, [(8, 25, True), (4, 25, False), (12, 25, False)]),
        (DrillType.CROQUETAS, [(12, 22.5, True), (6, 22.5, False), (18, 22.5, False)]),
        (DrillType.TRIANGLES, [(6, 25, True), (3, 25, False), (10, 25, False)]),
    ]
    
    for drill_type, cases in test_cases:
        config = drill_registry.get_config(drill_type)
        analyzer = drill_registry.get_analyzer(drill_type)
        
        if not config or not analyzer:
            print(f"   ❌ {drill_type.value}: Missing config or analyzer")
            continue
            
        print(f"\n🎯 {drill_type.value.upper()}: {config.success_criteria}")
        
        for count, duration, expected in cases:
            result = analyzer.check_benchmark(count, duration)
            status = "✅" if result == expected else "❌"
            print(f"   {status} {count} in {duration}s: {result}")

def main():
    """Run comprehensive analyzer tests"""
    print("🚀 Comprehensive Drill Analyzer Testing")
    print("=" * 70)
    
    try:
        # Test registrations
        missing_configs, missing_analyzers = test_all_drill_registrations()
        
        # Test analyzers
        results_summary = test_all_analyzers()
        
        # Test benchmarks
        test_benchmark_logic()
        
        # Final summary
        print("\n\n📊 FINAL TEST SUMMARY")
        print("=" * 70)
        
        print(f"\n🔧 Analyzer Registration:")
        if not missing_configs and not missing_analyzers:
            print("   ✅ All 8 drill analyzers properly registered")
        else:
            if missing_configs:
                print(f"   ❌ Missing configs: {[d.value for d in missing_configs]}")
            if missing_analyzers:
                print(f"   ❌ Missing analyzers: {[d.value for d in missing_analyzers]}")
        
        print(f"\n🧪 Analyzer Testing:")
        passed = sum(1 for status in results_summary.values() if status == "PASSED")
        total = len(results_summary)
        print(f"   📊 {passed}/{total} analyzers passed testing")
        
        for drill, status in results_summary.items():
            status_icon = "✅" if status == "PASSED" else "❌"
            print(f"   {status_icon} {drill}: {status}")
        
        print(f"\n🎯 Framework Status:")
        if passed == total and not missing_configs and not missing_analyzers:
            print("   🎉 All drill analyzers are ready for deployment!")
            print("   📦 Framework supports all 8 ball mastery drills")
            print("   🔧 Unified architecture with consistent results")
            print("   📊 Configurable success criteria for easy adjustment")
        else:
            print("   ⚠️  Some issues found - see details above")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()