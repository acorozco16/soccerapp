#!/usr/bin/env python3
"""
Test drill analysis with mock video data
Simulates the complete video → analysis → results pipeline
"""

import sys
from pathlib import Path
import json

# Add backend directory to path
sys.path.append(str(Path(__file__).parent))

def create_mock_video_data():
    """Create realistic mock video data that simulates VideoProcessor output"""
    return {
        "video_id": "test_video_123",
        "duration": 30.0,
        "frame_height": 720,
        "frame_width": 1280,
        "fps": 30,
        
        # Mock ball detections (simulating Bell Touches)
        "ball_detections": [
            {"timestamp": 1.0, "frame_number": 30, "position": (100, 600), "confidence": 0.8, "method": "yolo"},
            {"timestamp": 1.5, "frame_number": 45, "position": (200, 600), "confidence": 0.85, "method": "yolo"},
            {"timestamp": 2.0, "frame_number": 60, "position": (100, 600), "confidence": 0.82, "method": "yolo"},
            {"timestamp": 2.5, "frame_number": 75, "position": (200, 600), "confidence": 0.83, "method": "yolo"},
            {"timestamp": 3.0, "frame_number": 90, "position": (100, 600), "confidence": 0.81, "method": "yolo"},
            {"timestamp": 3.5, "frame_number": 105, "position": (200, 600), "confidence": 0.84, "method": "yolo"},
            {"timestamp": 4.0, "frame_number": 120, "position": (100, 600), "confidence": 0.82, "method": "yolo"},
            {"timestamp": 4.5, "frame_number": 135, "position": (200, 600), "confidence": 0.85, "method": "yolo"},
            {"timestamp": 5.0, "frame_number": 150, "position": (100, 600), "confidence": 0.83, "method": "yolo"},
            {"timestamp": 5.5, "frame_number": 165, "position": (200, 600), "confidence": 0.84, "method": "yolo"},
            {"timestamp": 6.0, "frame_number": 180, "position": (100, 600), "confidence": 0.82, "method": "yolo"},
            {"timestamp": 6.5, "frame_number": 195, "position": (200, 600), "confidence": 0.85, "method": "yolo"},
            {"timestamp": 7.0, "frame_number": 210, "position": (100, 600), "confidence": 0.83, "method": "yolo"},
            {"timestamp": 7.5, "frame_number": 225, "position": (200, 600), "confidence": 0.84, "method": "yolo"},
            {"timestamp": 8.0, "frame_number": 240, "position": (100, 600), "confidence": 0.82, "method": "yolo"},
            {"timestamp": 8.5, "frame_number": 255, "position": (200, 600), "confidence": 0.85, "method": "yolo"},
            {"timestamp": 9.0, "frame_number": 270, "position": (100, 600), "confidence": 0.83, "method": "yolo"},
            {"timestamp": 9.5, "frame_number": 285, "position": (200, 600), "confidence": 0.84, "method": "yolo"},
            {"timestamp": 10.0, "frame_number": 300, "position": (100, 600), "confidence": 0.82, "method": "yolo"},
            {"timestamp": 10.5, "frame_number": 315, "position": (200, 600), "confidence": 0.85, "method": "yolo"},
        ],
        
        # Mock foot positions (feet positioned for ground-level touches)
        "foot_positions": {
            30: [(100, 600), (200, 600)],   45: [(100, 600), (200, 600)],
            60: [(100, 600), (200, 600)],   75: [(100, 600), (200, 600)],
            90: [(100, 600), (200, 600)],   105: [(100, 600), (200, 600)],
            120: [(100, 600), (200, 600)],  135: [(100, 600), (200, 600)],
            150: [(100, 600), (200, 600)],  165: [(100, 600), (200, 600)],
            180: [(100, 600), (200, 600)],  195: [(100, 600), (200, 600)],
            210: [(100, 600), (200, 600)],  225: [(100, 600), (200, 600)],
            240: [(100, 600), (200, 600)],  255: [(100, 600), (200, 600)],
            270: [(100, 600), (200, 600)],  285: [(100, 600), (200, 600)],
            300: [(100, 600), (200, 600)],  315: [(100, 600), (200, 600)],
        },
        
        # Mock touch events (for compatibility)
        "touch_events": [
            {"timestamp": t["timestamp"], "frame": t["frame_number"], "position": t["position"], 
             "confidence": t["confidence"], "detection_method": t["method"]} 
            for t in [
                {"timestamp": 1.0, "frame_number": 30, "position": (100, 600), "confidence": 0.8, "method": "yolo"},
                {"timestamp": 1.5, "frame_number": 45, "position": (200, 600), "confidence": 0.85, "method": "yolo"},
            ]
        ],
        
        # Mock raw results (for juggling compatibility)
        "raw_results": {
            "total_ball_touches": 20,
            "confidence_score": 0.83,
            "video_duration": 30.0,
            "touch_range": {
                "min": 18,
                "max": 22,
                "display": "18-22 touches",
                "confidence_level": "high"
            }
        }
    }

def test_drill_analysis():
    """Test all drill analyzers with mock video data"""
    print("🧪 Testing Complete Drill Analysis Pipeline")
    print("=" * 60)
    
    # Import analyzers
    from drill_analyzer import drill_registry, DrillType
    
    # Import all analyzers to ensure registration
    from analyzers.bell_touches_analyzer import BellTouchesAnalyzer
    from analyzers.inside_outside_analyzer import InsideOutsideAnalyzer  
    from analyzers.sole_rolls_analyzer import SoleRollsAnalyzer
    from analyzers.outside_foot_push_analyzer import OutsideFootPushAnalyzer
    from analyzers.v_cuts_analyzer import VCutsAnalyzer
    from analyzers.croquetas_analyzer import CroquetasAnalyzer
    from analyzers.triangles_analyzer import TrianglesAnalyzer
    from analyzers.juggling_analyzer import JugglingAnalyzer
    
    # Create mock video data
    mock_data = create_mock_video_data()
    
    # Test each drill type
    test_results = {}
    
    drill_types = [
        DrillType.BELL_TOUCHES,
        DrillType.JUGGLING,
        DrillType.INSIDE_OUTSIDE,
        DrillType.SOLE_ROLLS,
        DrillType.OUTSIDE_FOOT_PUSH,
        DrillType.V_CUTS,
        DrillType.CROQUETAS,
        DrillType.TRIANGLES
    ]
    
    for drill_type in drill_types:
        print(f"\n🔧 Testing {drill_type.value.upper()}")
        print("-" * 40)
        
        try:
            # Get analyzer
            analyzer = drill_registry.get_analyzer(drill_type)
            if not analyzer:
                print(f"   ❌ No analyzer found")
                test_results[drill_type.value] = "FAILED - No analyzer"
                continue
            
            # Run analysis
            results = analyzer.analyze(mock_data)
            
            # Display results
            print(f"   📊 Count: {results.count_detected}")
            print(f"   📊 Range: {results.count_range['display']}")
            print(f"   📊 Benchmark: {'✅ MET' if results.benchmark_met else '❌ NOT MET'}")
            print(f"   📊 Confidence: {results.confidence:.2f}")
            
            if results.per_foot_counts:
                foot_summary = {k: v for k, v in results.per_foot_counts.items() 
                              if isinstance(v, (int, float))}
                if foot_summary:
                    print(f"   👣 Per Foot: {foot_summary}")
            
            # Test JSON serialization
            json_output = results.to_dict()
            print(f"   📄 JSON: {len(json.dumps(json_output))} chars")
            
            test_results[drill_type.value] = "PASSED"
            
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
            test_results[drill_type.value] = f"FAILED - {str(e)[:50]}..."
    
    return test_results

def test_api_data_format():
    """Test that results match expected API format"""
    print("\n\n🧪 Testing API Data Format")
    print("=" * 60)
    
    from drill_analyzer import drill_registry, DrillType
    from analyzers.bell_touches_analyzer import BellTouchesAnalyzer
    
    # Test with Bell Touches
    analyzer = drill_registry.get_analyzer(DrillType.BELL_TOUCHES)
    mock_data = create_mock_video_data()
    
    results = analyzer.analyze(mock_data)
    api_response = results.to_dict()
    
    # Validate API response structure
    required_keys = ['drill_type', 'success_criteria', 'results', 'metadata']
    missing_keys = [key for key in required_keys if key not in api_response]
    
    if missing_keys:
        print(f"   ❌ Missing required keys: {missing_keys}")
        return False
    
    # Validate results structure
    results_keys = ['count_detected', 'count_range', 'duration', 'benchmark_met', 'confidence']
    missing_results = [key for key in results_keys if key not in api_response['results']]
    
    if missing_results:
        print(f"   ❌ Missing results keys: {missing_results}")
        return False
    
    print("   ✅ API response structure valid")
    print(f"   📊 Sample response:")
    print(f"      Drill: {api_response['drill_type']}")
    print(f"      Count: {api_response['results']['count_detected']}")
    print(f"      Range: {api_response['results']['count_range']['display']}")
    print(f"      Benchmark: {api_response['results']['benchmark_met']}")
    
    return True

def main():
    """Run complete mock video test"""
    print("🎯 Mock Video Analysis Test")
    print("=" * 70)
    
    try:
        # Test drill analysis
        drill_results = test_drill_analysis()
        
        # Test API format
        api_valid = test_api_data_format()
        
        # Summary
        passed_drills = sum(1 for result in drill_results.values() if result == "PASSED")
        total_drills = len(drill_results)
        
        print(f"\n\n📊 MOCK VIDEO TEST SUMMARY")
        print("=" * 70)
        print(f"🔧 Drill Analysis: {passed_drills}/{total_drills} passed")
        print(f"📄 API Format: {'✅ Valid' if api_valid else '❌ Invalid'}")
        
        print(f"\n📋 Individual Results:")
        for drill, result in drill_results.items():
            status = "✅" if result == "PASSED" else "❌"
            print(f"   {status} {drill}: {result}")
        
        if passed_drills == total_drills and api_valid:
            print(f"\n🎉 MOCK VIDEO TEST SUCCESS!")
            print("   ✅ All drill analyzers working")
            print("   ✅ API format valid")
            print("   ✅ Ready to process real videos")
            print("\n🔄 Next Steps:")
            print("   1. Start FastAPI server: python main.py")
            print("   2. Test with real video file")
            print("   3. Build frontend interface")
        else:
            print(f"\n⚠️  Issues found in mock test")
            print("   Fix these before processing real videos")
        
    except Exception as e:
        print(f"\n❌ Mock video test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()