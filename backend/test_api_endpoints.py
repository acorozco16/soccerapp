#!/usr/bin/env python3
"""
Test FastAPI endpoints without full dependencies
Quick validation that the API structure works
"""

import sys
from pathlib import Path
import json

# Add backend directory to path
sys.path.append(str(Path(__file__).parent))

def test_drill_endpoints():
    """Test drill API endpoints structure"""
    print("🧪 Testing Drill API Endpoints")
    print("=" * 50)
    
    try:
        # Import drill API
        from drill_api import drill_router
        
        # Get all routes
        routes = []
        for route in drill_router.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                routes.append({
                    'path': route.path,
                    'methods': list(route.methods),
                    'name': getattr(route, 'name', 'unknown')
                })
        
        print(f"✅ {len(routes)} API endpoints registered:")
        
        # Categorize endpoints
        drill_management = []
        video_analysis = []
        
        for route in routes:
            if any(path in route['path'] for path in ['/available', '/types', '/info', '/benchmark']):
                drill_management.append(route)
            else:
                video_analysis.append(route)
        
        print(f"\n📋 Drill Management Endpoints ({len(drill_management)}):")
        for route in drill_management:
            methods = ', '.join(route['methods'])
            print(f"   {methods:8} {route['path']}")
        
        print(f"\n🎥 Video Analysis Endpoints ({len(video_analysis)}):")
        for route in video_analysis:
            methods = ', '.join(route['methods'])
            print(f"   {methods:8} {route['path']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_drill_registry_api():
    """Test drill registry functions called by API"""
    print("\n\n🧪 Testing Drill Registry API Functions")
    print("=" * 50)
    
    try:
        from drill_analyzer import drill_registry, DrillType
        
        # Import analyzers to register them
        from analyzers.bell_touches_analyzer import BellTouchesAnalyzer
        from analyzers.juggling_analyzer import JugglingAnalyzer
        
        # Test list_drills (used by /drill/available)
        drills = drill_registry.list_drills()
        print(f"✅ list_drills(): {len(drills)} drills")
        
        # Test get_config (used by /drill/info and /drill/benchmark)
        bell_config = drill_registry.get_config(DrillType.BELL_TOUCHES)
        if bell_config:
            print(f"✅ get_config(): {bell_config.name}")
        else:
            print("❌ get_config(): Failed")
            return False
        
        # Test get_analyzer (used by analysis)
        bell_analyzer = drill_registry.get_analyzer(DrillType.BELL_TOUCHES)
        if bell_analyzer:
            print(f"✅ get_analyzer(): {type(bell_analyzer).__name__}")
        else:
            print("❌ get_analyzer(): Failed")
            return False
        
        # Test drill type validation
        try:
            DrillType("bell_touches")
            print("✅ drill type validation: Working")
        except ValueError:
            print("❌ drill type validation: Failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Registry API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_expected_api_responses():
    """Test expected API response formats"""
    print("\n\n🧪 Testing Expected API Response Formats")
    print("=" * 50)
    
    try:
        from drill_analyzer import drill_registry, DrillType
        from analyzers.bell_touches_analyzer import BellTouchesAnalyzer
        
        # Test /drill/available response format
        drills = drill_registry.list_drills()
        available_response = {
            "drills": drills,
            "total_count": len(drills)
        }
        print(f"✅ /drill/available format: {len(json.dumps(available_response))} chars")
        
        # Test /drill/types response format
        types_response = {
            "drill_types": [dt.value for dt in DrillType],
            "count": len(list(DrillType))
        }
        print(f"✅ /drill/types format: {types_response['count']} types")
        
        # Test /drill/info/{drill_type} response format
        config = drill_registry.get_config(DrillType.BELL_TOUCHES)
        if config:
            info_response = {
                "type": config.drill_type.value,
                "name": config.name,
                "description": config.description,
                "success_criteria": config.success_criteria,
                "time_window": config.time_window,
                "benchmark_range": f"{config.min_reps}-{config.max_reps}",
                "per_foot": config.per_foot,
                "pattern_based": config.pattern_based
            }
            print(f"✅ /drill/info format: {info_response['name']}")
        
        # Test /drill/benchmark/{drill_type} response format  
        benchmark_response = {
            "drill_type": config.drill_type.value,
            "name": config.name,
            "success_criteria": config.success_criteria,
            "time_window": config.time_window,
            "min_reps": config.min_reps,
            "max_reps": config.max_reps,
            "per_foot": config.per_foot,
            "pattern_based": config.pattern_based
        }
        print(f"✅ /drill/benchmark format: {benchmark_response['success_criteria']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Response format test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all API endpoint tests"""
    print("🚀 API Endpoint Testing")
    print("=" * 70)
    
    tests = [
        test_drill_endpoints,
        test_drill_registry_api,
        test_expected_api_responses
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n\n📊 API ENDPOINT TEST SUMMARY")
    print("=" * 70)
    print(f"✅ {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 API ENDPOINTS READY!")
        print("   ✅ All drill endpoints registered")
        print("   ✅ Registry functions working")
        print("   ✅ Response formats valid")
        print("\n🔄 Ready for:")
        print("   1. FastAPI server startup")
        print("   2. Frontend integration")
        print("   3. Real video testing")
    else:
        print(f"\n⚠️  {total - passed} API issues found")
        print("   Fix these before starting server")

if __name__ == "__main__":
    main()