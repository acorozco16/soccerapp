#!/usr/bin/env python3
"""
Simple fix - just expose the analysis details that already exist
"""

def add_simple_debug_endpoint():
    """Add a simple endpoint to get detailed analysis results"""
    
    # Simple endpoint code
    endpoint_code = '''
@app.get("/debug/analysis/{analysis_id}")
async def get_detailed_analysis(analysis_id: str):
    """Get detailed analysis results for debugging"""
    try:
        # Check if we have analysis results stored
        result_file = f"/root/soccerapp/uploads/results/{analysis_id}_details.json"
        
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                details = json.load(f)
            return details
        else:
            # Try to get from database or return basic info
            return {
                "analysis_id": analysis_id,
                "status": "Analysis details not found",
                "suggestion": "Re-run analysis to get detailed results"
            }
            
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}")
        return {"error": str(e)}
'''
    
    # Read main.py
    with open('/root/soccerapp/backend/main.py', 'r') as f:
        content = f.read()
    
    # Remove any broken debug endpoints first
    lines = content.split('\n')
    filtered_lines = []
    skip_until_next_function = False
    
    for line in lines:
        if '@app.get("/debug/' in line:
            skip_until_next_function = True
            continue
        elif skip_until_next_function and (line.startswith('@app.') or line.startswith('def ') or line.startswith('async def')):
            skip_until_next_function = False
            
        if not skip_until_next_function:
            filtered_lines.append(line)
    
    content = '\n'.join(filtered_lines)
    
    # Add the simple endpoint before the main block
    if 'if __name__ == "__main__":' in content:
        content = content.replace(
            'if __name__ == "__main__":',
            endpoint_code + '\n\nif __name__ == "__main__":'
        )
    else:
        content += '\n' + endpoint_code
    
    # Write back
    with open('/root/soccerapp/backend/main.py', 'w') as f:
        f.write(content)
    
    print("✅ Added simple debug endpoint")

def enhance_analysis_details_storage():
    """Modify video processor to save detailed analysis results"""
    
    # Read video_processor.py
    with open('/root/soccerapp/backend/video_processor.py', 'r') as f:
        content = f.read()
    
    # Find where analysis results are returned
    if 'return analysis_result' in content:
        # Add code to save detailed results
        save_code = '''
        # Save detailed results for debugging
        try:
            os.makedirs("/root/soccerapp/uploads/results", exist_ok=True)
            details_file = f"/root/soccerapp/uploads/results/{video_id}_details.json"
            
            detailed_results = {
                "analysis_id": video_id,
                "total_touches": len(touch_events),
                "touch_events": [touch.to_dict() for touch in touch_events],
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "video_metadata": metadata,
                "confidence_threshold": self.yolo_confidence_threshold,
                "detection_summary": {
                    "yolo_detections": sum(1 for t in touch_events if "yolo" in t.detection_method.lower()),
                    "trajectory_detections": sum(1 for t in touch_events if "trajectory" in t.detection_method.lower()),
                    "traditional_detections": sum(1 for t in touch_events if "traditional" in t.detection_method.lower())
                }
            }
            
            with open(details_file, 'w') as f:
                json.dump(detailed_results, f, indent=2)
                
            logger.info(f"Saved detailed analysis results to {details_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save detailed results: {e}")
        
        '''
        
        # Insert before return statement
        content = content.replace(
            'return analysis_result',
            save_code + '\n        return analysis_result'
        )
        
        # Write back
        with open('/root/soccerapp/backend/video_processor.py', 'w') as f:
            f.write(content)
        
        print("✅ Enhanced analysis details storage")

if __name__ == "__main__":
    print("🔧 Adding simple debug capabilities...")
    add_simple_debug_endpoint()
    enhance_analysis_details_storage()
    print("✅ Simple debug capabilities added!")
    print("\nAfter server restart:")
    print("GET /debug/analysis/{analysis_id} - Get detailed touch detection info")