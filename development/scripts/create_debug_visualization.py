#!/usr/bin/env python3
"""
Debug Visualization Tool for Soccer Video Analysis
This will show you exactly what the computer vision is counting as "touches"
"""

def create_debug_endpoint():
    """Add debug visualization endpoint to see what CV is detecting"""
    
    debug_endpoint_code = '''
@app.get("/debug/video/{analysis_id}")
async def debug_video_analysis(analysis_id: str):
    """
    Debug endpoint to visualize what the computer vision detected
    Returns the processed video with annotations showing detected touches
    """
    try:
        # Find the video file
        video_path = f"/root/soccerapp/uploads/raw/{analysis_id}.mp4"
        debug_output_path = f"/root/soccerapp/uploads/debug/{analysis_id}_debug.mp4"
        
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Create debug directory if it doesn't exist
        os.makedirs("/root/soccerapp/uploads/debug", exist_ok=True)
        
        # Process video with debug annotations
        video_processor = VideoProcessor()
        debug_results = await video_processor.analyze_video_with_debug(video_path, analysis_id)
        
        return {
            "debug_video_url": f"https://soccertrainingapp.org/debug/video/{analysis_id}_debug.mp4",
            "detected_touches": debug_results["touches"],
            "touch_timestamps": debug_results["timestamps"],
            "confidence_scores": debug_results["confidences"],
            "detection_methods": debug_results["methods"]
        }
        
    except Exception as e:
        logger.error(f"Debug visualization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/download/{filename}")
async def download_debug_video(filename: str):
    """Serve debug videos for download"""
    file_path = f"/root/soccerapp/uploads/debug/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4", filename=filename)
    else:
        raise HTTPException(status_code=404, detail="Debug video not found")
'''
    
    # Read main.py
    with open('/root/soccerapp/backend/main.py', 'r') as f:
        content = f.read()
    
    # Add the debug endpoints before the last line
    insertion_point = content.rfind('if __name__ == "__main__":')
    if insertion_point != -1:
        content = content[:insertion_point] + debug_endpoint_code + '\n\n' + content[insertion_point:]
    else:
        content += '\n\n' + debug_endpoint_code
    
    # Write back
    with open('/root/soccerapp/backend/main.py', 'w') as f:
        f.write(content)
    
    print("✅ Added debug visualization endpoints")

def add_debug_method_to_video_processor():
    """Add debug analysis method to VideoProcessor"""
    
    debug_method_code = '''
    async def analyze_video_with_debug(self, video_path: str, video_id: str) -> Dict:
        """Analyze video and create debug visualization showing detected touches"""
        import cv2
        
        # Get regular analysis results first
        results = await self.analyze_video(video_path, video_id)
        
        # Create debug video with annotations
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output video writer
        output_path = f"/root/soccerapp/uploads/debug/{video_id}_debug.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        touch_events = results.get("details", {}).get("touch_events", [])
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            
            # Draw all detected touches near this timestamp
            for touch in touch_events:
                touch_time = touch.get("timestamp", 0)
                if abs(touch_time - current_time) < 0.5:  # Within 0.5 seconds
                    # Draw circle at touch position
                    pos = touch.get("position", [width//2, height//2])
                    cv2.circle(frame, (int(pos[0]), int(pos[1])), 20, (0, 255, 0), 3)
                    
                    # Add text showing touch number and confidence
                    touch_num = touch.get("frame", 0)
                    confidence = touch.get("confidence", 0)
                    method = touch.get("detection_method", "unknown")
                    
                    text = f"Touch #{touch_num} ({confidence:.2f})"
                    cv2.putText(frame, text, (int(pos[0])-50, int(pos[1])-30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Add method text
                    cv2.putText(frame, method, (int(pos[0])-50, int(pos[1])+40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Add frame counter and timestamp
            cv2.putText(frame, f"Frame: {frame_count} Time: {current_time:.1f}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add total touch count
            cv2.putText(frame, f"Total Touches: {len(touch_events)}", 
                       (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        return {
            "touches": len(touch_events),
            "timestamps": [t.get("timestamp", 0) for t in touch_events],
            "confidences": [t.get("confidence", 0) for t in touch_events],
            "methods": [t.get("detection_method", "unknown") for t in touch_events],
            "debug_video_path": output_path
        }
'''
    
    # Read video_processor.py
    with open('/root/soccerapp/backend/video_processor.py', 'r') as f:
        content = f.read()
    
    # Find a good insertion point (end of class)
    insertion_point = content.rfind('        return analysis_result')
    if insertion_point != -1:
        # Find the end of the method
        end_point = content.find('\n\n', insertion_point)
        if end_point != -1:
            content = content[:end_point] + '\n' + debug_method_code + content[end_point:]
        else:
            content += debug_method_code
    else:
        content += debug_method_code
    
    # Write back
    with open('/root/soccerapp/backend/video_processor.py', 'w') as f:
        f.write(content)
    
    print("✅ Added debug analysis method to VideoProcessor")

def add_missing_import():
    """Add missing FileResponse import"""
    
    # Read main.py
    with open('/root/soccerapp/backend/main.py', 'r') as f:
        content = f.read()
    
    # Add FileResponse import if not present
    if 'FileResponse' not in content:
        fastapi_import = 'from fastapi import'
        if fastapi_import in content:
            content = content.replace(
                'from fastapi import',
                'from fastapi import FileResponse,'
            )
        else:
            # Add new import line
            content = 'from fastapi.responses import FileResponse\n' + content
    
    # Write back
    with open('/root/soccerapp/backend/main.py', 'w') as f:
        f.write(content)
    
    print("✅ Added FileResponse import")

if __name__ == "__main__":
    print("🔍 Creating debug visualization tools...")
    add_missing_import()
    create_debug_endpoint()
    add_debug_method_to_video_processor()
    print("✅ Debug visualization tools created!")
    print("\nUsage after server restart:")
    print("GET /debug/video/{analysis_id} - Get debug analysis")
    print("GET /debug/download/{filename} - Download debug video")