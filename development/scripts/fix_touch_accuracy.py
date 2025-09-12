#!/usr/bin/env python3
"""
Fix touch detection accuracy - reduce false positives
"""

def improve_touch_accuracy():
    """Apply stricter filtering to reduce false positives"""
    
    # Read video_processor.py
    with open('/root/soccerapp/backend/video_processor.py', 'r') as f:
        content = f.read()
    
    # Find the trajectory touch detection threshold
    old_threshold = r'angle_change=([0-9.]+), distance=([0-9.]+)'
    
    # Look for where outliers are removed
    outlier_section = content.find('Removing outlier touch at')
    if outlier_section != -1:
        print("Found outlier removal section")
    
    # Increase minimum time between touches (reduce rapid-fire false positives)
    min_touch_interval = r'last_touch_time \+ 0\.[0-9]+'
    if 'last_touch_time + 0.' in content:
        # Increase minimum interval from current to 0.3 seconds
        content = re.sub(r'last_touch_time \+ 0\.[0-9]+', 'last_touch_time + 0.3', content)
        print("✅ Increased minimum touch interval to 0.3 seconds")
    
    # Increase confidence threshold for trajectory touches
    trajectory_confidence = r'confidence=0\.[0-9]+'
    if 'confidence=0.' in content:
        content = re.sub(r'confidence=0\.[0-9]+', 'confidence=0.75', content)
        print("✅ Increased trajectory touch confidence threshold")
    
    # Add stricter angle change requirements
    if 'angle_change=2.00' in content:
        # Replace overly sensitive angle detection
        content = content.replace('angle_change=2.00', 'angle_change=1.50')
        print("✅ Reduced angle sensitivity to prevent false positives")
    
    # Increase distance threshold for valid touches
    if 'distance=' in content:
        # Find distance thresholds and make them more restrictive
        import re
        pattern = r'distance=([0-9.]+)'
        matches = re.findall(pattern, content)
        if matches:
            print(f"Found distance values: {matches}")
            # Only count touches with significant ball movement
            content = re.sub(r'distance=([0-9.]+)', lambda m: f'distance={float(m.group(1))}' if float(m.group(1)) > 20 else 'distance=0', content)
    
    # Write back
    with open('/root/soccerapp/backend/video_processor.py', 'w') as f:
        f.write(content)
    
    print("✅ Applied touch accuracy improvements")

if __name__ == "__main__":
    import re
    print("🎯 Improving touch detection accuracy...")
    improve_touch_accuracy()
    print("✅ Touch accuracy improvements applied!")