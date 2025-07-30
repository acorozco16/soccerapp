#!/usr/bin/env python3
"""
Simulate the range system output to validate the approach
"""

def simulate_range_calculation(detected_touches, confidence, quality_score):
    """Simulate the _calculate_touch_range method"""
    
    # Base uncertainty factors
    base_uncertainty = 2  # ±2 touches base uncertainty
    
    # Adjust uncertainty based on confidence score
    if confidence >= 0.8:
        confidence_factor = 0.5  # High confidence: smaller range
    elif confidence >= 0.6:
        confidence_factor = 1.0  # Medium confidence: normal range
    else:
        confidence_factor = 1.5  # Low confidence: larger range
    
    # Adjust uncertainty based on video quality
    if quality_score >= 0.8:
        quality_factor = 0.8  # High quality: smaller range
    elif quality_score >= 0.6:
        quality_factor = 1.0  # Good quality: normal range
    else:
        quality_factor = 1.3  # Poor quality: larger range
    
    # Calculate range bounds
    uncertainty = int(base_uncertainty * confidence_factor * quality_factor)
    uncertainty = max(1, min(uncertainty, 5))  # Clamp between 1-5
    
    # Calculate range ensuring it doesn't go below 0
    range_min = max(0, detected_touches - uncertainty)
    range_max = detected_touches + uncertainty
    
    # Special handling for very low counts
    if detected_touches <= 3:
        range_min = max(0, detected_touches - 1)
        range_max = detected_touches + 2
    
    return {
        "min": range_min,
        "max": range_max,
        "display": f"{range_min}-{range_max} touches",
        "detected_count": detected_touches,
        "confidence_level": "high" if confidence >= 0.7 else "medium" if confidence >= 0.5 else "low",
        "explanation": f"Detected {detected_touches} touches with {confidence:.0%} confidence"
    }

def test_range_scenarios():
    """Test various scenarios"""
    
    scenarios = [
        # (detected_touches, confidence, quality_score, description)
        (22, 0.75, 0.8, "High quality video, good confidence - reference case"),
        (22, 0.55, 0.8, "High quality video, medium confidence"),
        (22, 0.35, 0.8, "High quality video, low confidence"),
        (22, 0.75, 0.5, "Poor quality video, good confidence"),
        (10, 0.65, 0.7, "Medium count, medium confidence"),
        (2, 0.45, 0.6, "Very low count (edge case)"),
        (35, 0.85, 0.9, "High count, excellent conditions"),
    ]
    
    print("🧪 Range System Test Results")
    print("=" * 60)
    
    for detected, conf, quality, desc in scenarios:
        result = simulate_range_calculation(detected, conf, quality)
        
        print(f"\n📋 {desc}")
        print(f"   Input: {detected} touches, {conf:.0%} confidence, {quality:.1f} quality")
        print(f"   ⚽ Output: {result['display']}")
        print(f"   📊 Level: {result['confidence_level']}")
        print(f"   💡 {result['explanation']}")
        
        # Calculate range size for analysis
        range_size = result['max'] - result['min']
        print(f"   📏 Range size: ±{range_size//2} touches")

if __name__ == "__main__":
    test_range_scenarios()