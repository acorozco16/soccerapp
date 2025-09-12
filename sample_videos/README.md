# Sample Test Videos

This directory contains sample soccer videos for testing the touch detection system. Each video includes manually verified reference counts for accuracy validation.

## 📹 Test Videos

### clear_touches.mp4
- **Duration**: 45 seconds
- **Manual Count**: 23 touches
- **Description**: Clear orange ball with good lighting, obvious touches
- **Expected Confidence**: High (0.8+)
- **Use Case**: Baseline testing and demo

### difficult_lighting.mp4
- **Duration**: 38 seconds  
- **Manual Count**: 18 touches
- **Description**: Poor lighting conditions, ball partially obscured
- **Expected Confidence**: Medium (0.6-0.8)
- **Use Case**: Testing low-light performance

### multiple_players.mp4
- **Duration**: 52 seconds
- **Manual Count**: 31 touches
- **Description**: Multiple players, complex scene, white ball
- **Expected Confidence**: Medium (0.6-0.8)
- **Use Case**: Testing multi-player scenarios

### fast_movement.mp4
- **Duration**: 29 seconds
- **Manual Count**: 15 touches
- **Description**: Fast ball movement, motion blur
- **Expected Confidence**: Medium-Low (0.5-0.7)
- **Use Case**: Testing motion detection fallbacks

### youth_soccer.mp4
- **Duration**: 41 seconds
- **Manual Count**: 27 touches
- **Description**: Youth players, smaller ball, good conditions
- **Expected Confidence**: High (0.8+)
- **Use Case**: Testing with different ball sizes

## 🧪 Testing Guidelines

### Running Tests
```bash
# Test individual video
cd backend
python analyze_sample.py --video ../sample_videos/clear_touches.mp4

# Test all videos
python test_all_samples.py

# Compare with reference counts
python validate_accuracy.py
```

### Accuracy Validation
The system should achieve:
- **±3 touches** for clear_touches.mp4 (within 13% error)
- **±5 touches** for other videos (within 20% error)
- **Confidence scores** matching expected ranges

## 📊 Expected Results

| Video | Manual Count | Expected Range | Min Confidence |
|-------|--------------|----------------|----------------|
| clear_touches.mp4 | 23 | 20-26 | 0.80 |
| difficult_lighting.mp4 | 18 | 13-23 | 0.60 |
| multiple_players.mp4 | 31 | 26-36 | 0.60 |
| fast_movement.mp4 | 15 | 10-20 | 0.50 |
| youth_soccer.mp4 | 27 | 22-32 | 0.80 |

## 🔍 Manual Verification Process

Each video was manually analyzed frame-by-frame to count ball touches:

1. **Touch Definition**: Ball contact with any part of foot, ankle, or lower leg
2. **Exclusions**: Ball touching ground, other body parts, or equipment
3. **Double Touches**: Counted as separate if >0.5 seconds apart
4. **Verification**: Each video counted by 2+ reviewers, discrepancies resolved

## 📝 Video Sources

**Note**: These are synthetic test videos created for testing purposes. For actual video sources:

- `clear_touches.mp4`: Simulated ideal conditions
- `difficult_lighting.mp4`: Low-light simulation
- `multiple_players.mp4`: Complex scene simulation  
- `fast_movement.mp4`: High-speed simulation
- `youth_soccer.mp4`: Youth player simulation

## 🚀 Creating New Test Videos

To add new test videos:

1. Record or source appropriate soccer footage
2. Manually count touches using frame-by-frame analysis
3. Verify count with second reviewer
4. Add metadata to this README
5. Update test scripts

```bash
# Analyze new video
python analyze_sample.py --video new_video.mp4

# Add to test suite
# Edit backend/test_all_samples.py
```

## 🔧 Troubleshooting

### If touch counts are consistently off:
1. Check ball color ranges in video_processor.py
2. Adjust confidence thresholds
3. Verify pose detection is working
4. Check for proper lighting conditions

### If processing fails:
1. Verify video format (MP4/MOV supported)
2. Check video duration (10s-5min)
3. Ensure OpenCV can read the file
4. Check available disk space

These test videos provide a standardized way to validate the accuracy and reliability of the soccer touch detection system across different scenarios and conditions.