#!/usr/bin/env python3
"""
CLI tool for testing the video analysis pipeline
Usage: python analyze_sample.py --video path/to/video.mp4
"""

import argparse
import asyncio
import json
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from video_processor import VideoProcessor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def analyze_video(video_path: str, output_dir: str = None):
    """Analyze a video and save results"""
    print(f"\n🎥 Analyzing video: {video_path}")
    print("=" * 50)
    
    # Check if file exists
    if not Path(video_path).exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return
    
    # Initialize processor
    processor = VideoProcessor()
    
    # Generate video ID
    video_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Process video
        print("⏳ Processing video... This may take a minute.")
        results = await processor.analyze_video(video_path, video_id)
        
        # Display results
        print("\n✅ Analysis Complete!")
        print("=" * 50)
        
        # Show range display (primary result)
        if 'touch_range' in results:
            touch_range = results['touch_range']
            print(f"⚽ Ball Touches: {touch_range['display']}")
            print(f"📊 Detected Count: {results['total_ball_touches']} (confidence: {touch_range['confidence_level']})")
            print(f"💡 {touch_range['explanation']}")
        else:
            print(f"📊 Total Ball Touches: {results['total_ball_touches']}")
        
        print(f"⏱️  Video Duration: {results['video_duration']:.1f} seconds")
        print(f"⚡ Touches per Minute: {results['touches_per_minute']}")
        print(f"🎯 Confidence Score: {results['confidence_score']}")
        print(f"🕐 Processing Time: {results['processing_time']:.1f} seconds")
        
        # Show touch events
        if results['touch_events']:
            print(f"\n🏃 Touch Events ({len(results['touch_events'])} total):")
            for i, touch in enumerate(results['touch_events'][:5], 1):
                print(f"  {i}. Time: {touch['timestamp']:.1f}s, "
                      f"Frame: {touch['frame']}, "
                      f"Confidence: {touch['confidence']:.2f}")
            if len(results['touch_events']) > 5:
                print(f"  ... and {len(results['touch_events']) - 5} more touches")
        
        # Save results
        if output_dir:
            output_path = Path(output_dir) / f"{video_id}_results.json"
        else:
            output_path = Path(video_path).parent / f"{Path(video_path).stem}_results.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
        
        # Show debug frames location
        if results['debug_frames']:
            frames_dir = Path(__file__).parent.parent / "uploads" / "frames" / video_id
            print(f"🖼️  Debug frames saved in: {frames_dir}")
            print(f"   Frames: {', '.join(results['debug_frames'])}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze soccer video for ball touches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_sample.py --video ../sample_videos/clear_touches.mp4
  python analyze_sample.py --video my_video.mp4 --output ./results/
        """
    )
    
    parser.add_argument(
        '--video', '-v',
        type=str,
        required=True,
        help='Path to video file (MP4 or MOV)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for results (default: same as video)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Use quick mode (process fewer frames)'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    asyncio.run(analyze_video(args.video, args.output))


if __name__ == "__main__":
    main()