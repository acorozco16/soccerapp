#!/usr/bin/env python3
"""
Collect remaining YouTube videos for soccer ball detection training
"""

import os
import json
import yt_dlp
import cv2
from pathlib import Path
from datetime import datetime
import hashlib

class RemainingVideoCollector:
    def __init__(self, output_dir="training_data/targeted_frames"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track already processed videos
        self.existing_frames = len(list(self.output_dir.glob("*.jpg")))
        print(f"📊 Found {self.existing_frames} existing frames")
        
        # Remaining full YouTube videos
        self.remaining_urls = [
            "https://www.youtube.com/watch?v=AqNds72O2Ao",
            "https://www.youtube.com/watch?v=xFp7btucv58",
            "https://www.youtube.com/watch?v=9wslCudJ6sU",
            "https://www.youtube.com/watch?v=UUjDJyhsJ1g",
            "https://www.youtube.com/watch?v=URvtp8tS5j8",
            "https://www.youtube.com/watch?v=3wP5nNWmrtU",
            "https://www.youtube.com/watch?v=NMJHu0IneFU",
            "https://www.youtube.com/watch?v=Cjmb90TjVbQ",
            "https://www.youtube.com/watch?v=BEcpRfM2jgQ",
        ]
        
        self.ydl_opts = {
            'format': 'best[height<=720]',
            'quiet': True,
            'no_warnings': True,
            'outtmpl': str(self.output_dir / 'temp_%(id)s.%(ext)s'),
            'socket_timeout': 30,
        }
        
        self.frame_interval = 1.5  # Extract frame every 1.5 seconds
        self.extracted_frames = []

    def download_video(self, url):
        """Download video from YouTube"""
        print(f"\n📥 Downloading: {url}")
        
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_id = info['id']
                video_title = info.get('title', 'Unknown')[:50]
                duration = info.get('duration', 0)
                
                # Find downloaded file
                video_file = None
                for ext in ['mp4', 'webm', 'mkv']:
                    temp_file = self.output_dir / f'temp_{video_id}.{ext}'
                    if temp_file.exists():
                        video_file = temp_file
                        break
                
                if video_file:
                    print(f"✅ Downloaded: {video_title} ({duration}s)")
                    estimated_frames = int(duration / self.frame_interval)
                    print(f"📸 Expecting ~{estimated_frames} frames")
                    return video_file, video_id, duration
                else:
                    print(f"❌ Failed to find downloaded file")
                    return None, None, 0
                    
        except Exception as e:
            print(f"❌ Error downloading {url}: {str(e)[:100]}")
            return None, None, 0

    def extract_frames(self, video_file, video_id, duration):
        """Extract frames from video"""
        try:
            cap = cv2.VideoCapture(str(video_file))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if fps <= 0:
                fps = 30  # Default fallback
            
            frame_interval_frames = int(fps * self.frame_interval)
            frames_extracted = 0
            
            print(f"📸 Extracting frames every {self.frame_interval}s...")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval_frames == 0:
                    # Generate unique filename
                    timestamp = frame_count / fps
                    frame_hash = hashlib.md5(f"{video_id}_{timestamp}".encode()).hexdigest()[:8]
                    filename = f"{video_id}_{frame_hash}_{int(timestamp)}s.jpg"
                    filepath = self.output_dir / filename
                    
                    # Save frame
                    cv2.imwrite(str(filepath), frame)
                    self.extracted_frames.append(filename)
                    frames_extracted += 1
                    
                    # Progress update
                    if frames_extracted % 50 == 0:
                        print(f"   ...extracted {frames_extracted} frames")
                
                frame_count += 1
            
            cap.release()
            
            # Clean up video file
            try:
                video_file.unlink()
            except:
                print(f"⚠️ Could not delete temp file: {video_file}")
            
            print(f"✅ Extracted {frames_extracted} frames")
            return frames_extracted
            
        except Exception as e:
            print(f"❌ Error extracting frames: {str(e)[:100]}")
            return 0

    def collect_all_videos(self):
        """Process all remaining videos"""
        print(f"🎯 Starting collection of remaining {len(self.remaining_urls)} full videos")
        print(f"📊 Already have {self.existing_frames} frames from YouTube Shorts")
        
        total_new_frames = 0
        successful_videos = 0
        
        for i, url in enumerate(self.remaining_urls, 1):
            print(f"\n{'='*60}")
            print(f"Processing video {i}/{len(self.remaining_urls)}")
            
            video_file, video_id, duration = self.download_video(url)
            
            if video_file and video_id:
                frames = self.extract_frames(video_file, video_id, duration)
                total_new_frames += frames
                if frames > 0:
                    successful_videos += 1
            
            print(f"📊 Progress: {total_new_frames} new frames extracted")
        
        # Save summary
        total_frames = self.existing_frames + total_new_frames
        summary = {
            "collection_date": datetime.now().isoformat(),
            "existing_frames": self.existing_frames,
            "new_frames_extracted": total_new_frames,
            "total_frames": total_frames,
            "successful_videos": successful_videos,
            "total_videos_attempted": len(self.remaining_urls)
        }
        
        summary_file = self.output_dir / "final_collection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✅ COLLECTION COMPLETE!")
        print(f"📊 Videos processed: {successful_videos}/{len(self.remaining_urls)}")
        print(f"🖼️ New frames extracted: {total_new_frames}")
        print(f"🎯 Total targeted frames: {total_frames}")
        print(f"📁 All frames saved to: {self.output_dir}")
        print(f"📄 Summary saved to: {summary_file}")
        
        return total_frames

if __name__ == "__main__":
    collector = RemainingVideoCollector()
    total_frames = collector.collect_all_videos()
    
    print(f"\n🎯 Next steps:")
    print(f"1. Label these {total_frames} targeted frames")
    print(f"2. Combine with your existing 4,234 labeled frames")
    print(f"3. Retrain the model to achieve 90%+ detection")