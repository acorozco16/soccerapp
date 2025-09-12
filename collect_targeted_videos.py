#!/usr/bin/env python3
"""
Collect targeted YouTube videos for improved soccer ball detection training
"""

import os
import json
import yt_dlp
import cv2
from pathlib import Path
from datetime import datetime
import hashlib

class TargetedVideoCollector:
    def __init__(self, output_dir="training_data/targeted_frames"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Your curated list of videos (removing duplicates)
        self.video_urls = [
            # YouTube Shorts (vertical format)
            "https://www.youtube.com/shorts/3dKFMrfgITs",
            "https://www.youtube.com/shorts/Y8TA06lX074",  # Note: appeared 3 times
            "https://www.youtube.com/shorts/IwCTlb-hZuc",
            "https://www.youtube.com/shorts/-EDICTP6NEQ",
            "https://www.youtube.com/shorts/BBIe7Yi-t68",
            "https://www.youtube.com/shorts/9ieaL6i9SdI",
            "https://www.youtube.com/shorts/zNjz5whWZ6Y",
            "https://www.youtube.com/shorts/xoZU7UGO_TI",
            "https://www.youtube.com/shorts/U4EZWFCje7w",  # Note: appeared 2 times
            "https://www.youtube.com/shorts/dvQOcAP9LNg",
            "https://www.youtube.com/shorts/fhzbx8FqFBA",
            "https://www.youtube.com/shorts/4-lNILXj8b8",
            "https://www.youtube.com/shorts/Q7n_lpjrSL4",
            "https://www.youtube.com/shorts/9XWqC1xZEr8",
            "https://www.youtube.com/shorts/FQSUK-sWTzc",
            "https://www.youtube.com/shorts/6ddaCw3BLWA",
            "https://www.youtube.com/shorts/cjXaQU1nOsI",
            "https://www.youtube.com/shorts/96Pl-20Pveg",  # Note: appeared 2 times
            "https://www.youtube.com/shorts/FNQzD6m7vfg",
            "https://www.youtube.com/shorts/whg73wTu6mo",
            "https://www.youtube.com/shorts/2a9C25cSr5s",
            "https://www.youtube.com/shorts/V3UqdDTfnZU",
            "https://www.youtube.com/shorts/yc66qOt9dDk",
            "https://www.youtube.com/shorts/PpNcdtKJzJ8",
            "https://www.youtube.com/shorts/DN-1uEbApxk",
            
            # Full YouTube videos
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
        }
        
        self.extracted_frames = []
        self.frame_interval = 1.5  # Extract frame every 1.5 seconds

    def download_video(self, url):
        """Download video from YouTube"""
        print(f"\n📥 Downloading: {url}")
        
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_id = info['id']
                video_title = info.get('title', 'Unknown')
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
                    return video_file, video_id, duration
                else:
                    print(f"❌ Failed to find downloaded file")
                    return None, None, 0
                    
        except Exception as e:
            print(f"❌ Error downloading {url}: {e}")
            return None, None, 0

    def extract_frames(self, video_file, video_id, duration):
        """Extract frames from video"""
        try:
            cap = cv2.VideoCapture(str(video_file))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps <= 0:
                fps = 30  # Default fallback
            
            frame_interval_frames = int(fps * self.frame_interval)
            frames_extracted = 0
            
            print(f"📸 Extracting frames every {self.frame_interval}s from {duration}s video...")
            
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
                
                frame_count += 1
            
            cap.release()
            
            # Clean up video file
            video_file.unlink()
            
            print(f"✅ Extracted {frames_extracted} frames")
            return frames_extracted
            
        except Exception as e:
            print(f"❌ Error extracting frames: {e}")
            return 0

    def collect_all_videos(self):
        """Process all videos"""
        print(f"🎯 Starting targeted video collection")
        print(f"📊 Processing {len(self.video_urls)} unique videos")
        
        total_frames = 0
        successful_videos = 0
        
        for i, url in enumerate(self.video_urls, 1):
            print(f"\n{'='*60}")
            print(f"Processing video {i}/{len(self.video_urls)}")
            
            video_file, video_id, duration = self.download_video(url)
            
            if video_file and video_id:
                frames = self.extract_frames(video_file, video_id, duration)
                total_frames += frames
                if frames > 0:
                    successful_videos += 1
        
        # Save summary
        summary = {
            "collection_date": datetime.now().isoformat(),
            "total_videos_processed": len(self.video_urls),
            "successful_videos": successful_videos,
            "total_frames_extracted": total_frames,
            "frame_files": self.extracted_frames,
            "average_frames_per_video": total_frames / successful_videos if successful_videos > 0 else 0
        }
        
        summary_file = self.output_dir / "collection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✅ COLLECTION COMPLETE!")
        print(f"📊 Videos processed: {successful_videos}/{len(self.video_urls)}")
        print(f"🖼️ Total frames extracted: {total_frames}")
        print(f"📁 Frames saved to: {self.output_dir}")
        print(f"📄 Summary saved to: {summary_file}")
        
        return total_frames

if __name__ == "__main__":
    collector = TargetedVideoCollector()
    total_frames = collector.collect_all_videos()
    
    print(f"\n🎯 Next steps:")
    print(f"1. Label these {total_frames} targeted frames")
    print(f"2. Combine with your existing 4,234 labeled frames")
    print(f"3. Retrain the model to achieve 90%+ detection")