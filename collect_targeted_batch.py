#!/usr/bin/env python3
"""
Collect targeted YouTube videos in batches for improved soccer ball detection training
"""

import os
import json
import yt_dlp
import cv2
from pathlib import Path
from datetime import datetime
import hashlib
import sys

class BatchVideoCollector:
    def __init__(self, output_dir="training_data/targeted_frames", batch_size=5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        
        # Track progress
        self.progress_file = self.output_dir / "collection_progress.json"
        self.load_progress()
        
        # Your curated list of videos (removing duplicates)
        self.all_video_urls = [
            # YouTube Shorts (vertical format)
            "https://www.youtube.com/shorts/3dKFMrfgITs",
            "https://www.youtube.com/shorts/Y8TA06lX074",
            "https://www.youtube.com/shorts/IwCTlb-hZuc",
            "https://www.youtube.com/shorts/-EDICTP6NEQ",
            "https://www.youtube.com/shorts/BBIe7Yi-t68",
            "https://www.youtube.com/shorts/9ieaL6i9SdI",
            "https://www.youtube.com/shorts/zNjz5whWZ6Y",
            "https://www.youtube.com/shorts/xoZU7UGO_TI",
            "https://www.youtube.com/shorts/U4EZWFCje7w",
            "https://www.youtube.com/shorts/dvQOcAP9LNg",
            "https://www.youtube.com/shorts/fhzbx8FqFBA",
            "https://www.youtube.com/shorts/4-lNILXj8b8",
            "https://www.youtube.com/shorts/Q7n_lpjrSL4",
            "https://www.youtube.com/shorts/9XWqC1xZEr8",
            "https://www.youtube.com/shorts/FQSUK-sWTzc",
            "https://www.youtube.com/shorts/6ddaCw3BLWA",
            "https://www.youtube.com/shorts/cjXaQU1nOsI",
            "https://www.youtube.com/shorts/96Pl-20Pveg",
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
        
        self.frame_interval = 1.5  # Extract frame every 1.5 seconds

    def load_progress(self):
        """Load progress from previous runs"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                "processed_urls": [],
                "extracted_frames": [],
                "total_frames": 0
            }
    
    def save_progress(self):
        """Save current progress"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def download_video(self, url):
        """Download video from YouTube"""
        print(f"\n📥 Downloading: {url}")
        
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_id = info['id']
                video_title = info.get('title', 'Unknown')[:50]  # Truncate long titles
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
                    self.progress["extracted_frames"].append(filename)
                    frames_extracted += 1
                
                frame_count += 1
            
            cap.release()
            
            # Clean up video file
            video_file.unlink()
            
            print(f"✅ Extracted {frames_extracted} frames")
            return frames_extracted
            
        except Exception as e:
            print(f"❌ Error extracting frames: {str(e)[:100]}")
            return 0

    def process_batch(self, batch_num=None):
        """Process a single batch of videos"""
        # Get unprocessed URLs
        remaining_urls = [url for url in self.all_video_urls if url not in self.progress["processed_urls"]]
        
        if not remaining_urls:
            print("✅ All videos have been processed!")
            return False
        
        # Determine which batch to process
        if batch_num is None:
            batch_urls = remaining_urls[:self.batch_size]
        else:
            start_idx = batch_num * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_urls = remaining_urls[start_idx:end_idx]
        
        if not batch_urls:
            print("✅ No more videos to process in this batch!")
            return False
        
        print(f"\n🎯 Processing batch of {len(batch_urls)} videos")
        print(f"📊 Total progress: {len(self.progress['processed_urls'])}/{len(self.all_video_urls)} videos")
        
        batch_frames = 0
        
        for url in batch_urls:
            video_file, video_id, duration = self.download_video(url)
            
            if video_file and video_id:
                frames = self.extract_frames(video_file, video_id, duration)
                batch_frames += frames
                self.progress["total_frames"] += frames
                
            # Mark as processed regardless of success
            self.progress["processed_urls"].append(url)
            self.save_progress()
        
        print(f"\n📊 Batch complete! Extracted {batch_frames} frames")
        print(f"📊 Total frames so far: {self.progress['total_frames']}")
        
        return True

    def get_status(self):
        """Get current collection status"""
        processed = len(self.progress["processed_urls"])
        total = len(self.all_video_urls)
        
        print(f"\n📊 COLLECTION STATUS")
        print(f"{'='*40}")
        print(f"Videos processed: {processed}/{total}")
        print(f"Frames extracted: {self.progress['total_frames']}")
        print(f"Frames saved to: {self.output_dir}")
        
        if processed < total:
            print(f"\n⏳ {total - processed} videos remaining")
            print(f"Run 'python collect_targeted_batch.py' to continue")
        else:
            print(f"\n✅ Collection complete!")

if __name__ == "__main__":
    collector = BatchVideoCollector(batch_size=5)
    
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        collector.get_status()
    else:
        # Process next batch
        if collector.process_batch():
            collector.get_status()
            print(f"\n🎯 Run this script again to process the next batch!")