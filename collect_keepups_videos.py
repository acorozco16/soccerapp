#!/usr/bin/env python3
"""
Soccer Keep-ups Video Collector for Labeling Pipeline
Targets specific keep-ups videos for ball detection training
"""

import os
import time
import json
import asyncio
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import yt_dlp
from youtubesearchpython import VideosSearch
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeepUpsCollector:
    def __init__(self, output_dir: str = "training_data"):
        self.output_dir = Path(output_dir)
        
        # Create directory structure
        self.raw_videos_dir = self.output_dir / "raw_videos"
        self.labeling_queue_dir = self.output_dir / "labeling_queue"
        
        for dir_path in [self.raw_videos_dir, self.labeling_queue_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Keep-ups specific search terms
        self.search_terms = [
            "soccer keep ups",
            "football keepups",
            "soccer juggling",
            "football juggling",
            "soccer ball juggling",
            "football keepy ups",
            "soccer freestyle keepups",
            "soccer skills keepups",
            "football keep up challenge",
            "soccer ball control juggling"
        ]
        
        # Collection results
        self.collected_videos = []
        self.metadata = {
            'collection_date': datetime.now().isoformat(),
            'videos': [],
            'total_frames_extracted': 0
        }

    def filter_keepups_video(self, video_data: dict) -> bool:
        """Filter for good keep-ups videos"""
        title = video_data.get('title', '').lower()
        duration_str = video_data.get('duration', '0:00')
        
        # Parse duration
        parts = duration_str.split(':')
        if len(parts) == 2:
            duration = int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            duration = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            duration = 0
        
        # Duration filter: 30 seconds to 5 minutes
        if not (30 <= duration <= 300):
            return False
        
        # View count filter
        view_count_text = video_data.get('viewCount', {}).get('text', '0')
        view_count = self._parse_view_count(view_count_text)
        if view_count < 1000:
            return False
        
        # Title keywords that indicate good keep-ups content
        good_keywords = [
            'keep up', 'keepup', 'juggling', 'freestyle', 'ball control',
            'touches', 'skills', 'keepy up', 'ball juggling'
        ]
        
        # Bad keywords that indicate not individual keep-ups
        bad_keywords = [
            'match', 'game', 'vs', 'team', 'compilation', 'fails', 
            'funny', 'trick shots', 'goals', 'highlights'
        ]
        
        # Check for good keywords
        has_good_keyword = any(keyword in title for keyword in good_keywords)
        has_bad_keyword = any(keyword in title for keyword in bad_keywords)
        
        return has_good_keyword and not has_bad_keyword

    def _parse_view_count(self, view_count_str: str) -> int:
        """Parse view count string to integer"""
        try:
            cleaned = view_count_str.lower().replace(',', '').replace(' views', '')
            
            if 'k' in cleaned:
                return int(float(cleaned.replace('k', '')) * 1000)
            elif 'm' in cleaned:
                return int(float(cleaned.replace('m', '')) * 1000000)
            else:
                return int(''.join(filter(str.isdigit, cleaned)) or 0)
        except:
            return 0

    async def search_keepups_videos(self, max_results: int = 50) -> List[dict]:
        """Search for keep-ups videos"""
        all_videos = []
        
        for search_term in self.search_terms:
            logger.info(f"Searching: {search_term}")
            
            try:
                videos_search = VideosSearch(search_term, limit=max_results)
                results = videos_search.result()
                
                for video in results['result']:
                    if self.filter_keepups_video(video):
                        video_info = {
                            'video_id': video['id'],
                            'title': video['title'],
                            'url': video['link'],
                            'duration': video.get('duration', ''),
                            'view_count': video.get('viewCount', {}).get('text', '0'),
                            'channel': video.get('channel', {}).get('name', ''),
                            'search_term': search_term
                        }
                        all_videos.append(video_info)
                
                # Respectful delay
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Search error for '{search_term}': {e}")
        
        # Remove duplicates
        unique_videos = {}
        for video in all_videos:
            if video['video_id'] not in unique_videos:
                unique_videos[video['video_id']] = video
        
        videos = list(unique_videos.values())
        logger.info(f"Found {len(videos)} unique keep-ups videos")
        return videos

    async def download_video(self, video_info: dict) -> bool:
        """Download a single keep-ups video"""
        video_id = video_info['video_id']
        output_path = self.raw_videos_dir / f"{video_id}.mp4"
        
        # Skip if already exists
        if output_path.exists():
            logger.info(f"Already exists: {video_info['title'][:50]}")
            video_info['download_path'] = str(output_path)
            return True
        
        ydl_opts = {
            'format': 'best[height<=720][ext=mp4]',
            'outtmpl': str(self.raw_videos_dir / '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_info['url']])
            
            if output_path.exists():
                video_info['download_path'] = str(output_path)
                logger.info(f"Downloaded: {video_info['title'][:50]}")
                return True
            else:
                logger.error(f"Download failed: {video_id}")
                return False
                
        except Exception as e:
            logger.error(f"Download error for {video_id}: {e}")
            return False

    def extract_frames_for_labeling(self, video_info: dict) -> List[str]:
        """Extract frames every 1-2 seconds for labeling"""
        video_path = video_info.get('download_path')
        if not video_path or not Path(video_path).exists():
            return []
        
        video_id = video_info['video_id']
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1.5)  # Extract every 1.5 seconds
        
        frame_paths = []
        frame_count = 0
        extracted_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Save frame for labeling
                    timestamp = frame_count / fps
                    frame_filename = f"{video_id}_frame_{timestamp:.1f}s.jpg"
                    frame_path = self.labeling_queue_dir / frame_filename
                    
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(str(frame_path))
                    extracted_count += 1
                
                frame_count += 1
        
        finally:
            cap.release()
        
        logger.info(f"Extracted {extracted_count} frames from {video_id}")
        video_info['frames_extracted'] = extracted_count
        video_info['frame_paths'] = frame_paths
        
        return frame_paths

    async def collect_keepups_dataset(self, target_videos: int = 25) -> dict:
        """Main collection function"""
        logger.info(f"🎯 Collecting {target_videos} keep-ups videos for labeling")
        
        # Search for videos
        logger.info("🔍 Searching for keep-ups videos...")
        videos = await self.search_keepups_videos()
        
        if len(videos) == 0:
            logger.error("❌ No suitable videos found")
            return {'success': False, 'error': 'No videos found'}
        
        # Select best videos (by view count)
        videos.sort(key=lambda x: self._parse_view_count(x['view_count']), reverse=True)
        selected_videos = videos[:target_videos]
        
        logger.info(f"📋 Selected top {len(selected_videos)} videos by view count")
        
        # Download and extract frames
        successful_videos = []
        total_frames = 0
        
        for i, video in enumerate(selected_videos, 1):
            logger.info(f"📹 Processing {i}/{len(selected_videos)}: {video['title'][:50]}")
            
            # Download video
            if await self.download_video(video):
                # Extract frames
                frame_paths = self.extract_frames_for_labeling(video)
                
                if frame_paths:
                    successful_videos.append(video)
                    total_frames += len(frame_paths)
                    logger.info(f"✅ Extracted {len(frame_paths)} frames")
                else:
                    logger.warning(f"⚠️ No frames extracted from {video['video_id']}")
            
            # Respectful delay
            await asyncio.sleep(1)
        
        # Save metadata
        self.metadata.update({
            'total_videos_downloaded': len(successful_videos),
            'total_frames_extracted': total_frames,
            'videos': successful_videos
        })
        
        metadata_path = self.output_dir / "keepups_collection_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        logger.info(f"✅ Collection complete!")
        logger.info(f"   Videos downloaded: {len(successful_videos)}")
        logger.info(f"   Frames extracted: {total_frames}")
        logger.info(f"   Metadata saved: {metadata_path}")
        
        return {
            'success': True,
            'videos_downloaded': len(successful_videos),
            'frames_extracted': total_frames,
            'metadata_path': str(metadata_path),
            'labeling_queue_dir': str(self.labeling_queue_dir)
        }

async def main():
    """Run keep-ups collection"""
    collector = KeepUpsCollector()
    
    try:
        result = await collector.collect_keepups_dataset(target_videos=25)
        
        if result['success']:
            print(f"\n🎉 Keep-ups collection successful!")
            print(f"   Videos: {result['videos_downloaded']}")
            print(f"   Frames: {result['frames_extracted']}")
            print(f"   Next: Build labeling interface")
        else:
            print(f"❌ Collection failed: {result.get('error', 'Unknown error')}")
            
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
    except Exception as e:
        logger.error(f"Collection failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())