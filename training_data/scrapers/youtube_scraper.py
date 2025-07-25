#!/usr/bin/env python3
"""
YouTube Soccer Video Scraper
Downloads soccer training videos with rate limiting and quality filtering
"""

import os
import time
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import yt_dlp
from youtubesearchpython import VideosSearch
import cv2
import numpy as np
from tqdm import tqdm
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    video_id: str
    title: str
    duration: int  # seconds
    upload_date: str
    view_count: int
    channel: str
    url: str
    download_path: str = ""
    quality_score: float = 0.0
    ball_visibility_score: float = 0.0
    frame_count: int = 0
    processed: bool = False


@dataclass
class ScrapingStats:
    total_searched: int = 0
    total_downloaded: int = 0
    total_processed: int = 0
    total_frames_extracted: int = 0
    total_frames_with_ball: int = 0
    errors: List[str] = None
    start_time: datetime = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.start_time is None:
            self.start_time = datetime.now()


class RateLimiter:
    """Rate limiter to prevent API abuse"""
    
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests = []
        
    async def wait_if_needed(self):
        now = datetime.now()
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < timedelta(minutes=1)]
        
        if len(self.requests) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.requests[0]).total_seconds()
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
        
        self.requests.append(now)


class YouTubeSoccerScraper:
    def __init__(self, output_dir: str = "./scraped_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.videos_dir = self.output_dir / "videos"
        self.frames_dir = self.output_dir / "frames"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.videos_dir, self.frames_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.rate_limiter = RateLimiter(requests_per_minute=20)
        self.downloaded_videos: Set[str] = set()
        self.stats = ScrapingStats()
        
        # Load existing downloads
        self._load_existing_downloads()
        
        # Soccer-specific search terms
        self.search_terms = [
            "soccer ball juggling training",
            "football skills practice",
            "soccer dribbling drills",
            "football ball control",
            "soccer first touch training",
            "football passing drills",
            "soccer shooting practice",
            "youth soccer training",
            "football technique practice",
            "soccer ball work drills"
        ]
    
    def _load_existing_downloads(self):
        """Load list of already downloaded videos"""
        metadata_file = self.metadata_dir / "downloaded.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    self.downloaded_videos = set(data.get('video_ids', []))
                logger.info(f"Loaded {len(self.downloaded_videos)} existing downloads")
            except Exception as e:
                logger.error(f"Error loading existing downloads: {e}")
    
    def _save_downloaded_list(self):
        """Save list of downloaded videos"""
        metadata_file = self.metadata_dir / "downloaded.json"
        data = {
            'video_ids': list(self.downloaded_videos),
            'last_updated': datetime.now().isoformat(),
            'stats': asdict(self.stats)
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    async def search_videos(self, query: str, max_results: int = 50) -> List[VideoInfo]:
        """Search for videos on YouTube"""
        await self.rate_limiter.wait_if_needed()
        
        try:
            videos_search = VideosSearch(query, limit=max_results)
            results = videos_search.result()
            
            video_infos = []
            for video in results['result']:
                # Parse duration
                duration_str = video.get('duration', '0:00')
                duration_parts = duration_str.split(':')
                if len(duration_parts) == 2:
                    duration = int(duration_parts[0]) * 60 + int(duration_parts[1])
                elif len(duration_parts) == 3:
                    duration = int(duration_parts[0]) * 3600 + int(duration_parts[1]) * 60 + int(duration_parts[2])
                else:
                    duration = 0
                
                # Filter by duration (30 seconds to 5 minutes)
                if not (30 <= duration <= 300):
                    continue
                
                # Skip if already downloaded
                video_id = video['id']
                if video_id in self.downloaded_videos:
                    continue
                
                video_info = VideoInfo(
                    video_id=video_id,
                    title=video['title'],
                    duration=duration,
                    upload_date=video.get('publishedTime', ''),
                    view_count=self._parse_view_count(video.get('viewCount', {}).get('text', '0')),
                    channel=video.get('channel', {}).get('name', ''),
                    url=video['link']
                )
                
                video_infos.append(video_info)
            
            self.stats.total_searched += len(video_infos)
            logger.info(f"Found {len(video_infos)} suitable videos for query: {query}")
            return video_infos
            
        except Exception as e:
            error_msg = f"Error searching for query '{query}': {e}"
            logger.error(error_msg)
            self.stats.errors.append(error_msg)
            return []
    
    def _parse_view_count(self, view_count_str: str) -> int:
        """Parse view count string to integer"""
        try:
            # Remove non-numeric characters except for multipliers
            cleaned = view_count_str.lower().replace(',', '').replace(' views', '')
            
            if 'k' in cleaned:
                return int(float(cleaned.replace('k', '')) * 1000)
            elif 'm' in cleaned:
                return int(float(cleaned.replace('m', '')) * 1000000)
            elif 'b' in cleaned:
                return int(float(cleaned.replace('b', '')) * 1000000000)
            else:
                return int(''.join(filter(str.isdigit, cleaned)) or 0)
        except:
            return 0
    
    async def download_video(self, video_info: VideoInfo) -> bool:
        """Download a single video"""
        await self.rate_limiter.wait_if_needed()
        
        output_path = self.videos_dir / f"{video_info.video_id}.mp4"
        
        # Check if already exists
        if output_path.exists():
            video_info.download_path = str(output_path)
            return True
        
        ydl_opts = {
            'format': 'best[height<=720][ext=mp4]',  # Limit to 720p MP4
            'outtmpl': str(self.videos_dir / '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_info.url])
            
            if output_path.exists():
                video_info.download_path = str(output_path)
                self.downloaded_videos.add(video_info.video_id)
                self.stats.total_downloaded += 1
                logger.info(f"Downloaded: {video_info.title[:50]}...")
                return True
            else:
                error_msg = f"Download failed for {video_info.video_id}: File not found after download"
                logger.error(error_msg)
                self.stats.errors.append(error_msg)
                return False
                
        except Exception as e:
            error_msg = f"Download error for {video_info.video_id}: {e}"
            logger.error(error_msg)
            self.stats.errors.append(error_msg)
            return False
    
    def assess_ball_visibility(self, frame: np.ndarray) -> float:
        """Assess how visible the soccer ball is in a frame"""
        # Convert to HSV for better ball detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for soccer balls (orange and white)
        orange_lower = np.array([5, 100, 100])
        orange_upper = np.array([15, 255, 255])
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        
        # Create masks
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        combined_mask = cv2.bitwise_or(orange_mask, white_mask)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_score = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 5000:  # Reasonable ball size
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.4:  # Reasonably circular
                        # Score based on size and circularity
                        size_score = min(area / 2000, 1.0)  # Normalize area
                        score = circularity * size_score
                        best_score = max(best_score, score)
        
        return best_score
    
    async def extract_frames(self, video_info: VideoInfo, interval: float = 0.5) -> List[str]:
        """Extract frames from video at specified interval"""
        if not video_info.download_path or not Path(video_info.download_path).exists():
            return []
        
        video_frames_dir = self.frames_dir / video_info.video_id
        video_frames_dir.mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(video_info.download_path)
        if not cap.isOpened():
            error_msg = f"Cannot open video: {video_info.download_path}"
            logger.error(error_msg)
            self.stats.errors.append(error_msg)
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)  # Extract every 0.5 seconds
        
        frame_paths = []
        frame_count = 0
        ball_visible_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Assess ball visibility
                    ball_score = self.assess_ball_visibility(frame)
                    
                    if ball_score > 0.3:  # Only save frames with decent ball visibility
                        frame_path = video_frames_dir / f"frame_{frame_count:06d}.jpg"
                        cv2.imwrite(str(frame_path), frame)
                        frame_paths.append(str(frame_path))
                        ball_visible_count += 1
                        
                        # Save metadata for this frame
                        frame_metadata = {
                            'frame_number': frame_count,
                            'timestamp': frame_count / fps,
                            'ball_visibility_score': ball_score,
                            'video_id': video_info.video_id
                        }
                        
                        metadata_path = video_frames_dir / f"frame_{frame_count:06d}.json"
                        with open(metadata_path, 'w') as f:
                            json.dump(frame_metadata, f, indent=2)
                
                frame_count += 1
        
        finally:
            cap.release()
        
        # Update video info
        video_info.frame_count = len(frame_paths)
        video_info.ball_visibility_score = ball_visible_count / max(1, frame_count // frame_interval)
        video_info.processed = True
        
        self.stats.total_frames_extracted += len(frame_paths)
        self.stats.total_frames_with_ball += ball_visible_count
        
        logger.info(f"Extracted {len(frame_paths)} frames with ball from {video_info.video_id}")
        return frame_paths
    
    async def scrape_soccer_videos(self, max_videos_per_term: int = 20, max_total_videos: int = 200) -> Dict:
        """Main scraping function"""
        logger.info(f"Starting soccer video scraping. Target: {max_total_videos} videos")
        
        all_videos = []
        
        # Search with different terms
        for term in self.search_terms:
            if len(all_videos) >= max_total_videos:
                break
                
            logger.info(f"Searching for: {term}")
            videos = await self.search_videos(term, max_videos_per_term)
            all_videos.extend(videos)
            
            # Add delay between searches
            await asyncio.sleep(2)
        
        # Remove duplicates and limit total
        unique_videos = {}
        for video in all_videos:
            if video.video_id not in unique_videos:
                unique_videos[video.video_id] = video
        
        videos_to_process = list(unique_videos.values())[:max_total_videos]
        logger.info(f"Found {len(videos_to_process)} unique videos to process")
        
        # Download and process videos
        successful_videos = []
        
        for i, video in enumerate(tqdm(videos_to_process, desc="Processing videos")):
            try:
                # Check system resources
                if psutil.virtual_memory().percent > 90:
                    logger.warning("High memory usage, pausing...")
                    await asyncio.sleep(5)
                
                # Download video
                if await self.download_video(video):
                    # Extract frames
                    frame_paths = await self.extract_frames(video)
                    
                    if frame_paths:
                        successful_videos.append(video)
                        self.stats.total_processed += 1
                
                # Save progress periodically
                if i % 10 == 0:
                    self._save_downloaded_list()
                
                # Add delay to be respectful
                await asyncio.sleep(1)
                
            except Exception as e:
                error_msg = f"Error processing video {video.video_id}: {e}"
                logger.error(error_msg)
                self.stats.errors.append(error_msg)
        
        # Final save
        self._save_downloaded_list()
        
        # Generate summary
        summary = {
            'total_videos_found': len(videos_to_process),
            'total_videos_processed': len(successful_videos),
            'total_frames_extracted': self.stats.total_frames_extracted,
            'total_frames_with_ball': self.stats.total_frames_with_ball,
            'processing_time': (datetime.now() - self.stats.start_time).total_seconds(),
            'errors': len(self.stats.errors),
            'videos': [asdict(video) for video in successful_videos]
        }
        
        # Save summary
        summary_path = self.metadata_dir / f"scraping_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Scraping complete! Summary saved to {summary_path}")
        return summary


async def main():
    """Example usage"""
    scraper = YouTubeSoccerScraper("./training_data/scraped_data")
    
    try:
        summary = await scraper.scrape_soccer_videos(
            max_videos_per_term=15,
            max_total_videos=100
        )
        
        print("\n=== SCRAPING SUMMARY ===")
        print(f"Videos processed: {summary['total_videos_processed']}")
        print(f"Frames extracted: {summary['total_frames_extracted']}")
        print(f"Frames with ball: {summary['total_frames_with_ball']}")
        print(f"Processing time: {summary['processing_time']:.1f}s")
        print(f"Errors: {summary['errors']}")
        
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Scraping failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())