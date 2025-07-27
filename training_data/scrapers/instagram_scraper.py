#!/usr/bin/env python3
"""
Instagram Soccer Data Scraper
Collects soccer training content from Instagram hashtags
"""

import os
import json
import asyncio
import aiohttp
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import hashlib
import requests
from urllib.parse import urlparse

import sys
sys.path.append(str(Path(__file__).parent.parent / "automation"))
from training_status import update_collection_status, Status

class InstagramSoccerScraper:
    def __init__(self, access_token: str, output_dir: str = "training_data/collected_data/instagram"):
        self.access_token = access_token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Instagram API endpoints
        self.api_base = "https://graph.instagram.com"
        
        # Soccer-related hashtags
        self.hashtags = [
            "soccertraining", "footballskills", "soccerdrills", "footballtraining",
            "soccerpractice", "ballcontrol", "juggling", "soccertricks",
            "footballworkout", "soccerfitness", "dribbling", "firsttouch",
            "passingdrills", "shootingpractice", "soccercoach", "youthsoccer"
        ]
        
        # Rate limiting
        self.requests_per_hour = 200  # Instagram API limit
        self.request_delay = 3600 / self.requests_per_hour  # Seconds between requests
        self.last_request_time = 0
        
        # Download settings
        self.session = None
        self.downloaded_posts = self._load_downloaded_posts()
    
    def _load_downloaded_posts(self) -> set:
        """Load list of already downloaded posts"""
        downloaded_file = self.output_dir / "downloaded_posts.json"
        if downloaded_file.exists():
            try:
                with open(downloaded_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('post_ids', []))
            except Exception as e:
                print(f"Warning: Could not load downloaded posts list: {e}")
        return set()
    
    def _save_downloaded_posts(self):
        """Save list of downloaded posts"""
        downloaded_file = self.output_dir / "downloaded_posts.json"
        try:
            data = {
                'post_ids': list(self.downloaded_posts),
                'last_updated': datetime.now().isoformat(),
                'total_count': len(self.downloaded_posts)
            }
            with open(downloaded_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving downloaded posts list: {e}")
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def _make_request(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make rate-limited API request"""
        await self._rate_limit()
        
        params['access_token'] = self.access_token
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limit exceeded
                        print("Rate limit exceeded, waiting 1 hour...")
                        await asyncio.sleep(3600)
                        return await self._make_request(url, {k: v for k, v in params.items() if k != 'access_token'})
                    else:
                        print(f"API request failed: {response.status}")
                        return None
        except Exception as e:
            print(f"Request error: {e}")
            return None
    
    async def _search_hashtag_posts(self, hashtag: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for posts with specific hashtag"""
        # Note: Instagram Basic Display API has limitations for hashtag search
        # This would require Instagram Graph API with proper permissions
        
        # For now, we'll use the user's own posts and filter by caption
        url = f"{self.api_base}/me/media"
        params = {
            'fields': 'id,media_type,media_url,thumbnail_url,caption,timestamp,permalink',
            'limit': limit
        }
        
        response = await self._make_request(url, params)
        if not response or 'data' not in response:
            return []
        
        # Filter posts that contain soccer-related keywords in caption
        soccer_posts = []
        soccer_keywords = ['soccer', 'football', 'ball', 'training', 'drill', 'practice']
        
        for post in response['data']:
            caption = post.get('caption', '').lower()
            if any(keyword in caption for keyword in soccer_keywords):
                soccer_posts.append(post)
        
        return soccer_posts
    
    async def _download_media(self, post: Dict[str, Any]) -> Optional[str]:
        """Download media file from post"""
        try:
            media_url = post.get('media_url')
            if not media_url:
                return None
            
            # Determine file extension
            parsed_url = urlparse(media_url)
            file_extension = Path(parsed_url.path).suffix or '.jpg'
            
            # Create filename
            post_id = post['id']
            filename = f"instagram_{post_id}{file_extension}"
            file_path = self.output_dir / "images" / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if already downloaded
            if file_path.exists():
                return str(file_path)
            
            # Download file
            async with aiohttp.ClientSession() as session:
                async with session.get(media_url) as response:
                    if response.status == 200:
                        content = await response.read()
                        with open(file_path, 'wb') as f:
                            f.write(content)
                        
                        # Save metadata
                        metadata_file = file_path.with_suffix('.json')
                        metadata = {
                            'post_id': post_id,
                            'instagram_url': post.get('permalink'),
                            'caption': post.get('caption'),
                            'timestamp': post.get('timestamp'),
                            'media_type': post.get('media_type'),
                            'downloaded_at': datetime.now().isoformat(),
                            'source': 'instagram',
                            'file_path': str(file_path)
                        }
                        
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        return str(file_path)
                    else:
                        print(f"Failed to download media: HTTP {response.status}")
                        return None
        
        except Exception as e:
            print(f"Error downloading media {post.get('id', 'unknown')}: {e}")
            return None
    
    def _assess_image_quality(self, image_path: Path) -> Dict[str, Any]:
        """Assess image quality for soccer training purposes"""
        try:
            import cv2
            import numpy as np
            
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return {'quality': 'low', 'reason': 'Cannot read image'}
            
            # Convert to different color spaces
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Quality metrics
            height, width = image.shape[:2]
            
            # 1. Resolution check
            min_resolution = 480
            resolution_score = min(1.0, min(height, width) / min_resolution)
            
            # 2. Brightness check
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128
            
            # 3. Contrast check
            contrast = np.std(gray)
            contrast_score = min(1.0, contrast / 64)
            
            # 4. Blur detection (Laplacian variance)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(1.0, blur_score / 1000)
            
            # 5. Color diversity (useful for ball detection)
            color_diversity = np.std(hsv[:,:,1])  # Saturation diversity
            color_score = min(1.0, color_diversity / 64)
            
            # Overall quality score
            overall_score = (resolution_score * 0.3 + 
                           brightness_score * 0.2 + 
                           contrast_score * 0.2 + 
                           blur_score * 0.2 + 
                           color_score * 0.1)
            
            # Categorize quality
            if overall_score >= 0.8:
                quality = 'high'
            elif overall_score >= 0.6:
                quality = 'medium'
            else:
                quality = 'low'
            
            return {
                'quality': quality,
                'overall_score': round(overall_score, 3),
                'resolution_score': round(resolution_score, 3),
                'brightness_score': round(brightness_score, 3),
                'contrast_score': round(contrast_score, 3),
                'blur_score': round(blur_score, 3),
                'color_score': round(color_score, 3),
                'image_size': [width, height]
            }
            
        except ImportError:
            return {'quality': 'unknown', 'reason': 'OpenCV not available'}
        except Exception as e:
            return {'quality': 'low', 'reason': f'Quality assessment failed: {e}'}
    
    async def scrape_soccer_content(self, max_posts: int = 100) -> Dict[str, Any]:
        """Scrape soccer content from Instagram"""
        
        update_collection_status(
            "instagram",
            status=Status.RUNNING,
            progress=0.0,
            items_collected=0,
            target_items=max_posts,
            current_operation="Starting Instagram scraping...",
            start_time=datetime.now()
        )
        
        start_time = datetime.now()
        downloaded_count = 0
        total_posts_found = 0
        quality_stats = {'high': 0, 'medium': 0, 'low': 0}
        errors = []
        
        try:
            # Search through hashtags
            all_posts = []
            
            for i, hashtag in enumerate(self.hashtags):
                if downloaded_count >= max_posts:
                    break
                
                update_collection_status(
                    "instagram",
                    progress=i / len(self.hashtags) * 0.5,  # First 50% is searching
                    current_operation=f"Searching hashtag: #{hashtag}"
                )
                
                try:
                    posts = await self._search_hashtag_posts(hashtag, limit=25)
                    all_posts.extend(posts)
                    total_posts_found += len(posts)
                    
                    print(f"Found {len(posts)} posts for #{hashtag}")
                    
                except Exception as e:
                    error_msg = f"Error searching #{hashtag}: {e}"
                    errors.append(error_msg)
                    print(error_msg)
            
            # Remove duplicates
            unique_posts = {}
            for post in all_posts:
                unique_posts[post['id']] = post
            
            all_posts = list(unique_posts.values())
            print(f"Found {len(all_posts)} unique posts total")
            
            # Download posts
            for i, post in enumerate(all_posts[:max_posts]):
                if post['id'] in self.downloaded_posts:
                    continue
                
                progress = 0.5 + (i / min(len(all_posts), max_posts)) * 0.5
                update_collection_status(
                    "instagram",
                    progress=progress,
                    current_operation=f"Downloading post {i+1}/{min(len(all_posts), max_posts)}"
                )
                
                try:
                    file_path = await self._download_media(post)
                    if file_path:
                        # Assess quality
                        quality_info = self._assess_image_quality(Path(file_path))
                        quality_stats[quality_info.get('quality', 'low')] += 1
                        
                        # Update metadata with quality info
                        metadata_file = Path(file_path).with_suffix('.json')
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            metadata['quality_assessment'] = quality_info
                            with open(metadata_file, 'w') as f:
                                json.dump(metadata, f, indent=2)
                        
                        self.downloaded_posts.add(post['id'])
                        downloaded_count += 1
                        
                        print(f"Downloaded: {file_path} (Quality: {quality_info.get('quality', 'unknown')})")
                
                except Exception as e:
                    error_msg = f"Error downloading post {post['id']}: {e}"
                    errors.append(error_msg)
                    print(error_msg)
                
                # Save progress periodically
                if downloaded_count % 10 == 0:
                    self._save_downloaded_posts()
                    update_collection_status(
                        "instagram",
                        items_collected=downloaded_count
                    )
            
            # Final save
            self._save_downloaded_posts()
            
            # Calculate final stats
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            summary = {
                'source': 'instagram',
                'started_at': start_time.isoformat(),
                'completed_at': end_time.isoformat(),
                'duration_seconds': round(duration, 2),
                'total_posts_found': total_posts_found,
                'posts_downloaded': downloaded_count,
                'quality_distribution': quality_stats,
                'hashtags_searched': self.hashtags,
                'errors': errors,
                'success_rate': round(downloaded_count / max(total_posts_found, 1) * 100, 2)
            }
            
            # Save summary
            summary_file = self.output_dir / f"scraping_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            update_collection_status(
                "instagram",
                status=Status.COMPLETED,
                progress=1.0,
                items_collected=downloaded_count,
                current_operation="Scraping completed"
            )
            
            print(f"\n✅ Instagram scraping completed!")
            print(f"📊 Downloaded {downloaded_count} posts in {duration:.1f} seconds")
            print(f"📈 Quality distribution: {quality_stats}")
            
            return summary
            
        except Exception as e:
            error_msg = f"Instagram scraping failed: {e}"
            update_collection_status(
                "instagram",
                status=Status.FAILED,
                error_message=error_msg
            )
            raise

async def main():
    """Test Instagram scraper"""
    # Load config
    config_file = Path("automation_config.json")
    if not config_file.exists():
        print("❌ Please run setup_automation.py first")
        return
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    instagram_token = config.get('api_keys', {}).get('instagram')
    if not instagram_token:
        print("❌ Instagram access token not configured")
        return
    
    scraper = InstagramSoccerScraper(instagram_token)
    
    try:
        result = await scraper.scrape_soccer_content(max_posts=50)
        print(f"✅ Scraping completed: {result}")
    except Exception as e:
        print(f"❌ Scraping failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())