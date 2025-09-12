#!/usr/bin/env python3
"""
Stock Photo API Scraper
Collects high-quality soccer images from stock photo APIs (Unsplash, Pexels)
"""

import os
import json
import asyncio
import aiohttp
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib
from urllib.parse import urlparse

import sys
sys.path.append(str(Path(__file__).parent.parent / "automation"))
from training_status import update_collection_status, Status

class StockPhotoScraper:
    def __init__(self, unsplash_key: str = None, pexels_key: str = None,
                 output_dir: str = "training_data/collected_data/stock_photos"):
        self.unsplash_key = unsplash_key
        self.pexels_key = pexels_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # API endpoints
        self.unsplash_api = "https://api.unsplash.com"
        self.pexels_api = "https://api.pexels.com/v1"
        
        # Search terms for soccer content
        self.search_terms = [
            "soccer ball", "football ball", "soccer training", "football training",
            "soccer player", "football player", "soccer field", "football field",
            "soccer practice", "football practice", "ball control", "dribbling",
            "soccer cleats", "football boots", "goalkeeper", "penalty kick",
            "soccer game", "football match", "youth soccer", "kids football"
        ]
        
        # Rate limiting
        self.unsplash_rate_limit = 50  # requests per hour
        self.pexels_rate_limit = 200   # requests per hour
        
        # Quality settings
        self.min_resolution = (800, 600)  # Minimum image resolution
        
        self.downloaded_images = self._load_downloaded_images()
    
    def _load_downloaded_images(self) -> set:
        """Load list of already downloaded images"""
        downloaded_file = self.output_dir / "downloaded_images.json"
        if downloaded_file.exists():
            try:
                with open(downloaded_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('image_ids', []))
            except Exception as e:
                print(f"Warning: Could not load downloaded images list: {e}")
        return set()
    
    def _save_downloaded_images(self):
        """Save list of downloaded images"""
        downloaded_file = self.output_dir / "downloaded_images.json"
        try:
            data = {
                'image_ids': list(self.downloaded_images),
                'last_updated': datetime.now().isoformat(),
                'total_count': len(self.downloaded_images)
            }
            with open(downloaded_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving downloaded images list: {e}")
    
    async def _make_unsplash_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make request to Unsplash API"""
        if not self.unsplash_key:
            return None
        
        headers = {"Authorization": f"Client-ID {self.unsplash_key}"}
        url = f"{self.unsplash_api}{endpoint}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 403:
                        print("Unsplash API rate limit exceeded")
                        return None
                    else:
                        print(f"Unsplash API error: {response.status}")
                        return None
        except Exception as e:
            print(f"Unsplash request error: {e}")
            return None
    
    async def _make_pexels_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make request to Pexels API"""
        if not self.pexels_key:
            return None
        
        headers = {"Authorization": self.pexels_key}
        url = f"{self.pexels_api}{endpoint}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        print("Pexels API rate limit exceeded")
                        return None
                    else:
                        print(f"Pexels API error: {response.status}")
                        return None
        except Exception as e:
            print(f"Pexels request error: {e}")
            return None
    
    async def _search_unsplash(self, query: str, per_page: int = 30) -> List[Dict[str, Any]]:
        """Search Unsplash for images"""
        params = {
            'query': query,
            'per_page': per_page,
            'orientation': 'all',
            'content_filter': 'high'
        }
        
        response = await self._make_unsplash_request('/search/photos', params)
        if not response:
            return []
        
        images = []
        for photo in response.get('results', []):
            # Filter by resolution
            width = photo.get('width', 0)
            height = photo.get('height', 0)
            
            if width >= self.min_resolution[0] and height >= self.min_resolution[1]:
                images.append({
                    'id': f"unsplash_{photo['id']}",
                    'source': 'unsplash',
                    'url': photo['urls']['regular'],
                    'download_url': photo['urls']['full'],
                    'width': width,
                    'height': height,
                    'description': photo.get('description') or photo.get('alt_description', ''),
                    'author': photo.get('user', {}).get('name', ''),
                    'unsplash_url': photo.get('links', {}).get('html', ''),
                    'query': query,
                    'likes': photo.get('likes', 0)
                })
        
        return images
    
    async def _search_pexels(self, query: str, per_page: int = 80) -> List[Dict[str, Any]]:
        """Search Pexels for images"""
        params = {
            'query': query,
            'per_page': per_page,
            'size': 'large'
        }
        
        response = await self._make_pexels_request('/search', params)
        if not response:
            return []
        
        images = []
        for photo in response.get('photos', []):
            # Filter by resolution
            width = photo.get('width', 0)
            height = photo.get('height', 0)
            
            if width >= self.min_resolution[0] and height >= self.min_resolution[1]:
                images.append({
                    'id': f"pexels_{photo['id']}",
                    'source': 'pexels',
                    'url': photo['src']['large'],
                    'download_url': photo['src']['original'],
                    'width': width,
                    'height': height,
                    'description': photo.get('alt', ''),
                    'author': photo.get('photographer', ''),
                    'pexels_url': photo.get('url', ''),
                    'query': query
                })
        
        return images
    
    async def _download_image(self, image_info: Dict[str, Any]) -> Optional[str]:
        """Download image from URL"""
        try:
            image_id = image_info['id']
            if image_id in self.downloaded_images:
                return None  # Already downloaded
            
            # Create filename
            source = image_info['source']
            original_id = image_id.replace(f"{source}_", "")
            filename = f"{source}_{original_id}.jpg"
            
            # Create source directory
            source_dir = self.output_dir / source
            source_dir.mkdir(exist_ok=True)
            
            file_path = source_dir / filename
            
            # Skip if file already exists
            if file_path.exists():
                self.downloaded_images.add(image_id)
                return str(file_path)
            
            # Download image
            download_url = image_info['download_url']
            
            async with aiohttp.ClientSession() as session:
                async with session.get(download_url) as response:
                    if response.status == 200:
                        content = await response.read()
                        
                        # Save image
                        with open(file_path, 'wb') as f:
                            f.write(content)
                        
                        # Save metadata
                        metadata_file = file_path.with_suffix('.json')
                        metadata = {
                            'image_id': image_id,
                            'source': source,
                            'original_url': image_info.get('url'),
                            'download_url': download_url,
                            'width': image_info.get('width'),
                            'height': image_info.get('height'),
                            'description': image_info.get('description'),
                            'author': image_info.get('author'),
                            'source_url': image_info.get(f'{source}_url'),
                            'query': image_info.get('query'),
                            'downloaded_at': datetime.now().isoformat(),
                            'file_path': str(file_path),
                            'file_size': len(content)
                        }
                        
                        if source == 'unsplash':
                            metadata['likes'] = image_info.get('likes', 0)
                        
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        self.downloaded_images.add(image_id)
                        return str(file_path)
                    else:
                        print(f"Failed to download {image_id}: HTTP {response.status}")
                        return None
        
        except Exception as e:
            print(f"Error downloading {image_info.get('id', 'unknown')}: {e}")
            return None
    
    def _assess_soccer_relevance(self, image_info: Dict[str, Any]) -> float:
        """Assess how relevant an image is for soccer training"""
        description = (image_info.get('description', '') + ' ' + 
                      image_info.get('query', '')).lower()
        
        # High relevance keywords
        high_relevance = ['soccer', 'football', 'ball', 'training', 'practice', 'drill']
        medium_relevance = ['sport', 'athlete', 'field', 'goal', 'kick', 'player']
        low_relevance = ['grass', 'green', 'stadium', 'team']
        
        score = 0.0
        
        # Count keyword matches
        for keyword in high_relevance:
            if keyword in description:
                score += 0.3
        
        for keyword in medium_relevance:
            if keyword in description:
                score += 0.2
        
        for keyword in low_relevance:
            if keyword in description:
                score += 0.1
        
        # Bonus for high resolution
        if (image_info.get('width', 0) >= 1920 and 
            image_info.get('height', 0) >= 1080):
            score += 0.1
        
        # Bonus for likes (Unsplash only)
        if image_info.get('likes', 0) > 50:
            score += 0.1
        
        return min(1.0, score)
    
    async def collect_stock_photos(self, max_images_per_source: int = 200) -> Dict[str, Any]:
        """Collect soccer images from stock photo APIs"""
        
        update_collection_status(
            "stock_photos",
            status=Status.RUNNING,
            progress=0.0,
            items_collected=0,
            target_items=max_images_per_source * 2,  # Both sources
            current_operation="Starting stock photo collection...",
            start_time=datetime.now()
        )
        
        start_time = datetime.now()
        downloaded_count = 0
        total_images_found = 0
        source_stats = {'unsplash': 0, 'pexels': 0}
        errors = []
        
        try:
            all_images = []
            
            # Search Unsplash
            if self.unsplash_key:
                update_collection_status(
                    "stock_photos",
                    progress=0.1,
                    current_operation="Searching Unsplash..."
                )
                
                for i, term in enumerate(self.search_terms):
                    try:
                        images = await self._search_unsplash(term, per_page=15)
                        all_images.extend(images)
                        total_images_found += len(images)
                        
                        # Rate limiting
                        await asyncio.sleep(1)  # Be nice to API
                        
                        progress = 0.1 + (i / len(self.search_terms)) * 0.2
                        update_collection_status(
                            "stock_photos",
                            progress=progress,
                            current_operation=f"Searching Unsplash: {term}"
                        )
                        
                    except Exception as e:
                        error_msg = f"Error searching Unsplash for '{term}': {e}"
                        errors.append(error_msg)
                        print(error_msg)
            
            # Search Pexels
            if self.pexels_key:
                update_collection_status(
                    "stock_photos",
                    progress=0.3,
                    current_operation="Searching Pexels..."
                )
                
                for i, term in enumerate(self.search_terms):
                    try:
                        images = await self._search_pexels(term, per_page=15)
                        all_images.extend(images)
                        total_images_found += len(images)
                        
                        # Rate limiting
                        await asyncio.sleep(0.5)  # Pexels allows more requests
                        
                        progress = 0.3 + (i / len(self.search_terms)) * 0.2
                        update_collection_status(
                            "stock_photos",
                            progress=progress,
                            current_operation=f"Searching Pexels: {term}"
                        )
                        
                    except Exception as e:
                        error_msg = f"Error searching Pexels for '{term}': {e}"
                        errors.append(error_msg)
                        print(error_msg)
            
            # Remove duplicates and sort by relevance
            unique_images = {}
            for image in all_images:
                image_id = image['id']
                if image_id not in unique_images:
                    image['relevance_score'] = self._assess_soccer_relevance(image)
                    unique_images[image_id] = image
            
            # Sort by relevance score (highest first)
            sorted_images = sorted(unique_images.values(), 
                                 key=lambda x: x['relevance_score'], 
                                 reverse=True)
            
            print(f"Found {len(sorted_images)} unique images, starting downloads...")
            
            # Download images
            max_total_images = min(len(sorted_images), max_images_per_source * 2)
            
            for i, image_info in enumerate(sorted_images[:max_total_images]):
                progress = 0.5 + (i / max_total_images) * 0.5
                update_collection_status(
                    "stock_photos",
                    progress=progress,
                    current_operation=f"Downloading {image_info['source']} image {i+1}/{max_total_images}"
                )
                
                try:
                    file_path = await self._download_image(image_info)
                    if file_path:
                        source_stats[image_info['source']] += 1
                        downloaded_count += 1
                        
                        print(f"Downloaded: {file_path} "
                              f"(Relevance: {image_info['relevance_score']:.2f})")
                
                except Exception as e:
                    error_msg = f"Error downloading {image_info['id']}: {e}"
                    errors.append(error_msg)
                    print(error_msg)
                
                # Save progress periodically
                if downloaded_count % 25 == 0:
                    self._save_downloaded_images()
                    update_collection_status(
                        "stock_photos",
                        items_collected=downloaded_count
                    )
                
                # Small delay between downloads
                await asyncio.sleep(0.1)
            
            # Final save
            self._save_downloaded_images()
            
            # Calculate final stats
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            summary = {
                'source': 'stock_photos',
                'started_at': start_time.isoformat(),
                'completed_at': end_time.isoformat(),
                'duration_seconds': round(duration, 2),
                'total_images_found': total_images_found,
                'images_downloaded': downloaded_count,
                'source_breakdown': source_stats,
                'search_terms': self.search_terms,
                'errors': errors,
                'success_rate': round(downloaded_count / max(total_images_found, 1) * 100, 2),
                'apis_used': {
                    'unsplash': self.unsplash_key is not None,
                    'pexels': self.pexels_key is not None
                }
            }
            
            # Save summary
            summary_file = self.output_dir / f"collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            update_collection_status(
                "stock_photos",
                status=Status.COMPLETED,
                progress=1.0,
                items_collected=downloaded_count,
                current_operation="Stock photo collection completed"
            )
            
            print(f"\n✅ Stock photo collection completed!")
            print(f"📊 Downloaded {downloaded_count} images in {duration:.1f} seconds")
            print(f"📈 Source breakdown: {source_stats}")
            
            return summary
            
        except Exception as e:
            error_msg = f"Stock photo collection failed: {e}"
            update_collection_status(
                "stock_photos",
                status=Status.FAILED,
                error_message=error_msg
            )
            raise

async def main():
    """Test stock photo scraper"""
    # Load config
    config_file = Path("automation_config.json")
    if not config_file.exists():
        print("❌ Please run setup_automation.py first")
        return
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    api_keys = config.get('api_keys', {})
    unsplash_key = api_keys.get('unsplash')
    pexels_key = api_keys.get('pexels')
    
    if not unsplash_key and not pexels_key:
        print("❌ No stock photo API keys configured")
        return
    
    scraper = StockPhotoScraper(
        unsplash_key=unsplash_key,
        pexels_key=pexels_key
    )
    
    try:
        result = await scraper.collect_stock_photos(max_images_per_source=100)
        print(f"✅ Collection completed: {result}")
    except Exception as e:
        print(f"❌ Collection failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())