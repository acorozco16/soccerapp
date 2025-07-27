#!/usr/bin/env python3
"""
Comprehensive Data Collection System
Runs all data sources in parallel for maximum efficiency
"""

import os
import json
import asyncio
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import concurrent.futures
import logging

# Add scrapers to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "scrapers"))
sys.path.append(str(Path(__file__).parent))

from youtube_scraper import YouTubeSoccerScraper
from instagram_scraper import InstagramSoccerScraper
from stock_api_scraper import StockPhotoScraper
from dataset_downloader import DatasetDownloader
from training_status import get_status_manager, update_collection_status, Status

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_data/logs/data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveDataCollector:
    def __init__(self, config_file: str = "automation_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.status_manager = get_status_manager()
        
        # Output directory
        self.output_dir = Path("training_data/collected_data")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collection stats
        self.collection_stats = {
            'started_at': None,
            'completed_at': None,
            'duration_seconds': 0,
            'sources_attempted': [],
            'sources_completed': [],
            'sources_failed': [],
            'total_items_collected': 0,
            'items_by_source': {},
            'quality_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'errors': []
        }
        
        # Initialize scrapers
        self._initialize_scrapers()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.config_file.exists():
            raise Exception(f"Configuration file not found: {self.config_file}")
        
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load configuration: {e}")
    
    def _initialize_scrapers(self):
        """Initialize all available scrapers based on configuration"""
        api_keys = self.config.get('api_keys', {})
        
        # YouTube scraper (multiple API keys for parallel processing)
        youtube_keys = api_keys.get('youtube', [])
        if youtube_keys:
            self.youtube_scrapers = []
            for i, key in enumerate(youtube_keys):
                scraper = YouTubeSoccerScraper(
                    
                    output_dir=str(self.output_dir / "youtube" / f"scraper_{i}")
                )
                scraper.scraper_id = i
                self.youtube_scrapers.append(scraper)
            logger.info(f"Initialized {len(self.youtube_scrapers)} YouTube scrapers")
        else:
            self.youtube_scrapers = []
            logger.warning("No YouTube API keys configured")
        
        # Instagram scraper
        instagram_token = api_keys.get('instagram')
        if instagram_token:
            self.instagram_scraper = InstagramSoccerScraper(
                access_token=instagram_token,
                output_dir=str(self.output_dir / "instagram")
            )
            logger.info("Instagram scraper initialized")
        else:
            self.instagram_scraper = None
            logger.warning("Instagram access token not configured")
        
        # Stock photo scrapers
        unsplash_key = api_keys.get('unsplash')
        pexels_key = api_keys.get('pexels')
        if unsplash_key or pexels_key:
            self.stock_scraper = StockPhotoScraper(
                unsplash_key=unsplash_key,
                pexels_key=pexels_key,
                output_dir=str(self.output_dir / "stock_photos")
            )
            logger.info("Stock photo scraper initialized")
        else:
            self.stock_scraper = None
            logger.warning("No stock photo API keys configured")
        
        # Dataset downloader
        self.dataset_downloader = DatasetDownloader(
            output_dir=str(self.output_dir / "datasets")
        )
        logger.info("Dataset downloader initialized")
    
    async def _collect_youtube_parallel(self, videos_per_scraper: int) -> Dict[str, Any]:
        """Run YouTube scrapers in parallel"""
        if not self.youtube_scrapers:
            return {'source': 'youtube', 'status': 'skipped', 'reason': 'No API keys configured'}
        
        logger.info(f"Starting parallel YouTube collection with {len(self.youtube_scrapers)} scrapers")
        
        # Divide search terms among scrapers
        all_search_terms = [
            "soccer ball juggling training",
            "football skills practice", 
            "soccer dribbling drills",
            "football ball control",
            "soccer first touch training",
            "football passing practice",
            "soccer shooting drills",
            "football goalkeeper training",
            "youth soccer training",
            "professional football skills",
            "soccer tricks tutorial",
            "football fitness training"
        ]
        
        # Distribute search terms among scrapers
        terms_per_scraper = len(all_search_terms) // len(self.youtube_scrapers)
        if terms_per_scraper == 0:
            terms_per_scraper = 1
        
        tasks = []
        for i, scraper in enumerate(self.youtube_scrapers):
            start_idx = i * terms_per_scraper
            end_idx = start_idx + terms_per_scraper
            if i == len(self.youtube_scrapers) - 1:  # Last scraper gets remaining terms
                scraper_terms = all_search_terms[start_idx:]
            else:
                scraper_terms = all_search_terms[start_idx:end_idx]
            
            # Create task for this scraper
            task = self._run_youtube_scraper(scraper, scraper_terms, videos_per_scraper)
            tasks.append(task)
        
        # Run all scrapers in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        total_videos = 0
        total_frames = 0
        combined_errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = f"YouTube scraper {i} failed: {result}"
                combined_errors.append(error_msg)
                logger.error(error_msg)
            elif isinstance(result, dict):
                total_videos += result.get('total_videos_processed', 0)
                total_frames += result.get('total_frames_extracted', 0)
                combined_errors.extend(result.get('errors', []))
        
        return {
            'source': 'youtube',
            'status': 'completed',
            'scrapers_used': len(self.youtube_scrapers),
            'total_videos_collected': total_videos,
            'total_frames_extracted': total_frames,
            'errors': combined_errors
        }
    
    async def _run_youtube_scraper(self, scraper, search_terms: List[str], max_videos: int) -> Dict[str, Any]:
        """Run individual YouTube scraper"""
        try:
            # Update the scraper's search terms
            scraper.search_terms = search_terms
            
            result = await scraper.scrape_soccer_videos(
                max_videos_per_term=max_videos // len(search_terms),
                max_total_videos=max_videos
            )
            
            logger.info(f"YouTube scraper {scraper.scraper_id} completed: {result.get('total_videos_processed', 0)} videos")
            return result
            
        except Exception as e:
            logger.error(f"YouTube scraper {scraper.scraper_id} failed: {e}")
            raise
    
    def _check_resource_limits(self) -> Dict[str, Any]:
        """Check system resources before starting collection"""
        try:
            # Disk space check
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            
            # Memory check
            memory = psutil.virtual_memory()
            available_memory_gb = memory.available / (1024**3)
            
            # CPU check
            cpu_count = psutil.cpu_count()
            
            training_config = self.config.get('training', {})
            max_disk_gb = training_config.get('max_disk_usage_gb', 50)
            
            warnings = []
            
            if free_gb < max_disk_gb:
                warnings.append(f"Low disk space: {free_gb:.1f} GB available, {max_disk_gb} GB configured as limit")
            
            if available_memory_gb < 2.0:
                warnings.append(f"Low memory: {available_memory_gb:.1f} GB available")
            
            return {
                'free_disk_gb': round(free_gb, 2),
                'available_memory_gb': round(available_memory_gb, 2),
                'cpu_count': cpu_count,
                'warnings': warnings,
                'can_proceed': len(warnings) == 0 or free_gb > 5.0  # Minimum 5GB
            }
            
        except Exception as e:
            logger.warning(f"Could not check system resources: {e}")
            return {'can_proceed': True, 'warnings': []}
    
    async def collect_all_data(self, target_images: int = 5000, 
                             enable_sources: List[str] = None) -> Dict[str, Any]:
        """Collect data from all configured sources in parallel"""
        
        self.collection_stats['started_at'] = datetime.now().isoformat()
        start_time = time.time()
        
        logger.info("🚀 Starting comprehensive data collection")
        logger.info(f"Target: {target_images} total items")
        
        # Check system resources
        resource_check = self._check_resource_limits()
        if not resource_check['can_proceed']:
            raise Exception(f"System resource check failed: {resource_check['warnings']}")
        
        if resource_check['warnings']:
            for warning in resource_check['warnings']:
                logger.warning(warning)
        
        # Determine which sources to use
        if enable_sources is None:
            enable_sources = ['youtube', 'instagram', 'stock_photos', 'datasets']
        
        # Calculate target per source
        active_sources = []
        if 'youtube' in enable_sources and self.youtube_scrapers:
            active_sources.append('youtube')
        if 'instagram' in enable_sources and self.instagram_scraper:
            active_sources.append('instagram')
        if 'stock_photos' in enable_sources and self.stock_scraper:
            active_sources.append('stock_photos')
        if 'datasets' in enable_sources:
            active_sources.append('datasets')
        
        if not active_sources:
            raise Exception("No data sources available or enabled")
        
        logger.info(f"Active sources: {active_sources}")
        
        # Distribute target across sources
        # YouTube gets 40%, stock photos 30%, Instagram 20%, datasets 10%
        source_weights = {
            'youtube': 0.4,
            'instagram': 0.2,
            'stock_photos': 0.3,
            'datasets': 0.1
        }
        
        source_targets = {}
        for source in active_sources:
            weight = source_weights.get(source, 1.0 / len(active_sources))
            source_targets[source] = max(1, int(target_images * weight))
        
        logger.info(f"Source targets: {source_targets}")
        
        # Create collection tasks
        tasks = []
        
        # YouTube collection (parallel scrapers)
        if 'youtube' in active_sources:
            youtube_target = source_targets['youtube']
            videos_per_scraper = max(1, youtube_target // len(self.youtube_scrapers))
            task = asyncio.create_task(
                self._collect_youtube_parallel(videos_per_scraper),
                name="youtube_collection"
            )
            tasks.append(task)
            self.collection_stats['sources_attempted'].append('youtube')
        
        # Instagram collection
        if 'instagram' in active_sources:
            task = asyncio.create_task(
                self.instagram_scraper.scrape_soccer_content(max_posts=source_targets['instagram']),
                name="instagram_collection"
            )
            tasks.append(task)
            self.collection_stats['sources_attempted'].append('instagram')
        
        # Stock photos collection
        if 'stock_photos' in active_sources:
            stock_target = source_targets['stock_photos']
            task = asyncio.create_task(
                self.stock_scraper.collect_stock_photos(max_images_per_source=stock_target // 2),
                name="stock_photos_collection"
            )
            tasks.append(task)
            self.collection_stats['sources_attempted'].append('stock_photos')
        
        # Dataset downloads (run separately as they're large)
        if 'datasets' in active_sources:
            # Start datasets first as they take longest
            dataset_task = asyncio.create_task(
                self.dataset_downloader.download_datasets(['open_images_sports', 'soccer_video_clips']),
                name="dataset_downloads"
            )
            tasks.append(dataset_task)
            self.collection_stats['sources_attempted'].append('datasets')
        
        # Monitor progress
        progress_task = asyncio.create_task(self._monitor_progress(tasks))
        
        # Run all collection tasks
        logger.info(f"Starting {len(tasks)} collection tasks in parallel...")
        
        try:
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Cancel progress monitoring
            progress_task.cancel()
            
            # Process results
            for i, result in enumerate(results):
                task_name = tasks[i].get_name()
                source_name = task_name.replace('_collection', '').replace('_downloads', '')
                
                if isinstance(result, Exception):
                    error_msg = f"{source_name} collection failed: {result}"
                    self.collection_stats['errors'].append(error_msg)
                    self.collection_stats['sources_failed'].append(source_name)
                    logger.error(error_msg)
                else:
                    self.collection_stats['sources_completed'].append(source_name)
                    
                    # Extract stats from result
                    if isinstance(result, dict):
                        items_collected = (
                            result.get('total_videos_collected', 0) +
                            result.get('posts_downloaded', 0) +
                            result.get('images_downloaded', 0) +
                            result.get('total_images', 0)
                        )
                        
                        self.collection_stats['items_by_source'][source_name] = items_collected
                        self.collection_stats['total_items_collected'] += items_collected
                        
                        # Quality distribution
                        if 'quality_distribution' in result:
                            quality_dist = result['quality_distribution']
                            if isinstance(quality_dist, dict):
                                for quality, count in quality_dist.items():
                                    if quality in self.collection_stats['quality_distribution']:
                                        self.collection_stats['quality_distribution'][quality] += count
                    
                    logger.info(f"✅ {source_name} collection completed: {result}")
            
            # Final statistics
            end_time = time.time()
            self.collection_stats['completed_at'] = datetime.now().isoformat()
            self.collection_stats['duration_seconds'] = round(end_time - start_time, 2)
            
            # Save collection summary
            summary_file = self.output_dir / f"collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w') as f:
                json.dump(self.collection_stats, f, indent=2)
            
            # Update final status
            total_sources = len(self.collection_stats['sources_attempted'])
            completed_sources = len(self.collection_stats['sources_completed'])
            
            final_status = Status.COMPLETED if completed_sources == total_sources else Status.FAILED
            
            update_collection_status(
                "comprehensive",
                status=final_status,
                progress=1.0,
                items_collected=self.collection_stats['total_items_collected'],
                current_operation="Data collection completed"
            )
            
            # Print summary
            self._print_collection_summary()
            
            return self.collection_stats
            
        except Exception as e:
            logger.error(f"Collection failed: {e}")
            
            update_collection_status(
                "comprehensive",
                status=Status.FAILED,
                error_message=str(e)
            )
            
            raise
    
    async def _monitor_progress(self, tasks: List[asyncio.Task]):
        """Monitor and report progress of all collection tasks"""
        try:
            while not all(task.done() for task in tasks):
                # Get current status from status manager
                overall_status = self.status_manager.get_overall_status()
                
                # Update comprehensive collection status
                collection_info = overall_status.get('collection', {})
                active_sources = collection_info.get('active_sources', 0)
                total_collected = collection_info.get('total_items_collected', 0)
                overall_progress = collection_info.get('overall_progress', 0.0)
                
                update_collection_status(
                    "comprehensive",
                    progress=overall_progress,
                    items_collected=total_collected,
                    current_operation=f"Running {active_sources} collection sources..."
                )
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
        except asyncio.CancelledError:
            pass
    
    def _print_collection_summary(self):
        """Print formatted collection summary"""
        stats = self.collection_stats
        
        print("\n" + "="*60)
        print("📊 DATA COLLECTION SUMMARY")
        print("="*60)
        
        # Overall stats
        print(f"\n⏱️  Duration: {stats['duration_seconds']:.1f} seconds")
        print(f"📈 Total Items Collected: {stats['total_items_collected']}")
        print(f"✅ Sources Completed: {len(stats['sources_completed'])}/{len(stats['sources_attempted'])}")
        
        # Per-source breakdown
        if stats['items_by_source']:
            print(f"\n📂 Items by Source:")
            for source, count in stats['items_by_source'].items():
                print(f"   {source}: {count}")
        
        # Quality distribution
        quality_dist = stats['quality_distribution']
        total_quality = sum(quality_dist.values())
        if total_quality > 0:
            print(f"\n🎯 Quality Distribution:")
            for quality, count in quality_dist.items():
                percentage = (count / total_quality) * 100
                print(f"   {quality}: {count} ({percentage:.1f}%)")
        
        # Errors
        if stats['errors']:
            print(f"\n⚠️  Errors ({len(stats['errors'])}):")
            for error in stats['errors'][:5]:  # Show first 5 errors
                print(f"   • {error}")
            if len(stats['errors']) > 5:
                print(f"   ... and {len(stats['errors']) - 5} more")
        
        # Success rate
        success_rate = (len(stats['sources_completed']) / max(len(stats['sources_attempted']), 1)) * 100
        print(f"\n🎉 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 75:
            print("✨ Collection completed successfully!")
        elif success_rate >= 50:
            print("⚠️ Collection partially successful")
        else:
            print("❌ Collection had significant issues")

async def main():
    parser = argparse.ArgumentParser(description="Comprehensive Soccer Data Collection")
    parser.add_argument('--target-images', type=int, default=5000,
                       help='Total target number of images/videos to collect')
    parser.add_argument('--sources', nargs='+', 
                       choices=['youtube', 'instagram', 'stock_photos', 'datasets'],
                       help='Specific sources to use (default: all available)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run with minimal data for testing')
    
    args = parser.parse_args()
    
    # Adjust for quick test
    if args.quick_test:
        args.target_images = 50
        args.sources = ['stock_photos']  # Fastest source for testing
        logger.info("Running in quick test mode")
    
    try:
        collector = ComprehensiveDataCollector()
        
        result = await collector.collect_all_data(
            target_images=args.target_images,
            enable_sources=args.sources
        )
        
        logger.info("🎉 Data collection completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Data collection failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)