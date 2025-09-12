#!/usr/bin/env python3
"""
Research Dataset Downloader
Downloads established soccer/sports datasets (SoccerNet, COCO, Open Images)
"""

import os
import json
import asyncio
import aiohttp
import zipfile
import tarfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib
import subprocess

import sys
sys.path.append(str(Path(__file__).parent.parent / "automation"))
from training_status import update_collection_status, Status

class DatasetDownloader:
    def __init__(self, output_dir: str = "training_data/collected_data/datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Available datasets
        self.datasets = {
            'coco_sports': {
                'name': 'COCO Sports Ball',
                'description': 'Sports ball annotations from MS COCO dataset',
                'url': 'http://images.cocodataset.org/zips/train2017.zip',
                'annotations_url': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                'category_id': 37,  # Sports ball category ID in COCO
                'size_gb': 18.0,
                'type': 'detection'
            },
            'open_images_sports': {
                'name': 'Open Images Sports',
                'description': 'Sports-related images from Open Images dataset',
                'base_url': 'https://storage.googleapis.com/openimages/web/',
                'size_gb': 2.0,
                'type': 'detection'
            },
            'soccer_video_clips': {
                'name': 'Soccer Video Clips',
                'description': 'Curated soccer video clips for training',
                'url': 'https://github.com/SoccerNet/sn-gamestate',  # Example
                'size_gb': 5.0,
                'type': 'video'
            }
        }
        
        self.downloaded_datasets = self._load_downloaded_datasets()
    
    def _load_downloaded_datasets(self) -> set:
        """Load list of already downloaded datasets"""
        downloaded_file = self.output_dir / "downloaded_datasets.json"
        if downloaded_file.exists():
            try:
                with open(downloaded_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('datasets', []))
            except Exception as e:
                print(f"Warning: Could not load downloaded datasets list: {e}")
        return set()
    
    def _save_downloaded_datasets(self):
        """Save list of downloaded datasets"""
        downloaded_file = self.output_dir / "downloaded_datasets.json"
        try:
            data = {
                'datasets': list(self.downloaded_datasets),
                'last_updated': datetime.now().isoformat(),
                'total_count': len(self.downloaded_datasets)
            }
            with open(downloaded_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving downloaded datasets list: {e}")
    
    def _check_disk_space(self, required_gb: float) -> bool:
        """Check if sufficient disk space is available"""
        try:
            import shutil
            free_bytes = shutil.disk_usage(self.output_dir).free
            free_gb = free_bytes / (1024**3)
            return free_gb >= required_gb * 1.2  # 20% buffer
        except Exception:
            return True  # Assume OK if can't check
    
    async def _download_file(self, url: str, output_path: Path, 
                           progress_callback=None) -> bool:
        """Download file with progress tracking"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        print(f"Failed to download: HTTP {response.status}")
                        return False
                    
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    with open(output_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if progress_callback and total_size > 0:
                                progress = downloaded / total_size
                                progress_callback(progress)
            
            return True
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def _extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """Extract archive file"""
        try:
            extract_to.mkdir(parents=True, exist_ok=True)
            
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix.lower() in ['.tar', '.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                print(f"Unsupported archive format: {archive_path.suffix}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error extracting {archive_path}: {e}")
            return False
    
    async def _download_coco_sports(self) -> Dict[str, Any]:
        """Download COCO dataset and filter for sports balls"""
        dataset_info = self.datasets['coco_sports']
        dataset_dir = self.output_dir / "coco_sports"
        dataset_dir.mkdir(exist_ok=True)
        
        # Check disk space
        if not self._check_disk_space(dataset_info['size_gb']):
            raise Exception(f"Insufficient disk space for COCO dataset ({dataset_info['size_gb']} GB required)")
        
        results = {
            'dataset': 'coco_sports',
            'started_at': datetime.now().isoformat(),
            'images_processed': 0,
            'sports_ball_images': 0,
            'annotations_created': 0
        }
        
        # Download images
        images_zip = dataset_dir / "train2017.zip"
        if not images_zip.exists():
            print("Downloading COCO training images...")
            
            def progress_callback(progress):
                update_collection_status(
                    "datasets",
                    progress=progress * 0.6,  # 60% for images
                    current_operation=f"Downloading COCO images: {progress*100:.1f}%"
                )
            
            success = await self._download_file(
                dataset_info['url'], 
                images_zip, 
                progress_callback
            )
            
            if not success:
                raise Exception("Failed to download COCO images")
        
        # Download annotations
        annotations_zip = dataset_dir / "annotations_trainval2017.zip"
        if not annotations_zip.exists():
            print("Downloading COCO annotations...")
            
            def progress_callback(progress):
                update_collection_status(
                    "datasets",
                    progress=0.6 + progress * 0.2,  # 20% for annotations
                    current_operation=f"Downloading COCO annotations: {progress*100:.1f}%"
                )
            
            success = await self._download_file(
                dataset_info['annotations_url'], 
                annotations_zip, 
                progress_callback
            )
            
            if not success:
                raise Exception("Failed to download COCO annotations")
        
        # Extract files
        print("Extracting COCO dataset...")
        update_collection_status(
            "datasets",
            progress=0.8,
            current_operation="Extracting COCO dataset..."
        )
        
        if not self._extract_archive(images_zip, dataset_dir):
            raise Exception("Failed to extract COCO images")
        
        if not self._extract_archive(annotations_zip, dataset_dir):
            raise Exception("Failed to extract COCO annotations")
        
        # Process annotations to filter sports balls
        print("Processing COCO annotations for sports balls...")
        update_collection_status(
            "datasets",
            progress=0.9,
            current_operation="Filtering sports ball annotations..."
        )
        
        try:
            annotations_file = dataset_dir / "annotations" / "instances_train2017.json"
            
            if annotations_file.exists():
                with open(annotations_file, 'r') as f:
                    coco_data = json.load(f)
                
                # Filter for sports ball category (ID 37)
                sports_ball_annotations = [
                    ann for ann in coco_data['annotations'] 
                    if ann['category_id'] == dataset_info['category_id']
                ]
                
                # Get unique image IDs with sports balls
                sports_ball_image_ids = set(ann['image_id'] for ann in sports_ball_annotations)
                
                # Filter images
                sports_ball_images = [
                    img for img in coco_data['images'] 
                    if img['id'] in sports_ball_image_ids
                ]
                
                # Create filtered dataset
                filtered_data = {
                    'info': coco_data['info'],
                    'licenses': coco_data['licenses'],
                    'categories': [cat for cat in coco_data['categories'] if cat['id'] == 37],
                    'images': sports_ball_images,
                    'annotations': sports_ball_annotations
                }
                
                # Save filtered dataset
                filtered_file = dataset_dir / "sports_ball_annotations.json"
                with open(filtered_file, 'w') as f:
                    json.dump(filtered_data, f, indent=2)
                
                results['images_processed'] = len(coco_data['images'])
                results['sports_ball_images'] = len(sports_ball_images)
                results['annotations_created'] = len(sports_ball_annotations)
                
                print(f"Filtered {len(sports_ball_images)} images with sports balls from {len(coco_data['images'])} total images")
            
        except Exception as e:
            print(f"Error processing COCO annotations: {e}")
        
        results['completed_at'] = datetime.now().isoformat()
        return results
    
    async def _download_open_images_sports(self) -> Dict[str, Any]:
        """Download Open Images sports subset"""
        dataset_dir = self.output_dir / "open_images_sports"
        dataset_dir.mkdir(exist_ok=True)
        
        results = {
            'dataset': 'open_images_sports',
            'started_at': datetime.now().isoformat(),
            'images_downloaded': 0,
            'annotations_created': 0
        }
        
        # This would require using the Open Images API or downloading specific class files
        # For now, we'll create a placeholder implementation
        
        update_collection_status(
            "datasets",
            progress=0.5,
            current_operation="Processing Open Images sports data..."
        )
        
        # In a real implementation, you would:
        # 1. Download the Open Images class descriptions
        # 2. Filter for sports-related classes (Ball, Football, Soccer ball, etc.)
        # 3. Download the corresponding image URLs and annotations
        # 4. Download the actual images
        
        # For now, create a sample structure
        sample_info = {
            'name': 'Open Images Sports Sample',
            'description': 'Placeholder for Open Images sports data',
            'classes': ['Ball', 'Football', 'Person'],
            'note': 'This is a placeholder. Implement actual Open Images download logic.'
        }
        
        info_file = dataset_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(sample_info, f, indent=2)
        
        results['completed_at'] = datetime.now().isoformat()
        results['note'] = 'Placeholder implementation - requires Open Images API integration'
        
        return results
    
    async def _download_soccer_videos(self) -> Dict[str, Any]:
        """Download curated soccer video clips"""
        dataset_dir = self.output_dir / "soccer_videos"
        dataset_dir.mkdir(exist_ok=True)
        
        results = {
            'dataset': 'soccer_videos',
            'started_at': datetime.now().isoformat(),
            'videos_downloaded': 0
        }
        
        # This would download curated soccer video clips
        # Implementation would depend on the actual source
        
        update_collection_status(
            "datasets",
            progress=0.7,
            current_operation="Downloading soccer video clips..."
        )
        
        # Placeholder implementation
        sample_info = {
            'name': 'Soccer Video Clips',
            'description': 'Curated soccer training video clips',
            'format': 'MP4',
            'resolution': '720p',
            'note': 'Placeholder implementation - add actual video sources'
        }
        
        info_file = dataset_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(sample_info, f, indent=2)
        
        results['completed_at'] = datetime.now().isoformat()
        results['note'] = 'Placeholder implementation - requires actual video sources'
        
        return results
    
    async def download_datasets(self, dataset_names: List[str] = None) -> Dict[str, Any]:
        """Download specified datasets"""
        if dataset_names is None:
            dataset_names = list(self.datasets.keys())
        
        update_collection_status(
            "datasets",
            status=Status.RUNNING,
            progress=0.0,
            items_collected=0,
            target_items=len(dataset_names),
            current_operation="Starting dataset downloads...",
            start_time=datetime.now()
        )
        
        start_time = datetime.now()
        results = {
            'started_at': start_time.isoformat(),
            'datasets_requested': dataset_names,
            'datasets_completed': [],
            'datasets_failed': [],
            'total_images': 0,
            'total_annotations': 0
        }
        
        try:
            for i, dataset_name in enumerate(dataset_names):
                if dataset_name in self.downloaded_datasets:
                    print(f"Dataset {dataset_name} already downloaded, skipping...")
                    results['datasets_completed'].append(dataset_name)
                    continue
                
                if dataset_name not in self.datasets:
                    print(f"Unknown dataset: {dataset_name}")
                    results['datasets_failed'].append(dataset_name)
                    continue
                
                try:
                    print(f"Downloading dataset: {dataset_name}")
                    
                    # Download specific dataset
                    if dataset_name == 'coco_sports':
                        dataset_result = await self._download_coco_sports()
                    elif dataset_name == 'open_images_sports':
                        dataset_result = await self._download_open_images_sports()
                    elif dataset_name == 'soccer_video_clips':
                        dataset_result = await self._download_soccer_videos()
                    else:
                        raise Exception(f"No download method for {dataset_name}")
                    
                    # Update results
                    results['datasets_completed'].append(dataset_name)
                    results['total_images'] += dataset_result.get('images_processed', 0)
                    results['total_images'] += dataset_result.get('sports_ball_images', 0)
                    results['total_annotations'] += dataset_result.get('annotations_created', 0)
                    
                    # Mark as downloaded
                    self.downloaded_datasets.add(dataset_name)
                    self._save_downloaded_datasets()
                    
                    # Save individual dataset result
                    result_file = self.output_dir / dataset_name / "download_result.json"
                    result_file.parent.mkdir(exist_ok=True)
                    with open(result_file, 'w') as f:
                        json.dump(dataset_result, f, indent=2)
                    
                except Exception as e:
                    error_msg = f"Failed to download {dataset_name}: {e}"
                    print(error_msg)
                    results['datasets_failed'].append({
                        'dataset': dataset_name,
                        'error': str(e)
                    })
                
                # Update progress
                progress = (i + 1) / len(dataset_names)
                update_collection_status(
                    "datasets",
                    progress=progress,
                    items_collected=len(results['datasets_completed']),
                    current_operation=f"Completed {i+1}/{len(dataset_names)} datasets"
                )
            
            # Final summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results['completed_at'] = end_time.isoformat()
            results['duration_seconds'] = round(duration, 2)
            results['success_rate'] = len(results['datasets_completed']) / len(dataset_names) * 100
            
            # Save summary
            summary_file = self.output_dir / f"download_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            update_collection_status(
                "datasets",
                status=Status.COMPLETED,
                progress=1.0,
                items_collected=len(results['datasets_completed']),
                current_operation="Dataset downloads completed"
            )
            
            print(f"\n✅ Dataset downloads completed!")
            print(f"📊 Completed: {len(results['datasets_completed'])}")
            print(f"📊 Failed: {len(results['datasets_failed'])}")
            print(f"📊 Total images: {results['total_images']}")
            print(f"📊 Total annotations: {results['total_annotations']}")
            
            return results
            
        except Exception as e:
            error_msg = f"Dataset download failed: {e}"
            update_collection_status(
                "datasets",
                status=Status.FAILED,
                error_message=error_msg
            )
            raise

async def main():
    """Test dataset downloader"""
    downloader = DatasetDownloader()
    
    try:
        # Download only COCO sports for testing (smaller subset)
        result = await downloader.download_datasets(['open_images_sports', 'soccer_video_clips'])
        print(f"✅ Download completed: {result}")
    except Exception as e:
        print(f"❌ Download failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())