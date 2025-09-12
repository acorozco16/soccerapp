#!/usr/bin/env python3
"""
Cleanup Script for Soccer Video Analysis App
Removes old videos, temporary files, and manages disk space
"""

import os
import sys
import time
import shutil
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

def get_file_size_mb(path: Path) -> float:
    """Get file size in MB"""
    if path.is_file():
        return path.stat().st_size / 1024 / 1024
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / 1024 / 1024
    return 0

def get_disk_usage() -> Dict[str, float]:
    """Get current disk usage statistics"""
    total, used, free = shutil.disk_usage('/')
    return {
        'total_gb': total / 1024 / 1024 / 1024,
        'used_gb': used / 1024 / 1024 / 1024,
        'free_gb': free / 1024 / 1024 / 1024,
        'used_percent': (used / total) * 100
    }

def cleanup_old_videos(days_old: int = 7, dry_run: bool = False) -> Dict[str, Any]:
    """Remove videos older than specified days"""
    uploads_dir = Path("uploads")
    cutoff_date = datetime.now() - timedelta(days=days_old)
    
    if not uploads_dir.exists():
        return {"error": "Uploads directory not found"}
    
    # Connect to database
    db_path = Path("backend/database.db")
    if not db_path.exists():
        return {"error": "Database not found"}
    
    deleted_videos = []
    total_space_freed = 0
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Find old videos
        cursor.execute("""
            SELECT id, filename, file_path, results_path, upload_time 
            FROM videos 
            WHERE upload_time < ?
        """, (cutoff_date.isoformat(),))
        
        old_videos = cursor.fetchall()
        
        for video_id, filename, file_path, results_path, upload_time in old_videos:
            try:
                # Calculate space to be freed
                space_freed = 0
                
                # Video file
                if file_path and Path(file_path).exists():
                    space_freed += get_file_size_mb(Path(file_path))
                    if not dry_run:
                        Path(file_path).unlink()
                
                # Results file
                if results_path and Path(results_path).exists():
                    space_freed += get_file_size_mb(Path(results_path))
                    if not dry_run:
                        Path(results_path).unlink()
                
                # Frame directory
                frames_dir = uploads_dir / "frames" / video_id
                if frames_dir.exists():
                    space_freed += get_file_size_mb(frames_dir)
                    if not dry_run:
                        shutil.rmtree(frames_dir)
                
                # Remove from database
                if not dry_run:
                    cursor.execute("DELETE FROM videos WHERE id = ?", (video_id,))
                
                deleted_videos.append({
                    'video_id': video_id,
                    'filename': filename,
                    'upload_time': upload_time,
                    'space_freed_mb': round(space_freed, 2)
                })
                
                total_space_freed += space_freed
                
            except Exception as e:
                print(f"Error deleting video {video_id}: {e}")
        
        if not dry_run:
            conn.commit()
        
        conn.close()
        
        return {
            'deleted_count': len(deleted_videos),
            'deleted_videos': deleted_videos,
            'total_space_freed_mb': round(total_space_freed, 2),
            'cutoff_date': cutoff_date.isoformat(),
            'dry_run': dry_run
        }
        
    except Exception as e:
        return {"error": f"Database operation failed: {e}"}

def cleanup_temp_files(dry_run: bool = False) -> Dict[str, Any]:
    """Remove temporary files and orphaned data"""
    uploads_dir = Path("uploads")
    
    if not uploads_dir.exists():
        return {"error": "Uploads directory not found"}
    
    # Connect to database to get valid video IDs
    db_path = Path("backend/database.db")
    valid_video_ids = set()
    
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM videos")
            valid_video_ids = {row[0] for row in cursor.fetchall()}
            conn.close()
        except Exception as e:
            print(f"Warning: Could not read database: {e}")
    
    removed_items = []
    total_space_freed = 0
    
    # Clean up orphaned frame directories
    frames_dir = uploads_dir / "frames"
    if frames_dir.exists():
        for frame_subdir in frames_dir.iterdir():
            if frame_subdir.is_dir() and frame_subdir.name not in valid_video_ids:
                space_freed = get_file_size_mb(frame_subdir)
                if not dry_run:
                    shutil.rmtree(frame_subdir)
                
                removed_items.append({
                    'type': 'orphaned_frames',
                    'path': str(frame_subdir),
                    'space_freed_mb': round(space_freed, 2)
                })
                total_space_freed += space_freed
    
    # Clean up temporary files (common temp patterns)
    temp_patterns = ['*.tmp', '*.temp', '*~', '.DS_Store', 'Thumbs.db']
    
    for pattern in temp_patterns:
        for temp_file in uploads_dir.rglob(pattern):
            if temp_file.is_file():
                space_freed = get_file_size_mb(temp_file)
                if not dry_run:
                    temp_file.unlink()
                
                removed_items.append({
                    'type': 'temp_file',
                    'path': str(temp_file),
                    'space_freed_mb': round(space_freed, 2)
                })
                total_space_freed += space_freed
    
    # Clean up empty directories
    for dir_path in uploads_dir.rglob('*'):
        if dir_path.is_dir() and not any(dir_path.iterdir()):
            if not dry_run:
                try:
                    dir_path.rmdir()
                    removed_items.append({
                        'type': 'empty_directory',
                        'path': str(dir_path),
                        'space_freed_mb': 0
                    })
                except OSError:
                    pass  # Directory not empty or permission error
    
    return {
        'removed_count': len(removed_items),
        'removed_items': removed_items,
        'total_space_freed_mb': round(total_space_freed, 2),
        'dry_run': dry_run
    }

def cleanup_failed_videos(dry_run: bool = False) -> Dict[str, Any]:
    """Remove videos that failed processing"""
    db_path = Path("backend/database.db")
    
    if not db_path.exists():
        return {"error": "Database not found"}
    
    removed_videos = []
    total_space_freed = 0
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Find failed videos
        cursor.execute("""
            SELECT id, filename, file_path, results_path, upload_time 
            FROM videos 
            WHERE status = 'ERROR'
        """)
        
        failed_videos = cursor.fetchall()
        uploads_dir = Path("uploads")
        
        for video_id, filename, file_path, results_path, upload_time in failed_videos:
            try:
                space_freed = 0
                
                # Video file
                if file_path and Path(file_path).exists():
                    space_freed += get_file_size_mb(Path(file_path))
                    if not dry_run:
                        Path(file_path).unlink()
                
                # Results file
                if results_path and Path(results_path).exists():
                    space_freed += get_file_size_mb(Path(results_path))
                    if not dry_run:
                        Path(results_path).unlink()
                
                # Frame directory
                frames_dir = uploads_dir / "frames" / video_id
                if frames_dir.exists():
                    space_freed += get_file_size_mb(frames_dir)
                    if not dry_run:
                        shutil.rmtree(frames_dir)
                
                # Remove from database
                if not dry_run:
                    cursor.execute("DELETE FROM videos WHERE id = ?", (video_id,))
                
                removed_videos.append({
                    'video_id': video_id,
                    'filename': filename,
                    'upload_time': upload_time,
                    'space_freed_mb': round(space_freed, 2)
                })
                
                total_space_freed += space_freed
                
            except Exception as e:
                print(f"Error removing failed video {video_id}: {e}")
        
        if not dry_run:
            conn.commit()
        
        conn.close()
        
        return {
            'removed_count': len(removed_videos),
            'removed_videos': removed_videos,
            'total_space_freed_mb': round(total_space_freed, 2),
            'dry_run': dry_run
        }
        
    except Exception as e:
        return {"error": f"Database operation failed: {e}"}

def get_storage_stats() -> Dict[str, Any]:
    """Get current storage statistics"""
    uploads_dir = Path("uploads")
    
    if not uploads_dir.exists():
        return {"error": "Uploads directory not found"}
    
    stats = {
        'disk_usage': get_disk_usage(),
        'upload_directory': {
            'total_size_mb': round(get_file_size_mb(uploads_dir), 2),
            'subdirectories': {}
        }
    }
    
    # Analyze subdirectories
    for subdir in ['raw', 'processed', 'frames']:
        subdir_path = uploads_dir / subdir
        if subdir_path.exists():
            file_count = len([f for f in subdir_path.rglob('*') if f.is_file()])
            stats['upload_directory']['subdirectories'][subdir] = {
                'size_mb': round(get_file_size_mb(subdir_path), 2),
                'file_count': file_count
            }
    
    # Database stats
    db_path = Path("backend/database.db")
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM videos")
            total_videos = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM videos WHERE status = 'COMPLETED'")
            completed_videos = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM videos WHERE status = 'ERROR'")
            failed_videos = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM videos WHERE status IN ('PENDING', 'PROCESSING')")
            pending_videos = cursor.fetchone()[0]
            
            conn.close()
            
            stats['database'] = {
                'size_mb': round(get_file_size_mb(db_path), 2),
                'total_videos': total_videos,
                'completed_videos': completed_videos,
                'failed_videos': failed_videos,
                'pending_videos': pending_videos
            }
            
        except Exception as e:
            stats['database'] = {"error": f"Could not read database: {e}"}
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Soccer App Cleanup Tool")
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--days-old', type=int, default=7,
                       help='Delete videos older than N days (default: 7)')
    parser.add_argument('--failed-only', action='store_true',
                       help='Only remove failed videos')
    parser.add_argument('--temp-only', action='store_true',
                       help='Only clean temporary files')
    parser.add_argument('--stats', action='store_true',
                       help='Show storage statistics only')
    
    args = parser.parse_args()
    
    print("🧹 Soccer App Cleanup Tool")
    print("=" * 40)
    
    # Show current stats
    if args.stats:
        stats = get_storage_stats()
        
        disk = stats.get('disk_usage', {})
        print(f"\n📊 Disk Usage:")
        print(f"  Total: {disk.get('total_gb', 0):.1f} GB")
        print(f"  Used: {disk.get('used_gb', 0):.1f} GB ({disk.get('used_percent', 0):.1f}%)")
        print(f"  Free: {disk.get('free_gb', 0):.1f} GB")
        
        uploads = stats.get('upload_directory', {})
        print(f"\n📁 Upload Directory: {uploads.get('total_size_mb', 0):.1f} MB")
        
        for subdir, info in uploads.get('subdirectories', {}).items():
            print(f"  {subdir}: {info.get('size_mb', 0):.1f} MB ({info.get('file_count', 0)} files)")
        
        db = stats.get('database', {})
        if 'error' not in db:
            print(f"\n🗄️ Database: {db.get('size_mb', 0):.1f} MB")
            print(f"  Total videos: {db.get('total_videos', 0)}")
            print(f"  Completed: {db.get('completed_videos', 0)}")
            print(f"  Failed: {db.get('failed_videos', 0)}")
            print(f"  Pending: {db.get('pending_videos', 0)}")
        
        return
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be deleted")
    
    total_space_freed = 0
    
    # Clean temporary files
    if args.temp_only or not (args.failed_only):
        print(f"\n🧽 Cleaning temporary files...")
        temp_result = cleanup_temp_files(dry_run=args.dry_run)
        
        if 'error' in temp_result:
            print(f"❌ Error: {temp_result['error']}")
        else:
            print(f"  Removed {temp_result['removed_count']} items")
            print(f"  Space freed: {temp_result['total_space_freed_mb']:.1f} MB")
            total_space_freed += temp_result['total_space_freed_mb']
    
    # Clean failed videos
    if args.failed_only or not (args.temp_only):
        print(f"\n❌ Cleaning failed videos...")
        failed_result = cleanup_failed_videos(dry_run=args.dry_run)
        
        if 'error' in failed_result:
            print(f"❌ Error: {failed_result['error']}")
        else:
            print(f"  Removed {failed_result['removed_count']} failed videos")
            print(f"  Space freed: {failed_result['total_space_freed_mb']:.1f} MB")
            total_space_freed += failed_result['total_space_freed_mb']
    
    # Clean old videos (unless only doing specific cleanups)
    if not (args.temp_only or args.failed_only):
        print(f"\n🗓️ Cleaning videos older than {args.days_old} days...")
        old_result = cleanup_old_videos(days_old=args.days_old, dry_run=args.dry_run)
        
        if 'error' in old_result:
            print(f"❌ Error: {old_result['error']}")
        else:
            print(f"  Removed {old_result['deleted_count']} old videos")
            print(f"  Space freed: {old_result['total_space_freed_mb']:.1f} MB")
            total_space_freed += old_result['total_space_freed_mb']
    
    print(f"\n✅ Cleanup complete!")
    print(f"💾 Total space freed: {total_space_freed:.1f} MB")
    
    if args.dry_run:
        print("🔍 This was a dry run - no files were actually deleted")
        print("   Run without --dry-run to perform the cleanup")

if __name__ == "__main__":
    main()