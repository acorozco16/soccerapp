#!/usr/bin/env python3
"""
Backup Script for Soccer Video Analysis App
Creates backups of database and important video data
"""

import os
import sys
import shutil
import sqlite3
import zipfile
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

def create_database_backup(backup_dir: Path, include_videos: bool = False) -> Dict[str, Any]:
    """Create backup of database and optionally video files"""
    db_path = Path("backend/database.db")
    
    if not db_path.exists():
        return {"error": "Database file not found"}
    
    backup_name = f"soccer_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_path = backup_dir / f"{backup_name}.zip"
    
    try:
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
            # Add database
            backup_zip.write(db_path, "database.db")
            
            # Add configuration files
            config_files = [
                "backend/requirements.txt",
                "frontend/package.json",
                ".env.example",
                "README.md"
            ]
            
            for config_file in config_files:
                config_path = Path(config_file)
                if config_path.exists():
                    backup_zip.write(config_path, config_file)
            
            video_count = 0
            video_size_mb = 0
            
            if include_videos:
                # Connect to database to get video info
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, filename, file_path, results_path 
                    FROM videos 
                    WHERE status = 'COMPLETED'
                """)
                
                completed_videos = cursor.fetchall()
                uploads_dir = Path("uploads")
                
                for video_id, filename, file_path, results_path in completed_videos:
                    try:
                        # Add video file
                        if file_path and Path(file_path).exists():
                            video_path = Path(file_path)
                            video_size_mb += video_path.stat().st_size / 1024 / 1024
                            backup_zip.write(video_path, f"videos/{video_id}/{filename}")
                            video_count += 1
                        
                        # Add results file
                        if results_path and Path(results_path).exists():
                            results_file = Path(results_path)
                            backup_zip.write(results_file, f"results/{video_id}_results.json")
                        
                        # Add a few sample frames
                        frames_dir = uploads_dir / "frames" / video_id
                        if frames_dir.exists():
                            frame_files = list(frames_dir.glob("*.jpg"))[:5]  # First 5 frames
                            for frame_file in frame_files:
                                backup_zip.write(frame_file, f"frames/{video_id}/{frame_file.name}")
                    
                    except Exception as e:
                        print(f"Warning: Could not backup video {video_id}: {e}")
                
                conn.close()
        
        backup_size_mb = backup_path.stat().st_size / 1024 / 1024
        
        return {
            "backup_file": str(backup_path),
            "backup_size_mb": round(backup_size_mb, 2),
            "database_included": True,
            "videos_included": include_videos,
            "video_count": video_count,
            "video_data_size_mb": round(video_size_mb, 2),
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": f"Backup creation failed: {e}"}

def restore_database_backup(backup_file: Path, restore_videos: bool = False) -> Dict[str, Any]:
    """Restore database and optionally videos from backup"""
    if not backup_file.exists():
        return {"error": "Backup file not found"}
    
    restore_dir = Path("restore_temp")
    restore_dir.mkdir(exist_ok=True)
    
    try:
        with zipfile.ZipFile(backup_file, 'r') as backup_zip:
            # Extract all files
            backup_zip.extractall(restore_dir)
        
        # Restore database
        db_backup = restore_dir / "database.db"
        if db_backup.exists():
            # Backup current database
            current_db = Path("backend/database.db")
            if current_db.exists():
                backup_current = Path(f"backend/database_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
                shutil.copy2(current_db, backup_current)
            
            # Restore database
            shutil.copy2(db_backup, current_db)
        
        restored_videos = 0
        
        if restore_videos:
            # Restore videos
            videos_dir = restore_dir / "videos"
            if videos_dir.exists():
                uploads_dir = Path("uploads")
                uploads_dir.mkdir(exist_ok=True)
                
                for video_subdir in videos_dir.iterdir():
                    if video_subdir.is_dir():
                        target_dir = uploads_dir / "raw"
                        target_dir.mkdir(exist_ok=True)
                        
                        for video_file in video_subdir.iterdir():
                            if video_file.is_file():
                                shutil.copy2(video_file, target_dir)
                                restored_videos += 1
            
            # Restore results
            results_dir = restore_dir / "results"
            if results_dir.exists():
                processed_dir = Path("uploads/processed")
                processed_dir.mkdir(exist_ok=True, parents=True)
                
                for results_file in results_dir.glob("*.json"):
                    shutil.copy2(results_file, processed_dir)
            
            # Restore frames
            frames_backup = restore_dir / "frames"
            if frames_backup.exists():
                frames_dir = Path("uploads/frames")
                frames_dir.mkdir(exist_ok=True, parents=True)
                
                for video_frames in frames_backup.iterdir():
                    if video_frames.is_dir():
                        target_frames = frames_dir / video_frames.name
                        if target_frames.exists():
                            shutil.rmtree(target_frames)
                        shutil.copytree(video_frames, target_frames)
        
        # Cleanup
        shutil.rmtree(restore_dir)
        
        return {
            "database_restored": True,
            "videos_restored": restore_videos,
            "restored_video_count": restored_videos,
            "restored_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        # Cleanup on error
        if restore_dir.exists():
            shutil.rmtree(restore_dir)
        return {"error": f"Restore failed: {e}"}

def list_backups(backup_dir: Path) -> List[Dict[str, Any]]:
    """List available backup files"""
    if not backup_dir.exists():
        return []
    
    backups = []
    
    for backup_file in backup_dir.glob("soccer_backup_*.zip"):
        try:
            stat = backup_file.stat()
            size_mb = stat.st_size / 1024 / 1024
            modified_time = datetime.fromtimestamp(stat.st_mtime)
            
            # Try to get info from zip file
            video_count = 0
            has_videos = False
            
            try:
                with zipfile.ZipFile(backup_file, 'r') as zf:
                    files = zf.namelist()
                    has_videos = any(f.startswith('videos/') for f in files)
                    video_count = len([f for f in files if f.startswith('videos/') and f.endswith(('.mp4', '.mov'))])
            except:
                pass
            
            backups.append({
                "filename": backup_file.name,
                "path": str(backup_file),
                "size_mb": round(size_mb, 2),
                "created_at": modified_time.isoformat(),
                "has_videos": has_videos,
                "video_count": video_count
            })
            
        except Exception as e:
            print(f"Warning: Could not read backup {backup_file}: {e}")
    
    # Sort by creation time (newest first)
    backups.sort(key=lambda x: x['created_at'], reverse=True)
    
    return backups

def export_video_metadata(output_file: Path) -> Dict[str, Any]:
    """Export video metadata to JSON for external analysis"""
    db_path = Path("backend/database.db")
    
    if not db_path.exists():
        return {"error": "Database file not found"}
    
    try:
        import json
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, filename, status, upload_time, total_touches, 
                   confidence_score, processing_time, file_size, 
                   duration, fps, resolution_width, resolution_height,
                   process_start_time, process_end_time
            FROM videos
        """)
        
        videos = []
        for row in cursor.fetchall():
            video = {
                'id': row[0],
                'filename': row[1],
                'status': row[2],
                'upload_time': row[3],
                'total_touches': row[4],
                'confidence_score': row[5],
                'processing_time': row[6],
                'file_size': row[7],
                'duration': row[8],
                'fps': row[9],
                'resolution_width': row[10],
                'resolution_height': row[11],
                'process_start_time': row[12],
                'process_end_time': row[13]
            }
            videos.append(video)
        
        conn.close()
        
        export_data = {
            'export_time': datetime.now().isoformat(),
            'total_videos': len(videos),
            'videos': videos
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return {
            'export_file': str(output_file),
            'total_videos': len(videos),
            'exported_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": f"Export failed: {e}"}

def main():
    parser = argparse.ArgumentParser(description="Soccer App Backup Tool")
    parser.add_argument('--backup-dir', type=str, default='./backups',
                       help='Directory to store backups')
    parser.add_argument('--create', action='store_true',
                       help='Create a new backup')
    parser.add_argument('--include-videos', action='store_true',
                       help='Include video files in backup (large size)')
    parser.add_argument('--restore', type=str,
                       help='Restore from backup file')
    parser.add_argument('--restore-videos', action='store_true',
                       help='Also restore video files during restore')
    parser.add_argument('--list', action='store_true',
                       help='List available backups')
    parser.add_argument('--export-metadata', type=str,
                       help='Export video metadata to JSON file')
    
    args = parser.parse_args()
    
    backup_dir = Path(args.backup_dir)
    backup_dir.mkdir(exist_ok=True)
    
    print("💾 Soccer App Backup Tool")
    print("=" * 40)
    
    if args.create:
        print(f"\n📦 Creating backup...")
        if args.include_videos:
            print("⚠️ Including videos - this may take a while and create a large file")
        
        result = create_database_backup(backup_dir, include_videos=args.include_videos)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Backup created successfully!")
            print(f"📁 File: {result['backup_file']}")
            print(f"📊 Size: {result['backup_size_mb']:.1f} MB")
            if result['videos_included']:
                print(f"🎥 Videos: {result['video_count']} videos ({result['video_data_size_mb']:.1f} MB)")
    
    elif args.restore:
        restore_file = Path(args.restore)
        print(f"\n🔄 Restoring from backup: {restore_file.name}")
        
        if args.restore_videos:
            print("⚠️ Will also restore video files")
        
        result = restore_database_backup(restore_file, restore_videos=args.restore_videos)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Restore completed successfully!")
            if result['videos_restored']:
                print(f"🎥 Restored {result['restored_video_count']} video files")
    
    elif args.list:
        print(f"\n📋 Available backups in {backup_dir}:")
        backups = list_backups(backup_dir)
        
        if not backups:
            print("  No backups found")
        else:
            for backup in backups:
                print(f"\n📦 {backup['filename']}")
                print(f"   Size: {backup['size_mb']:.1f} MB")
                print(f"   Created: {backup['created_at']}")
                if backup['has_videos']:
                    print(f"   Videos: {backup['video_count']} included")
                else:
                    print(f"   Videos: Database only")
    
    elif args.export_metadata:
        output_file = Path(args.export_metadata)
        print(f"\n📊 Exporting metadata to: {output_file}")
        
        result = export_video_metadata(output_file)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Export completed successfully!")
            print(f"📁 File: {result['export_file']}")
            print(f"📊 Videos: {result['total_videos']} records exported")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()