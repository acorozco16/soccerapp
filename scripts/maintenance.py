#!/usr/bin/env python3
"""
Maintenance Script for Soccer Video Analysis App
Performs routine maintenance, optimization, and health checks
"""

import os
import sys
import sqlite3
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

def optimize_database() -> Dict[str, Any]:
    """Optimize database performance"""
    db_path = Path("backend/database.db")
    
    if not db_path.exists():
        return {"error": "Database file not found"}
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get stats before optimization
        cursor.execute("SELECT COUNT(*) FROM videos")
        video_count = cursor.fetchone()[0]
        
        initial_size = db_path.stat().st_size / 1024 / 1024
        
        # Vacuum database (reclaim space and defragment)
        cursor.execute("VACUUM")
        
        # Analyze tables for query optimization
        cursor.execute("ANALYZE")
        
        # Update statistics
        cursor.execute("PRAGMA optimize")
        
        conn.commit()
        conn.close()
        
        final_size = db_path.stat().st_size / 1024 / 1024
        space_saved = initial_size - final_size
        
        return {
            "video_count": video_count,
            "initial_size_mb": round(initial_size, 2),
            "final_size_mb": round(final_size, 2),
            "space_saved_mb": round(space_saved, 2),
            "optimized_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": f"Database optimization failed: {e}"}

def check_file_integrity() -> Dict[str, Any]:
    """Check integrity of stored files"""
    db_path = Path("backend/database.db")
    
    if not db_path.exists():
        return {"error": "Database file not found"}
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, filename, file_path, results_path, status
            FROM videos
        """)
        
        videos = cursor.fetchall()
        conn.close()
        
        missing_files = []
        orphaned_files = []
        corrupted_files = []
        
        uploads_dir = Path("uploads")
        
        # Check each video's files
        for video_id, filename, file_path, results_path, status in videos:
            # Check video file
            if file_path:
                video_file = Path(file_path)
                if not video_file.exists():
                    missing_files.append({
                        'video_id': video_id,
                        'filename': filename,
                        'missing_file': file_path,
                        'type': 'video'
                    })
                else:
                    # Check if file is corrupted (basic size check)
                    if video_file.stat().st_size < 1024:  # Less than 1KB
                        corrupted_files.append({
                            'video_id': video_id,
                            'filename': filename,
                            'file_path': file_path,
                            'size_bytes': video_file.stat().st_size
                        })
            
            # Check results file for completed videos
            if status == 'COMPLETED' and results_path:
                results_file = Path(results_path)
                if not results_file.exists():
                    missing_files.append({
                        'video_id': video_id,
                        'filename': filename,
                        'missing_file': results_path,
                        'type': 'results'
                    })
        
        # Check for orphaned files
        valid_video_ids = {video[0] for video in videos}
        
        if uploads_dir.exists():
            # Check for orphaned video files
            for video_file in uploads_dir.rglob('*.mp4'):
                # Try to find if this file belongs to any video
                if not any(str(video_file) in (video[2] or '') for video in videos):
                    orphaned_files.append({
                        'file_path': str(video_file),
                        'type': 'video',
                        'size_mb': round(video_file.stat().st_size / 1024 / 1024, 2)
                    })
            
            # Check for orphaned frame directories
            frames_dir = uploads_dir / "frames"
            if frames_dir.exists():
                for frame_dir in frames_dir.iterdir():
                    if frame_dir.is_dir() and frame_dir.name not in valid_video_ids:
                        orphaned_files.append({
                            'file_path': str(frame_dir),
                            'type': 'frames',
                            'size_mb': round(sum(f.stat().st_size for f in frame_dir.rglob('*') if f.is_file()) / 1024 / 1024, 2)
                        })
        
        return {
            "total_videos_checked": len(videos),
            "missing_files": missing_files,
            "missing_count": len(missing_files),
            "orphaned_files": orphaned_files,
            "orphaned_count": len(orphaned_files),
            "corrupted_files": corrupted_files,
            "corrupted_count": len(corrupted_files),
            "checked_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": f"Integrity check failed: {e}"}

def generate_usage_report(days: int = 30) -> Dict[str, Any]:
    """Generate usage statistics report"""
    db_path = Path("backend/database.db")
    
    if not db_path.exists():
        return {"error": "Database file not found"}
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Total videos
        cursor.execute("SELECT COUNT(*) FROM videos")
        total_videos = cursor.fetchone()[0]
        
        # Videos in date range
        cursor.execute("""
            SELECT COUNT(*) FROM videos 
            WHERE upload_time >= ? AND upload_time <= ?
        """, (start_date.isoformat(), end_date.isoformat()))
        period_videos = cursor.fetchone()[0]
        
        # Status breakdown
        cursor.execute("""
            SELECT status, COUNT(*) FROM videos 
            WHERE upload_time >= ? AND upload_time <= ?
            GROUP BY status
        """, (start_date.isoformat(), end_date.isoformat()))
        status_breakdown = dict(cursor.fetchall())
        
        # Processing time stats
        cursor.execute("""
            SELECT AVG(processing_time), MIN(processing_time), MAX(processing_time)
            FROM videos 
            WHERE status = 'COMPLETED' 
            AND upload_time >= ? AND upload_time <= ?
            AND processing_time IS NOT NULL
        """, (start_date.isoformat(), end_date.isoformat()))
        
        processing_stats = cursor.fetchone()
        avg_processing_time = processing_stats[0] if processing_stats[0] else 0
        min_processing_time = processing_stats[1] if processing_stats[1] else 0
        max_processing_time = processing_stats[2] if processing_stats[2] else 0
        
        # Touch count stats
        cursor.execute("""
            SELECT AVG(total_touches), MIN(total_touches), MAX(total_touches)
            FROM videos 
            WHERE status = 'COMPLETED' 
            AND upload_time >= ? AND upload_time <= ?
            AND total_touches IS NOT NULL
        """, (start_date.isoformat(), end_date.isoformat()))
        
        touch_stats = cursor.fetchone()
        avg_touches = touch_stats[0] if touch_stats[0] else 0
        min_touches = touch_stats[1] if touch_stats[1] else 0
        max_touches = touch_stats[2] if touch_stats[2] else 0
        
        # Confidence score stats
        cursor.execute("""
            SELECT AVG(confidence_score), MIN(confidence_score), MAX(confidence_score)
            FROM videos 
            WHERE status = 'COMPLETED' 
            AND upload_time >= ? AND upload_time <= ?
            AND confidence_score IS NOT NULL
        """, (start_date.isoformat(), end_date.isoformat()))
        
        confidence_stats = cursor.fetchone()
        avg_confidence = confidence_stats[0] if confidence_stats[0] else 0
        min_confidence = confidence_stats[1] if confidence_stats[1] else 0
        max_confidence = confidence_stats[2] if confidence_stats[2] else 0
        
        # Daily upload counts
        cursor.execute("""
            SELECT DATE(upload_time) as upload_date, COUNT(*) as daily_count
            FROM videos 
            WHERE upload_time >= ? AND upload_time <= ?
            GROUP BY DATE(upload_time)
            ORDER BY upload_date
        """, (start_date.isoformat(), end_date.isoformat()))
        
        daily_uploads = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "report_period_days": days,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_videos_all_time": total_videos,
            "videos_in_period": period_videos,
            "status_breakdown": status_breakdown,
            "processing_time": {
                "average_seconds": round(avg_processing_time, 2) if avg_processing_time else 0,
                "min_seconds": round(min_processing_time, 2) if min_processing_time else 0,
                "max_seconds": round(max_processing_time, 2) if max_processing_time else 0
            },
            "touch_counts": {
                "average": round(avg_touches, 1) if avg_touches else 0,
                "min": int(min_touches) if min_touches else 0,
                "max": int(max_touches) if max_touches else 0
            },
            "confidence_scores": {
                "average": round(avg_confidence, 3) if avg_confidence else 0,
                "min": round(min_confidence, 3) if min_confidence else 0,
                "max": round(max_confidence, 3) if max_confidence else 0
            },
            "daily_uploads": daily_uploads,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": f"Report generation failed: {e}"}

def system_health_check() -> Dict[str, Any]:
    """Comprehensive system health check"""
    try:
        # Import health check from backend
        from health import get_health_status
        health_status = get_health_status()
        
        # Add maintenance-specific checks
        maintenance_checks = {}
        
        # Check log file sizes
        log_files = [
            "training_pipeline.log",
            "backend/app.log"
        ]
        
        large_logs = []
        for log_file in log_files:
            log_path = Path(log_file)
            if log_path.exists():
                size_mb = log_path.stat().st_size / 1024 / 1024
                if size_mb > 100:  # Log files over 100MB
                    large_logs.append({
                        "file": log_file,
                        "size_mb": round(size_mb, 2)
                    })
        
        maintenance_checks['large_logs'] = large_logs
        
        # Check for old temporary files
        temp_dirs = [Path("restore_temp"), Path("temp"), Path("/tmp/soccer_*")]
        old_temp_files = []
        
        for temp_pattern in temp_dirs:
            if temp_pattern.exists():
                for temp_file in temp_pattern.rglob('*'):
                    if temp_file.is_file():
                        # Check if older than 1 day
                        age_hours = (datetime.now().timestamp() - temp_file.stat().st_mtime) / 3600
                        if age_hours > 24:
                            old_temp_files.append({
                                "file": str(temp_file),
                                "age_hours": round(age_hours, 1),
                                "size_mb": round(temp_file.stat().st_size / 1024 / 1024, 2)
                            })
        
        maintenance_checks['old_temp_files'] = old_temp_files
        
        # Combine health status with maintenance checks
        health_status['maintenance_checks'] = maintenance_checks
        
        return health_status
        
    except ImportError:
        # Fallback if health module not available
        return {
            "error": "Health check module not available",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Health check failed: {e}"}

def main():
    parser = argparse.ArgumentParser(description="Soccer App Maintenance Tool")
    parser.add_argument('--optimize-db', action='store_true',
                       help='Optimize database performance')
    parser.add_argument('--check-integrity', action='store_true',
                       help='Check file integrity')
    parser.add_argument('--usage-report', type=int, metavar='DAYS', default=30,
                       help='Generate usage report for last N days')
    parser.add_argument('--health-check', action='store_true',
                       help='Run comprehensive health check')
    parser.add_argument('--all', action='store_true',
                       help='Run all maintenance tasks')
    
    args = parser.parse_args()
    
    print("🔧 Soccer App Maintenance Tool")
    print("=" * 40)
    
    if args.all or args.optimize_db:
        print("\n🚀 Optimizing database...")
        result = optimize_database()
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Database optimized successfully!")
            print(f"📊 Videos: {result['video_count']}")
            print(f"💾 Size: {result['initial_size_mb']:.1f} MB → {result['final_size_mb']:.1f} MB")
            if result['space_saved_mb'] > 0:
                print(f"💰 Space saved: {result['space_saved_mb']:.1f} MB")
    
    if args.all or args.check_integrity:
        print("\n🔍 Checking file integrity...")
        result = check_file_integrity()
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Integrity check completed!")
            print(f"📊 Videos checked: {result['total_videos_checked']}")
            
            if result['missing_count'] > 0:
                print(f"⚠️ Missing files: {result['missing_count']}")
                for missing in result['missing_files'][:5]:  # Show first 5
                    print(f"   📁 {missing['type']}: {missing['filename']}")
            
            if result['orphaned_count'] > 0:
                print(f"🗑️ Orphaned files: {result['orphaned_count']}")
                total_orphaned_mb = sum(f.get('size_mb', 0) for f in result['orphaned_files'])
                print(f"   💾 Total size: {total_orphaned_mb:.1f} MB")
            
            if result['corrupted_count'] > 0:
                print(f"⚠️ Potentially corrupted files: {result['corrupted_count']}")
            
            if all(result[key] == 0 for key in ['missing_count', 'orphaned_count', 'corrupted_count']):
                print("✨ All files are in good condition!")
    
    if args.all or args.usage_report:
        days = args.usage_report if not args.all else 30
        print(f"\n📊 Generating usage report ({days} days)...")
        result = generate_usage_report(days=days)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"📈 Usage Report ({days} days)")
            print(f"   Total videos (all time): {result['total_videos_all_time']}")
            print(f"   Videos in period: {result['videos_in_period']}")
            
            status_breakdown = result.get('status_breakdown', {})
            if status_breakdown:
                print(f"   Status breakdown:")
                for status, count in status_breakdown.items():
                    print(f"     {status}: {count}")
            
            processing = result.get('processing_time', {})
            if processing.get('average_seconds', 0) > 0:
                print(f"   Average processing time: {processing['average_seconds']:.1f}s")
            
            touches = result.get('touch_counts', {})
            if touches.get('average', 0) > 0:
                print(f"   Average touches detected: {touches['average']:.1f}")
            
            confidence = result.get('confidence_scores', {})
            if confidence.get('average', 0) > 0:
                print(f"   Average confidence: {confidence['average']:.3f}")
    
    if args.all or args.health_check:
        print("\n🏥 Running health check...")
        result = system_health_check()
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            status = result.get('status', 'unknown')
            if status == 'healthy':
                print("✅ System is healthy!")
            elif status == 'degraded':
                print("⚠️ System has warnings")
            else:
                print("❌ System is unhealthy")
            
            summary = result.get('summary', {})
            if summary:
                print(f"   Checks: {summary.get('healthy', 0)}/{summary.get('total_checks', 0)} healthy")
                if summary.get('warnings', 0) > 0:
                    print(f"   Warnings: {summary.get('warnings', 0)}")
                if summary.get('errors', 0) > 0:
                    print(f"   Errors: {summary.get('errors', 0)}")
            
            # Show maintenance-specific issues
            maint_checks = result.get('maintenance_checks', {})
            if maint_checks.get('large_logs'):
                print(f"⚠️ Large log files detected:")
                for log in maint_checks['large_logs']:
                    print(f"   📝 {log['file']}: {log['size_mb']:.1f} MB")
            
            if maint_checks.get('old_temp_files'):
                temp_count = len(maint_checks['old_temp_files'])
                temp_size = sum(f['size_mb'] for f in maint_checks['old_temp_files'])
                print(f"🗑️ Old temporary files: {temp_count} files ({temp_size:.1f} MB)")
    
    if not any([args.optimize_db, args.check_integrity, args.usage_report, args.health_check, args.all]):
        parser.print_help()

if __name__ == "__main__":
    main()