#!/usr/bin/env python3
"""
Health Check Endpoint
Provides system health status for monitoring and deployment
"""

import os
import sqlite3
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

def check_database_health() -> Dict[str, Any]:
    """Check database connectivity and status"""
    try:
        db_path = "database.db"
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM videos")
            count = cursor.fetchone()[0]
            conn.close()
            
            return {
                "status": "healthy",
                "total_videos": count,
                "database_size_mb": round(os.path.getsize(db_path) / 1024 / 1024, 2)
            }
        else:
            return {
                "status": "warning",
                "message": "Database file not found - will be created on first use"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Database error: {str(e)}"
        }

def check_upload_directory() -> Dict[str, Any]:
    """Check upload directory status"""
    try:
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        if upload_dir.exists() and upload_dir.is_dir():
            # Count files and calculate total size
            files = list(upload_dir.rglob("*"))
            total_files = len([f for f in files if f.is_file()])
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            
            return {
                "status": "healthy",
                "total_files": total_files,
                "total_size_mb": round(total_size / 1024 / 1024, 2),
                "writable": os.access(upload_dir, os.W_OK)
            }
        else:
            return {
                "status": "error",
                "message": "Upload directory not accessible"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Upload directory error: {str(e)}"
        }

def check_system_resources() -> Dict[str, Any]:
    """Check system resource usage"""
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            "status": "healthy",
            "memory": {
                "total_gb": round(memory.total / 1024 / 1024 / 1024, 2),
                "available_gb": round(memory.available / 1024 / 1024 / 1024, 2),
                "percent_used": memory.percent
            },
            "disk": {
                "total_gb": round(disk.total / 1024 / 1024 / 1024, 2),
                "free_gb": round(disk.free / 1024 / 1024 / 1024, 2),
                "percent_used": round((disk.used / disk.total) * 100, 1)
            },
            "cpu_percent": cpu_percent
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"System resources error: {str(e)}"
        }

def check_dependencies() -> Dict[str, Any]:
    """Check if required dependencies are available"""
    try:
        import cv2
        import mediapipe
        import numpy
        import fastapi
        import uvicorn
        
        return {
            "status": "healthy",
            "versions": {
                "opencv": cv2.__version__,
                "mediapipe": mediapipe.__version__,
                "numpy": numpy.__version__,
                "fastapi": fastapi.__version__,
                "uvicorn": uvicorn.__version__
            }
        }
    except ImportError as e:
        return {
            "status": "error",
            "message": f"Missing dependency: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Dependency check error: {str(e)}"
        }

def get_health_status() -> Dict[str, Any]:
    """Get comprehensive health status"""
    
    # Run all health checks
    database_health = check_database_health()
    upload_health = check_upload_directory()
    system_health = check_system_resources()
    dependency_health = check_dependencies()
    
    # Determine overall status
    all_checks = [database_health, upload_health, system_health, dependency_health]
    error_count = sum(1 for check in all_checks if check.get("status") == "error")
    warning_count = sum(1 for check in all_checks if check.get("status") == "warning")
    
    if error_count > 0:
        overall_status = "unhealthy"
    elif warning_count > 0:
        overall_status = "degraded"
    else:
        overall_status = "healthy"
    
    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "checks": {
            "database": database_health,
            "uploads": upload_health,
            "system": system_health,
            "dependencies": dependency_health
        },
        "summary": {
            "total_checks": len(all_checks),
            "healthy": len([c for c in all_checks if c.get("status") == "healthy"]),
            "warnings": warning_count,
            "errors": error_count
        }
    }

if __name__ == "__main__":
    # CLI usage for debugging
    import json
    health = get_health_status()
    print(json.dumps(health, indent=2))