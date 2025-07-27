#!/usr/bin/env python3
"""
Training Dashboard API
Provides REST endpoints for the training dashboard
"""

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# Add required paths
import sys
sys.path.append(str(Path(__file__).parent.parent / "automation"))

from training_status import get_status_manager
from collect_all_data import ComprehensiveDataCollector
from train_progressive import ProgressiveTrainer
from deploy_improved_model import ModelDeploymentManager

logger = logging.getLogger(__name__)

# Create API router
training_api = APIRouter(prefix="/training-api")

# Request models
class DataCollectionRequest(BaseModel):
    target_images: int = 1000
    sources: Optional[List[str]] = None

class TrainingRequest(BaseModel):
    auto_deploy: bool = False
    min_improvement: Optional[float] = None

class DeploymentRequest(BaseModel):
    model_path: Optional[str] = None
    test_first: bool = True
    rollback_on_failure: bool = True

# Global instances
status_manager = None
active_tasks = {}

def get_status_manager_instance():
    """Get global status manager instance"""
    global status_manager
    if status_manager is None:
        status_manager = get_status_manager()
    return status_manager

@training_api.get("/")
async def dashboard_home():
    """Serve the training dashboard"""
    dashboard_file = Path(__file__).parent / "training_dashboard.html"
    if not dashboard_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    with open(dashboard_file, 'r') as f:
        content = f.read()
    
    return HTMLResponse(content=content)

@training_api.get("/assets/{filename}")
async def serve_assets(filename: str):
    """Serve dashboard assets"""
    asset_file = Path(__file__).parent / filename
    if not asset_file.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    
    return FileResponse(asset_file)

@training_api.get("/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        status_mgr = get_status_manager_instance()
        
        # Get overall status
        overall_status = status_mgr.get_overall_status()
        
        # Get detailed component status
        collection_status = status_mgr.get_collection_status()
        training_status = status_mgr.get_training_status()
        system_health = status_mgr.get_system_health()
        
        # Get recent activities from logs
        recent_activities = _get_recent_activities()
        
        # Get performance history
        performance_history = _get_performance_history()
        
        # Combine all information
        response = {
            **overall_status,
            'detailed_collection': collection_status,
            'detailed_training': training_status,
            'detailed_system': system_health,
            'recent_activities': recent_activities,
            'performance_history': performance_history,
            'active_tasks': list(active_tasks.keys())
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@training_api.get("/data-collection")
async def get_data_collection_status():
    """Get detailed data collection status"""
    try:
        status_mgr = get_status_manager_instance()
        collection_status = status_mgr.get_collection_status()
        
        # Add additional statistics
        collected_data_stats = _analyze_collected_data()
        
        return {
            'status': collection_status,
            'statistics': collected_data_stats,
            'last_updated': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting data collection status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@training_api.get("/training-progress")
async def get_training_progress():
    """Get detailed training progress"""
    try:
        status_mgr = get_status_manager_instance()
        training_status = status_mgr.get_training_status()
        
        # Add training history
        training_history = _get_training_history()
        
        return {
            'current_training': training_status,
            'history': training_history,
            'last_updated': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting training progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@training_api.post("/trigger-collection")
async def trigger_data_collection(request: DataCollectionRequest, background_tasks: BackgroundTasks):
    """Trigger data collection process"""
    try:
        if 'data_collection' in active_tasks:
            raise HTTPException(status_code=409, detail="Data collection already running")
        
        # Start data collection in background
        background_tasks.add_task(
            _run_data_collection,
            request.target_images,
            request.sources
        )
        
        active_tasks['data_collection'] = {
            'started_at': datetime.now().isoformat(),
            'target_images': request.target_images,
            'sources': request.sources
        }
        
        return {
            'success': True,
            'message': 'Data collection started',
            'task_id': 'data_collection',
            'parameters': request.dict()
        }
        
    except Exception as e:
        logger.error(f"Error triggering data collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@training_api.post("/trigger-training")
async def trigger_model_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Trigger model training process"""
    try:
        if 'model_training' in active_tasks:
            raise HTTPException(status_code=409, detail="Model training already running")
        
        # Start training in background
        background_tasks.add_task(
            _run_progressive_training,
            request.auto_deploy,
            request.min_improvement
        )
        
        active_tasks['model_training'] = {
            'started_at': datetime.now().isoformat(),
            'auto_deploy': request.auto_deploy,
            'min_improvement': request.min_improvement
        }
        
        return {
            'success': True,
            'message': 'Model training started',
            'task_id': 'model_training',
            'parameters': request.dict()
        }
        
    except Exception as e:
        logger.error(f"Error triggering model training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@training_api.post("/deploy-model")
async def deploy_model(request: DeploymentRequest, background_tasks: BackgroundTasks):
    """Deploy trained model"""
    try:
        if 'model_deployment' in active_tasks:
            raise HTTPException(status_code=409, detail="Model deployment already running")
        
        # Find model to deploy if not specified
        model_path = request.model_path
        if not model_path:
            model_path = _find_latest_trained_model()
            if not model_path:
                raise HTTPException(status_code=404, detail="No trained model found for deployment")
        
        # Start deployment in background
        background_tasks.add_task(
            _run_model_deployment,
            model_path,
            request.test_first,
            request.rollback_on_failure
        )
        
        active_tasks['model_deployment'] = {
            'started_at': datetime.now().isoformat(),
            'model_path': model_path,
            'test_first': request.test_first,
            'rollback_on_failure': request.rollback_on_failure
        }
        
        return {
            'success': True,
            'message': 'Model deployment started',
            'task_id': 'model_deployment',
            'model_path': model_path,
            'parameters': request.dict()
        }
        
    except Exception as e:
        logger.error(f"Error triggering model deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@training_api.get("/model-history")
async def get_model_history():
    """Get model performance history"""
    try:
        history = _get_model_deployment_history()
        return {
            'history': history,
            'total_deployments': len(history),
            'last_updated': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting model history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@training_api.get("/system-health")
async def get_system_health():
    """Get detailed system health information"""
    try:
        status_mgr = get_status_manager_instance()
        system_health = status_mgr.get_system_health()
        
        # Add additional system information
        additional_info = _get_additional_system_info()
        
        return {
            'health': system_health,
            'additional_info': additional_info,
            'last_updated': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions

async def _run_data_collection(target_images: int, sources: List[str] = None):
    """Run data collection as background task"""
    try:
        collector = ComprehensiveDataCollector()
        result = await collector.collect_all_data(
            target_images=target_images,
            enable_sources=sources
        )
        
        logger.info(f"Data collection completed: {result}")
        
    except Exception as e:
        logger.error(f"Data collection task failed: {e}")
    finally:
        # Remove from active tasks
        active_tasks.pop('data_collection', None)

async def _run_progressive_training(auto_deploy: bool, min_improvement: float = None):
    """Run progressive training as background task"""
    try:
        trainer = ProgressiveTrainer()
        result = await trainer.run_progressive_training(
            auto_deploy=auto_deploy,
            min_improvement=min_improvement
        )
        
        logger.info(f"Progressive training completed: {result}")
        
    except Exception as e:
        logger.error(f"Progressive training task failed: {e}")
    finally:
        # Remove from active tasks
        active_tasks.pop('model_training', None)

async def _run_model_deployment(model_path: str, test_first: bool, rollback_on_failure: bool):
    """Run model deployment as background task"""
    try:
        deployer = ModelDeploymentManager()
        result = await deployer.deploy_model(
            model_path=model_path,
            test_first=test_first,
            rollback_on_failure=rollback_on_failure
        )
        
        logger.info(f"Model deployment completed: {result}")
        
    except Exception as e:
        logger.error(f"Model deployment task failed: {e}")
    finally:
        # Remove from active tasks
        active_tasks.pop('model_deployment', None)

# Helper functions

def _get_recent_activities() -> List[Dict[str, Any]]:
    """Get recent system activities"""
    try:
        status_mgr = get_status_manager_instance()
        
        # Get historical data from different components
        activities = []
        
        # Training activities
        training_history = status_mgr.get_historical_data('training', days=7)
        for record in training_history[:10]:
            activities.append({
                'component': 'training',
                'message': f"Training {record['status']}: {record.get('metrics', {})}",
                'timestamp': record['timestamp'],
                'status': record['status'].lower()
            })
        
        # Collection activities
        collection_history = status_mgr.get_historical_data('collection', days=7)
        for record in collection_history[:10]:
            activities.append({
                'component': 'collection',
                'message': f"Data collection {record['status']}",
                'timestamp': record['timestamp'],
                'status': record['status'].lower()
            })
        
        # Sort by timestamp (most recent first)
        activities.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return activities[:20]  # Return top 20
        
    except Exception as e:
        logger.error(f"Error getting recent activities: {e}")
        return []

def _get_performance_history() -> List[Dict[str, Any]]:
    """Get model performance history"""
    try:
        # Read training session results
        experiments_dir = Path("training_data/experiments")
        if not experiments_dir.exists():
            return []
        
        history = []
        
        # Find training session result files
        for result_file in experiments_dir.glob("*_results.json"):
            try:
                with open(result_file, 'r') as f:
                    session_data = json.load(f)
                
                best_model = session_data.get('best_model', {})
                if best_model:
                    history.append({
                        'timestamp': session_data.get('completed_at', ''),
                        'accuracy': best_model.get('final_accuracy', 0),
                        'model_version': best_model.get('stage_name', ''),
                        'improvement': best_model.get('improvement_percent', 0)
                    })
                    
            except Exception as e:
                logger.warning(f"Error reading result file {result_file}: {e}")
        
        # Sort by timestamp
        history.sort(key=lambda x: x['timestamp'])
        
        return history
        
    except Exception as e:
        logger.error(f"Error getting performance history: {e}")
        return []

def _analyze_collected_data() -> Dict[str, Any]:
    """Analyze currently collected data"""
    try:
        stats = {
            'total_files': 0,
            'total_size_mb': 0,
            'by_source': {},
            'by_quality': {'high': 0, 'medium': 0, 'low': 0}
        }
        
        collected_data_dir = Path("training_data/collected_data")
        if not collected_data_dir.exists():
            return stats
        
        # Analyze each source directory
        for source_dir in collected_data_dir.iterdir():
            if not source_dir.is_dir():
                continue
            
            source_stats = {
                'files': 0,
                'size_mb': 0
            }
            
            # Count files and calculate size
            for file_path in source_dir.rglob("*"):
                if file_path.is_file() and not file_path.name.endswith('.json'):
                    source_stats['files'] += 1
                    source_stats['size_mb'] += file_path.stat().st_size / (1024 * 1024)
            
            stats['by_source'][source_dir.name] = source_stats
            stats['total_files'] += source_stats['files']
            stats['total_size_mb'] += source_stats['size_mb']
        
        # Analyze processed dataset quality
        processed_dir = Path("training_data/processed_dataset/images")
        if processed_dir.exists():
            for quality in ['high_quality', 'medium_quality', 'low_quality']:
                quality_dir = processed_dir / quality
                if quality_dir.exists():
                    count = len(list(quality_dir.glob("*.jpg")))
                    stats['by_quality'][quality.replace('_quality', '')] = count
        
        return stats
        
    except Exception as e:
        logger.error(f"Error analyzing collected data: {e}")
        return {}

def _get_training_history() -> List[Dict[str, Any]]:
    """Get training session history"""
    try:
        # Read from database if available
        db_path = Path("backend/database.db")
        if db_path.exists():
            import sqlite3
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, start_time, end_time, status, model_version,
                       accuracy_after, improvement_percent
                FROM training_sessions
                ORDER BY start_time DESC
                LIMIT 20
            """)
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    'session_id': row[0],
                    'start_time': row[1],
                    'end_time': row[2],
                    'status': row[3],
                    'model_version': row[4],
                    'final_accuracy': row[5] or 0,
                    'improvement': row[6] or 0
                })
            
            conn.close()
            return history
        
        return []
        
    except Exception as e:
        logger.error(f"Error getting training history: {e}")
        return []

def _find_latest_trained_model() -> Optional[str]:
    """Find the most recently trained model"""
    try:
        models_dir = Path("training_data/models")
        if not models_dir.exists():
            return None
        
        # Look for model files
        model_files = list(models_dir.rglob("*.pt"))
        if not model_files:
            return None
        
        # Return most recent
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
        return str(latest_model)
        
    except Exception as e:
        logger.error(f"Error finding latest trained model: {e}")
        return None

def _get_model_deployment_history() -> List[Dict[str, Any]]:
    """Get model deployment history"""
    try:
        # Read from database if available
        db_path = Path("backend/database.db")
        if db_path.exists():
            import sqlite3
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, model_version, deployment_time, status, performance_metrics
                FROM model_deployments
                ORDER BY deployment_time DESC
                LIMIT 50
            """)
            
            history = []
            for row in cursor.fetchall():
                try:
                    metrics = json.loads(row[4]) if row[4] else {}
                except:
                    metrics = {}
                
                history.append({
                    'deployment_id': row[0],
                    'model_version': row[1],
                    'deployment_time': row[2],
                    'status': row[3],
                    'metrics': metrics
                })
            
            conn.close()
            return history
        
        return []
        
    except Exception as e:
        logger.error(f"Error getting deployment history: {e}")
        return []

def _get_additional_system_info() -> Dict[str, Any]:
    """Get additional system information"""
    try:
        import psutil
        import platform
        
        return {
            'platform': platform.system(),
            'architecture': platform.architecture()[0],
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'disk_total_gb': round(psutil.disk_usage('/').total / (1024**3), 2),
            'uptime_hours': round((datetime.now().timestamp() - psutil.boot_time()) / 3600, 1)
        }
        
    except Exception as e:
        logger.error(f"Error getting additional system info: {e}")
        return {}

# Initialize when module is imported
def init_dashboard_api():
    """Initialize dashboard API"""
    global status_manager
    status_manager = get_status_manager_instance()
    logger.info("Training dashboard API initialized")

# Auto-initialize
init_dashboard_api()