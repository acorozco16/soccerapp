#!/usr/bin/env python3
"""
Training Status Manager
Centralized status tracking for all automation components
"""

import os
import json
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class Status(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

@dataclass
class CollectionStatus:
    source: str
    status: Status
    progress: float  # 0.0 to 1.0
    items_collected: int
    target_items: int
    current_operation: str
    start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    error_message: Optional[str] = None

@dataclass
class TrainingStatus:
    session_id: str
    status: Status
    current_stage: str
    progress: float  # 0.0 to 1.0
    epoch: int
    total_epochs: int
    current_accuracy: float
    best_accuracy: float
    start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    error_message: Optional[str] = None

@dataclass
class SystemHealth:
    disk_usage_percent: float
    memory_usage_percent: float
    cpu_usage_percent: float
    active_processes: int
    pending_tasks: int
    last_error: Optional[str] = None
    timestamp: datetime = None

class TrainingStatusManager:
    def __init__(self, db_path: str = "backend/database.db"):
        self.db_path = Path(db_path)
        self.status_file = Path("training_data/automation/current_status.json")
        self.lock = threading.Lock()
        self._collection_status = {}
        self._training_status = None
        self._system_health = None
        self._load_status()
    
    def _load_status(self):
        """Load current status from file"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
                    
                # Reconstruct status objects
                if 'collection' in data:
                    for source, status_data in data['collection'].items():
                        self._collection_status[source] = CollectionStatus(**status_data)
                
                if 'training' in data and data['training']:
                    self._training_status = TrainingStatus(**data['training'])
                
                if 'system_health' in data and data['system_health']:
                    health_data = data['system_health']
                    if 'timestamp' in health_data:
                        health_data['timestamp'] = datetime.fromisoformat(health_data['timestamp'])
                    self._system_health = SystemHealth(**health_data)
                    
            except Exception as e:
                print(f"Warning: Could not load status file: {e}")
    
    def _save_status(self):
        """Save current status to file"""
        try:
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'collection': {
                    source: asdict(status) for source, status in self._collection_status.items()
                },
                'training': asdict(self._training_status) if self._training_status else None,
                'system_health': asdict(self._system_health) if self._system_health else None,
                'last_updated': datetime.now().isoformat()
            }
            
            # Convert datetime objects to ISO strings
            def convert_datetime(obj):
                if isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, Status):
                    return obj.value
                return obj
            
            data = convert_datetime(data)
            
            with open(self.status_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving status: {e}")
    
    def update_collection_status(self, source: str, **kwargs):
        """Update data collection status"""
        with self.lock:
            if source not in self._collection_status:
                self._collection_status[source] = CollectionStatus(
                    source=source,
                    status=Status.IDLE,
                    progress=0.0,
                    items_collected=0,
                    target_items=0,
                    current_operation=""
                )
            
            # Update fields
            for field, value in kwargs.items():
                if hasattr(self._collection_status[source], field):
                    if field == 'status' and isinstance(value, str):
                        value = Status(value)
                    setattr(self._collection_status[source], field, value)
            
            # Auto-calculate estimated completion
            if (self._collection_status[source].progress > 0 and 
                self._collection_status[source].start_time and
                self._collection_status[source].status == Status.RUNNING):
                
                elapsed = datetime.now() - self._collection_status[source].start_time
                if self._collection_status[source].progress > 0:
                    total_time = elapsed / self._collection_status[source].progress
                    remaining_time = total_time - elapsed
                    self._collection_status[source].estimated_completion = datetime.now() + remaining_time
            
            self._save_status()
    
    def update_training_status(self, session_id: str, **kwargs):
        """Update training status"""
        with self.lock:
            if not self._training_status or self._training_status.session_id != session_id:
                self._training_status = TrainingStatus(
                    session_id=session_id,
                    status=Status.IDLE,
                    current_stage="",
                    progress=0.0,
                    epoch=0,
                    total_epochs=0,
                    current_accuracy=0.0,
                    best_accuracy=0.0
                )
            
            # Update fields
            for field, value in kwargs.items():
                if hasattr(self._training_status, field):
                    if field == 'status' and isinstance(value, str):
                        value = Status(value)
                    setattr(self._training_status, field, value)
            
            # Auto-calculate progress from epoch if not provided
            if ('epoch' in kwargs or 'total_epochs' in kwargs) and 'progress' not in kwargs:
                if self._training_status.total_epochs > 0:
                    self._training_status.progress = self._training_status.epoch / self._training_status.total_epochs
            
            # Auto-calculate estimated completion
            if (self._training_status.progress > 0 and 
                self._training_status.start_time and
                self._training_status.status == Status.RUNNING):
                
                elapsed = datetime.now() - self._training_status.start_time
                if self._training_status.progress > 0:
                    total_time = elapsed / self._training_status.progress
                    remaining_time = total_time - elapsed
                    self._training_status.estimated_completion = datetime.now() + remaining_time
            
            self._save_status()
    
    def update_system_health(self, **kwargs):
        """Update system health metrics"""
        with self.lock:
            if not self._system_health:
                self._system_health = SystemHealth(
                    disk_usage_percent=0.0,
                    memory_usage_percent=0.0,
                    cpu_usage_percent=0.0,
                    active_processes=0,
                    pending_tasks=0,
                    timestamp=datetime.now()
                )
            
            # Update fields
            for field, value in kwargs.items():
                if hasattr(self._system_health, field):
                    setattr(self._system_health, field, value)
            
            self._system_health.timestamp = datetime.now()
            self._save_status()
    
    def get_collection_status(self, source: Optional[str] = None) -> Dict[str, CollectionStatus]:
        """Get data collection status"""
        with self.lock:
            if source:
                return {source: self._collection_status.get(source)}
            return self._collection_status.copy()
    
    def get_training_status(self) -> Optional[TrainingStatus]:
        """Get current training status"""
        with self.lock:
            return self._training_status
    
    def get_system_health(self) -> Optional[SystemHealth]:
        """Get current system health"""
        with self.lock:
            return self._system_health
    
    def get_overall_status(self) -> Dict[str, Any]:
        """Get comprehensive status overview"""
        with self.lock:
            # Collection summary
            collection_summary = {
                'active_sources': len([s for s in self._collection_status.values() if s.status == Status.RUNNING]),
                'total_sources': len(self._collection_status),
                'total_items_collected': sum(s.items_collected for s in self._collection_status.values()),
                'overall_progress': sum(s.progress for s in self._collection_status.values()) / max(len(self._collection_status), 1)
            }
            
            # Training summary
            training_summary = None
            if self._training_status:
                training_summary = {
                    'status': self._training_status.status.value,
                    'current_stage': self._training_status.current_stage,
                    'progress': self._training_status.progress,
                    'current_accuracy': self._training_status.current_accuracy,
                    'best_accuracy': self._training_status.best_accuracy
                }
            
            # System summary
            system_summary = None
            if self._system_health:
                system_summary = {
                    'disk_usage': self._system_health.disk_usage_percent,
                    'memory_usage': self._system_health.memory_usage_percent,
                    'cpu_usage': self._system_health.cpu_usage_percent,
                    'health_status': self._get_health_status()
                }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'collection': collection_summary,
                'training': training_summary,
                'system': system_summary,
                'alerts': self._get_active_alerts()
            }
    
    def _get_health_status(self) -> str:
        """Determine overall health status"""
        if not self._system_health:
            return "unknown"
        
        if (self._system_health.disk_usage_percent > 90 or 
            self._system_health.memory_usage_percent > 90):
            return "critical"
        elif (self._system_health.disk_usage_percent > 80 or 
              self._system_health.memory_usage_percent > 80):
            return "warning"
        else:
            return "healthy"
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts"""
        alerts = []
        
        # System health alerts
        if self._system_health:
            if self._system_health.disk_usage_percent > 90:
                alerts.append({
                    'type': 'critical',
                    'component': 'system',
                    'message': f"Disk usage critical: {self._system_health.disk_usage_percent:.1f}%"
                })
            elif self._system_health.disk_usage_percent > 80:
                alerts.append({
                    'type': 'warning',
                    'component': 'system',
                    'message': f"Disk usage high: {self._system_health.disk_usage_percent:.1f}%"
                })
            
            if self._system_health.memory_usage_percent > 90:
                alerts.append({
                    'type': 'critical',
                    'component': 'system',
                    'message': f"Memory usage critical: {self._system_health.memory_usage_percent:.1f}%"
                })
        
        # Collection alerts
        for source, status in self._collection_status.items():
            if status.status == Status.FAILED:
                alerts.append({
                    'type': 'error',
                    'component': 'collection',
                    'message': f"Data collection failed for {source}: {status.error_message}"
                })
        
        # Training alerts
        if self._training_status and self._training_status.status == Status.FAILED:
            alerts.append({
                'type': 'error',
                'component': 'training',
                'message': f"Training failed: {self._training_status.error_message}"
            })
        
        return alerts
    
    def log_to_database(self, component: str, event: str, data: Dict[str, Any]):
        """Log events to database"""
        if not self.db_path.exists():
            return
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO system_health_logs (component, status, metrics, alerts)
                VALUES (?, ?, ?, ?)
            """, (
                component,
                event,
                json.dumps(data),
                json.dumps(self._get_active_alerts())
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error logging to database: {e}")
    
    def clear_completed_tasks(self):
        """Clear completed collection tasks"""
        with self.lock:
            completed_sources = [
                source for source, status in self._collection_status.items()
                if status.status in [Status.COMPLETED, Status.FAILED]
            ]
            
            for source in completed_sources:
                # Keep for historical purposes but mark as archived
                self._collection_status[source].current_operation = "archived"
            
            self._save_status()
    
    def reset_training_status(self):
        """Reset training status (for new session)"""
        with self.lock:
            self._training_status = None
            self._save_status()
    
    def get_historical_data(self, component: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get historical data from database"""
        if not self.db_path.exists():
            return []
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            start_date = datetime.now() - timedelta(days=days)
            
            cursor.execute("""
                SELECT timestamp, component, status, metrics
                FROM system_health_logs
                WHERE component = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """, (component, start_date.isoformat()))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'timestamp': row[0],
                    'component': row[1],
                    'status': row[2],
                    'metrics': json.loads(row[3]) if row[3] else {}
                })
            
            conn.close()
            return results
            
        except Exception as e:
            print(f"Error getting historical data: {e}")
            return []

# Global status manager instance
_status_manager = None

def get_status_manager() -> TrainingStatusManager:
    """Get global status manager instance"""
    global _status_manager
    if _status_manager is None:
        _status_manager = TrainingStatusManager()
    return _status_manager

def update_collection_status(source: str, **kwargs):
    """Convenience function to update collection status"""
    get_status_manager().update_collection_status(source, **kwargs)

def update_training_status(session_id: str, **kwargs):
    """Convenience function to update training status"""
    get_status_manager().update_training_status(session_id, **kwargs)

def update_system_health(**kwargs):
    """Convenience function to update system health"""
    get_status_manager().update_system_health(**kwargs)

if __name__ == "__main__":
    # Test the status manager
    manager = get_status_manager()
    
    # Test collection status
    manager.update_collection_status(
        "youtube",
        status=Status.RUNNING,
        progress=0.5,
        items_collected=500,
        target_items=1000,
        current_operation="Downloading videos...",
        start_time=datetime.now()
    )
    
    # Test training status
    manager.update_training_status(
        "test_session_001",
        status=Status.RUNNING,
        current_stage="Stage 1",
        progress=0.3,
        epoch=30,
        total_epochs=100,
        current_accuracy=0.82,
        best_accuracy=0.85,
        start_time=datetime.now()
    )
    
    # Test system health
    manager.update_system_health(
        disk_usage_percent=75.0,
        memory_usage_percent=60.0,
        cpu_usage_percent=45.0,
        active_processes=3,
        pending_tasks=5
    )
    
    # Print overall status
    import pprint
    pprint.pprint(manager.get_overall_status())