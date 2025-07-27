#!/usr/bin/env python3
"""
Intelligent Model Deployment System
Handles production deployment with A/B testing and rollback capabilities
"""

import os
import json
import asyncio
import time
import uuid
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import sqlite3
import subprocess

# Add required paths
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "backend"))
sys.path.append(str(Path(__file__).parent))

from training_status import get_status_manager, update_collection_status, Status

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_data/logs/model_deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelDeploymentManager:
    def __init__(self, config_file: str = "automation_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.status_manager = get_status_manager()
        
        # Deployment directories
        self.models_dir = Path("training_data/models")
        self.production_dir = self.models_dir / "production"
        self.staging_dir = self.models_dir / "staging"
        self.backup_dir = self.models_dir / "backups"
        
        # Create directories
        for dir_path in [self.production_dir, self.staging_dir, self.backup_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Deployment tracking
        self.deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.ab_test_duration_hours = 24  # A/B test duration
        
        # Performance thresholds
        self.min_accuracy_threshold = 0.7
        self.max_processing_time_increase = 20  # Max 20% increase in processing time
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.config_file.exists():
            raise Exception(f"Configuration file not found: {self.config_file}")
        
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load configuration: {e}")
    
    def _get_current_production_model(self) -> Optional[Dict[str, Any]]:
        """Get current production model information"""
        try:
            model_files = list(self.production_dir.glob("*.pt"))
            if not model_files:
                return None
            
            # Get most recent model
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            
            # Load metadata
            metadata_file = latest_model.with_suffix('.json')
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            return {
                'model_path': str(latest_model),
                'metadata': metadata,
                'deployed_at': datetime.fromtimestamp(latest_model.stat().st_mtime).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting current production model: {e}")
            return None
    
    def _backup_current_model(self) -> Optional[str]:
        """Backup current production model"""
        try:
            current_model = self._get_current_production_model()
            if not current_model:
                logger.info("No current production model to backup")
                return None
            
            model_path = Path(current_model['model_path'])
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_path.name}"
            backup_path = self.backup_dir / backup_name
            
            # Copy model file
            shutil.copy2(model_path, backup_path)
            
            # Copy metadata
            metadata_src = model_path.with_suffix('.json')
            if metadata_src.exists():
                metadata_dst = backup_path.with_suffix('.json')
                shutil.copy2(metadata_src, metadata_dst)
            
            # Create backup info
            backup_info = {
                'backup_id': backup_name.replace('.pt', ''),
                'original_path': str(model_path),
                'backup_path': str(backup_path),
                'backup_time': datetime.now().isoformat(),
                'metadata': current_model['metadata']
            }
            
            backup_info_file = backup_path.with_suffix('.backup.json')
            with open(backup_info_file, 'w') as f:
                json.dump(backup_info, f, indent=2)
            
            logger.info(f"Current model backed up to: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to backup current model: {e}")
            return None
    
    async def _validate_model_performance(self, model_path: str) -> Dict[str, Any]:
        """Validate model performance against test samples"""
        logger.info(f"Validating model performance: {model_path}")
        
        try:
            # Load the model
            from ultralytics import YOLO
            model = YOLO(model_path)
            
            # Test against sample videos
            sample_videos_dir = Path("sample_videos")
            if not sample_videos_dir.exists():
                return {'error': 'No sample videos available for validation'}
            
            validation_results = {
                'model_path': model_path,
                'validation_time': datetime.now().isoformat(),
                'test_results': [],
                'average_accuracy': 0.0,
                'average_processing_time': 0.0,
                'passes_validation': False
            }
            
            # Known test videos with expected results
            test_videos = {
                'clear_touches.mp4': {'expected_touches': 23, 'max_error': 3},
                'difficult_lighting.mp4': {'expected_touches': 18, 'max_error': 5},
                'multiple_players.mp4': {'expected_touches': 31, 'max_error': 5}
            }
            
            total_accuracy = 0.0
            total_processing_time = 0.0
            valid_tests = 0
            
            for video_name, test_info in test_videos.items():
                video_path = sample_videos_dir / video_name
                if not video_path.exists():
                    continue
                
                try:
                    start_time = time.time()
                    
                    # Run inference (simplified - in practice would use full video processor)
                    results = model(str(video_path))
                    
                    processing_time = time.time() - start_time
                    
                    # Count detected objects (placeholder for actual touch counting)
                    detected_count = len(results[0].boxes) if results[0].boxes is not None else 0
                    
                    # Calculate accuracy
                    expected = test_info['expected_touches']
                    max_error = test_info['max_error']
                    error = abs(detected_count - expected)
                    accuracy = max(0.0, 1.0 - (error / max_error)) if max_error > 0 else 0.0
                    
                    test_result = {
                        'video': video_name,
                        'expected_touches': expected,
                        'detected_count': detected_count,
                        'error': error,
                        'accuracy': accuracy,
                        'processing_time': processing_time,
                        'passes': error <= max_error
                    }
                    
                    validation_results['test_results'].append(test_result)
                    
                    total_accuracy += accuracy
                    total_processing_time += processing_time
                    valid_tests += 1
                    
                    logger.info(f"Validated {video_name}: {detected_count}/{expected} touches, {accuracy:.2f} accuracy")
                    
                except Exception as e:
                    logger.warning(f"Validation failed for {video_name}: {e}")
                    validation_results['test_results'].append({
                        'video': video_name,
                        'error': str(e),
                        'passes': False
                    })
            
            if valid_tests > 0:
                validation_results['average_accuracy'] = total_accuracy / valid_tests
                validation_results['average_processing_time'] = total_processing_time / valid_tests
                
                # Determine if validation passes
                min_accuracy = self.min_accuracy_threshold
                passes_accuracy = validation_results['average_accuracy'] >= min_accuracy
                
                validation_results['passes_validation'] = passes_accuracy
                validation_results['validation_summary'] = {
                    'tests_run': valid_tests,
                    'tests_passed': sum(1 for r in validation_results['test_results'] if r.get('passes', False)),
                    'meets_accuracy_threshold': passes_accuracy,
                    'accuracy_threshold': min_accuracy
                }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Model validation error: {e}")
            return {'error': str(e), 'passes_validation': False}
    
    async def _compare_with_current_model(self, new_model_path: str) -> Dict[str, Any]:
        """Compare new model with current production model"""
        logger.info("Comparing new model with current production model")
        
        current_model = self._get_current_production_model()
        if not current_model:
            logger.info("No current production model for comparison")
            return {
                'has_current_model': False,
                'recommendation': 'DEPLOY_AS_FIRST',
                'reason': 'No current production model exists'
            }
        
        # Validate both models
        logger.info("Validating new model...")
        new_validation = await self._validate_model_performance(new_model_path)
        
        logger.info("Validating current model...")
        current_validation = await self._validate_model_performance(current_model['model_path'])
        
        comparison = {
            'has_current_model': True,
            'current_model': current_model,
            'new_model_path': new_model_path,
            'current_validation': current_validation,
            'new_validation': new_validation,
            'comparison_time': datetime.now().isoformat()
        }
        
        # Compare performance
        current_accuracy = current_validation.get('average_accuracy', 0.0)
        new_accuracy = new_validation.get('average_accuracy', 0.0)
        
        current_time = current_validation.get('average_processing_time', 0.0)
        new_time = new_validation.get('average_processing_time', 0.0)
        
        accuracy_improvement = new_accuracy - current_accuracy
        time_increase_percent = ((new_time - current_time) / max(current_time, 0.001)) * 100
        
        comparison['performance_comparison'] = {
            'accuracy_improvement': accuracy_improvement,
            'processing_time_increase_percent': time_increase_percent,
            'new_model_passes_validation': new_validation.get('passes_validation', False),
            'current_model_passes_validation': current_validation.get('passes_validation', False)
        }
        
        # Make deployment recommendation
        if not new_validation.get('passes_validation', False):
            recommendation = 'DO_NOT_DEPLOY'
            reason = 'New model fails validation tests'
        elif accuracy_improvement <= 0 and time_increase_percent > self.max_processing_time_increase:
            recommendation = 'DO_NOT_DEPLOY'
            reason = f'No accuracy improvement and processing time increased by {time_increase_percent:.1f}%'
        elif accuracy_improvement >= 0.05:  # 5% improvement
            recommendation = 'DEPLOY_IMMEDIATELY'
            reason = f'Significant accuracy improvement: +{accuracy_improvement:.2f}'
        elif accuracy_improvement > 0:
            recommendation = 'DEPLOY_WITH_AB_TEST'
            reason = f'Modest improvement: +{accuracy_improvement:.2f}, requires A/B testing'
        else:
            recommendation = 'DEPLOY_TO_STAGING'
            reason = 'Minor or no improvement, deploy to staging for further testing'
        
        comparison['recommendation'] = recommendation
        comparison['reason'] = reason
        
        logger.info(f"Comparison complete: {recommendation} - {reason}")
        
        return comparison
    
    async def _deploy_to_production(self, model_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy model to production"""
        logger.info(f"Deploying model to production: {model_path}")
        
        try:
            # Backup current model
            backup_path = self._backup_current_model()
            
            # Copy new model to production
            model_file = Path(model_path)
            production_model_name = f"soccer_ball_production_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            production_path = self.production_dir / production_model_name
            
            shutil.copy2(model_file, production_path)
            
            # Create production metadata
            production_metadata = {
                **metadata,
                'deployed_at': datetime.now().isoformat(),
                'deployment_id': self.deployment_id,
                'original_path': model_path,
                'backup_path': backup_path,
                'deployment_type': 'production'
            }
            
            metadata_path = production_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(production_metadata, f, indent=2)
            
            # Log deployment to database
            self._log_deployment(production_path, production_metadata, 'DEPLOYED')
            
            # Update integration point (this would restart/reload the main app)
            await self._update_integration_point(str(production_path))
            
            deployment_result = {
                'success': True,
                'production_path': str(production_path),
                'backup_path': backup_path,
                'deployment_time': datetime.now().isoformat(),
                'metadata': production_metadata
            }
            
            logger.info(f"Model deployed successfully to: {production_path}")
            return deployment_result
            
        except Exception as e:
            logger.error(f"Production deployment failed: {e}")
            raise
    
    async def _deploy_to_staging(self, model_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy model to staging for testing"""
        logger.info(f"Deploying model to staging: {model_path}")
        
        try:
            model_file = Path(model_path)
            staging_model_name = f"soccer_ball_staging_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            staging_path = self.staging_dir / staging_model_name
            
            shutil.copy2(model_file, staging_path)
            
            # Create staging metadata
            staging_metadata = {
                **metadata,
                'deployed_at': datetime.now().isoformat(),
                'deployment_id': self.deployment_id,
                'original_path': model_path,
                'deployment_type': 'staging'
            }
            
            metadata_path = staging_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(staging_metadata, f, indent=2)
            
            # Log deployment
            self._log_deployment(staging_path, staging_metadata, 'STAGED')
            
            deployment_result = {
                'success': True,
                'staging_path': str(staging_path),
                'deployment_time': datetime.now().isoformat(),
                'metadata': staging_metadata
            }
            
            logger.info(f"Model deployed to staging: {staging_path}")
            return deployment_result
            
        except Exception as e:
            logger.error(f"Staging deployment failed: {e}")
            raise
    
    async def _start_ab_test(self, model_path: str, duration_hours: int = 24) -> Dict[str, Any]:
        """Start A/B test between new model and current production model"""
        logger.info(f"Starting A/B test for {duration_hours} hours")
        
        try:
            # Deploy to staging first
            staging_result = await self._deploy_to_staging(model_path, {
                'ab_test': True,
                'test_duration_hours': duration_hours,
                'test_start_time': datetime.now().isoformat()
            })
            
            # Create A/B test configuration
            ab_test_config = {
                'test_id': f"ab_test_{self.deployment_id}",
                'start_time': datetime.now().isoformat(),
                'duration_hours': duration_hours,
                'end_time': (datetime.now() + timedelta(hours=duration_hours)).isoformat(),
                'production_model': self._get_current_production_model(),
                'test_model': staging_result,
                'traffic_split': 0.5,  # 50/50 split
                'status': 'RUNNING',
                'metrics': {
                    'requests_production': 0,
                    'requests_test': 0,
                    'accuracy_production': [],
                    'accuracy_test': [],
                    'processing_time_production': [],
                    'processing_time_test': []
                }
            }
            
            # Save A/B test config
            ab_test_file = self.staging_dir / f"ab_test_{self.deployment_id}.json"
            with open(ab_test_file, 'w') as f:
                json.dump(ab_test_config, f, indent=2)
            
            logger.info(f"A/B test started: {ab_test_config['test_id']}")
            
            # Schedule evaluation after test period
            # In a real implementation, this would use a task scheduler
            logger.info(f"A/B test will run until {ab_test_config['end_time']}")
            
            return ab_test_config
            
        except Exception as e:
            logger.error(f"A/B test start failed: {e}")
            raise
    
    async def _update_integration_point(self, model_path: str):
        """Update the integration point in the main video processor"""
        try:
            # Create a symlink or configuration file that the main app can read
            integration_file = Path("training_data/models/current_production_model.txt")
            with open(integration_file, 'w') as f:
                f.write(model_path)
            
            logger.info(f"Integration point updated: {model_path}")
            
            # In a real deployment, you might:
            # 1. Send a signal to reload the model
            # 2. Update a configuration file
            # 3. Restart relevant services
            # 4. Update a database record
            
        except Exception as e:
            logger.warning(f"Failed to update integration point: {e}")
    
    def _log_deployment(self, model_path: Path, metadata: Dict[str, Any], status: str):
        """Log deployment to database"""
        try:
            db_path = Path("backend/database.db")
            if not db_path.exists():
                logger.warning("Main database not found, skipping deployment logging")
                return
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO model_deployments (
                    id, model_version, deployment_time, previous_model,
                    status, performance_metrics
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                self.deployment_id,
                str(model_path.name),
                datetime.now().isoformat(),
                metadata.get('backup_path', ''),
                status,
                json.dumps({
                    'deployment_type': metadata.get('deployment_type'),
                    'original_path': metadata.get('original_path'),
                    'metadata': metadata
                })
            ))
            
            conn.commit()
            conn.close()
            
            logger.info("Deployment logged to database")
            
        except Exception as e:
            logger.error(f"Failed to log deployment: {e}")
    
    async def rollback_deployment(self, backup_path: str = None) -> Dict[str, Any]:
        """Rollback to previous model version"""
        logger.info("Starting deployment rollback")
        
        try:
            if backup_path:
                # Use specified backup
                backup_model = Path(backup_path)
            else:
                # Find most recent backup
                backup_files = list(self.backup_dir.glob("backup_*.pt"))
                if not backup_files:
                    raise Exception("No backup models found for rollback")
                
                backup_model = max(backup_files, key=lambda p: p.stat().st_mtime)
            
            if not backup_model.exists():
                raise Exception(f"Backup model not found: {backup_model}")
            
            # Load backup metadata
            backup_info_file = backup_model.with_suffix('.backup.json')
            backup_info = {}
            if backup_info_file.exists():
                with open(backup_info_file, 'r') as f:
                    backup_info = json.load(f)
            
            # Backup current model before rollback
            current_backup = self._backup_current_model()
            
            # Copy backup to production
            production_name = f"soccer_ball_production_rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            production_path = self.production_dir / production_name
            
            shutil.copy2(backup_model, production_path)
            
            # Create rollback metadata
            rollback_metadata = {
                **backup_info.get('metadata', {}),
                'rollback_time': datetime.now().isoformat(),
                'rollback_id': f"rollback_{self.deployment_id}",
                'backup_source': str(backup_model),
                'replaced_model': current_backup,
                'deployment_type': 'rollback'
            }
            
            metadata_path = production_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(rollback_metadata, f, indent=2)
            
            # Update integration point
            await self._update_integration_point(str(production_path))
            
            # Log rollback
            self._log_deployment(production_path, rollback_metadata, 'ROLLED_BACK')
            
            rollback_result = {
                'success': True,
                'rollback_time': datetime.now().isoformat(),
                'production_path': str(production_path),
                'backup_source': str(backup_model),
                'replaced_model': current_backup,
                'metadata': rollback_metadata
            }
            
            logger.info(f"Rollback completed successfully: {production_path}")
            return rollback_result
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise
    
    async def deploy_model(self, model_path: str, test_first: bool = True,
                          rollback_on_failure: bool = True) -> Dict[str, Any]:
        """Deploy model with testing and rollback capabilities"""
        
        logger.info(f"🚀 Starting intelligent model deployment: {model_path}")
        logger.info(f"Test first: {test_first}, Rollback on failure: {rollback_on_failure}")
        
        deployment_result = {
            'deployment_id': self.deployment_id,
            'model_path': model_path,
            'started_at': datetime.now().isoformat(),
            'test_first': test_first,
            'rollback_on_failure': rollback_on_failure,
            'steps_completed': [],
            'final_status': None,
            'error_message': None
        }
        
        try:
            # Step 1: Validate model file exists
            model_file = Path(model_path)
            if not model_file.exists():
                raise Exception(f"Model file not found: {model_path}")
            
            deployment_result['steps_completed'].append('model_validation')
            
            # Step 2: Load model metadata
            metadata_file = model_file.with_suffix('.json')
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            deployment_result['model_metadata'] = metadata
            deployment_result['steps_completed'].append('metadata_loaded')
            
            # Step 3: Performance validation (if test_first is True)
            if test_first:
                logger.info("🧪 Running performance validation...")
                validation_result = await self._validate_model_performance(model_path)
                deployment_result['validation_result'] = validation_result
                deployment_result['steps_completed'].append('performance_validation')
                
                if not validation_result.get('passes_validation', False):
                    if rollback_on_failure:
                        logger.warning("Model failed validation, but no deployment to rollback from")
                    raise Exception(f"Model failed performance validation: {validation_result}")
            
            # Step 4: Compare with current production model
            logger.info("📊 Comparing with current production model...")
            comparison_result = await self._compare_with_current_model(model_path)
            deployment_result['comparison_result'] = comparison_result
            deployment_result['steps_completed'].append('model_comparison')
            
            recommendation = comparison_result['recommendation']
            logger.info(f"Deployment recommendation: {recommendation}")
            
            # Step 5: Execute deployment based on recommendation
            if recommendation == 'DO_NOT_DEPLOY':
                deployment_result['final_status'] = 'REJECTED'
                deployment_result['reason'] = comparison_result['reason']
                logger.info(f"❌ Deployment rejected: {comparison_result['reason']}")
                
            elif recommendation == 'DEPLOY_IMMEDIATELY':
                logger.info("✅ Deploying immediately to production...")
                production_result = await self._deploy_to_production(model_path, metadata)
                deployment_result['production_deployment'] = production_result
                deployment_result['steps_completed'].append('production_deployment')
                deployment_result['final_status'] = 'DEPLOYED_TO_PRODUCTION'
                
            elif recommendation == 'DEPLOY_WITH_AB_TEST':
                logger.info("🔬 Starting A/B test deployment...")
                ab_test_result = await self._start_ab_test(model_path, self.ab_test_duration_hours)
                deployment_result['ab_test'] = ab_test_result
                deployment_result['steps_completed'].append('ab_test_started')
                deployment_result['final_status'] = 'AB_TEST_STARTED'
                
            elif recommendation == 'DEPLOY_TO_STAGING':
                logger.info("🚧 Deploying to staging...")
                staging_result = await self._deploy_to_staging(model_path, metadata)
                deployment_result['staging_deployment'] = staging_result
                deployment_result['steps_completed'].append('staging_deployment')
                deployment_result['final_status'] = 'DEPLOYED_TO_STAGING'
                
            elif recommendation == 'DEPLOY_AS_FIRST':
                logger.info("🆕 Deploying as first production model...")
                production_result = await self._deploy_to_production(model_path, metadata)
                deployment_result['production_deployment'] = production_result
                deployment_result['steps_completed'].append('production_deployment')
                deployment_result['final_status'] = 'DEPLOYED_AS_FIRST'
            
            # Final step: Save deployment record
            deployment_result['completed_at'] = datetime.now().isoformat()
            
            deployment_file = Path(f"training_data/logs/deployment_{self.deployment_id}.json")
            deployment_file.parent.mkdir(parents=True, exist_ok=True)
            with open(deployment_file, 'w') as f:
                json.dump(deployment_result, f, indent=2, default=str)
            
            logger.info(f"🎉 Deployment completed: {deployment_result['final_status']}")
            return deployment_result
            
        except Exception as e:
            error_msg = f"Deployment failed: {e}"
            logger.error(error_msg)
            
            deployment_result['error_message'] = error_msg
            deployment_result['final_status'] = 'FAILED'
            deployment_result['completed_at'] = datetime.now().isoformat()
            
            # Attempt rollback if requested and there's something to rollback to
            if rollback_on_failure and self._get_current_production_model():
                try:
                    logger.info("🔄 Attempting automatic rollback...")
                    rollback_result = await self.rollback_deployment()
                    deployment_result['rollback_result'] = rollback_result
                    deployment_result['steps_completed'].append('automatic_rollback')
                    logger.info("✅ Automatic rollback completed")
                except Exception as rollback_error:
                    logger.error(f"Automatic rollback failed: {rollback_error}")
                    deployment_result['rollback_error'] = str(rollback_error)
            
            raise

async def main():
    parser = argparse.ArgumentParser(description="Intelligent Model Deployment")
    parser.add_argument('model_path', help='Path to model file to deploy')
    parser.add_argument('--test-first', action='store_true', default=True,
                       help='Run performance tests before deployment')
    parser.add_argument('--rollback-on-failure', action='store_true', default=True,
                       help='Rollback on deployment failure')
    parser.add_argument('--force-production', action='store_true',
                       help='Force deployment to production (skip recommendations)')
    parser.add_argument('--rollback', type=str, metavar='BACKUP_PATH',
                       help='Rollback to specified backup model')
    
    args = parser.parse_args()
    
    try:
        deployer = ModelDeploymentManager()
        
        if args.rollback:
            # Rollback mode
            result = await deployer.rollback_deployment(args.rollback)
            logger.info("🔄 Rollback completed successfully!")
        else:
            # Deployment mode
            result = await deployer.deploy_model(
                model_path=args.model_path,
                test_first=args.test_first,
                rollback_on_failure=args.rollback_on_failure
            )
            logger.info("🚀 Deployment completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Operation failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)