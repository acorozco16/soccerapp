#!/usr/bin/env python3
"""
Progressive Training System
Automatically trains and improves soccer ball detection models
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
import concurrent.futures

# Add required paths
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "models"))
sys.path.append(str(Path(__file__).parent.parent.parent / "backend"))

from yolo_trainer import YOLOTrainer, TrainingConfig
from dataset_manager import DatasetManager
from training_status import get_status_manager, update_training_status, Status

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../../training_data/logs/progressive_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProgressiveTrainer:
    def __init__(self, config_file: str = "../../automation_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.status_manager = get_status_manager()
        
        # Training configuration
        self.training_config = self.config.get('training', {})
        self.min_accuracy_improvement = self.training_config.get('min_accuracy_improvement', 5.0)
        self.target_accuracy = self.training_config.get('target_accuracy', 90.0)
        self.max_training_hours = self.training_config.get('max_training_hours', 4)
        self.training_stages = self.training_config.get('training_stages', [100, 500, 1000, 5000])
        
        # Initialize components
        self.dataset_manager = DatasetManager(
            processed_data_dir="../processed_dataset",
            datasets_dir="../datasets"
        )
        
        self.trainer = YOLOTrainer(
            models_dir="training_data/models",
            experiments_dir="training_data/experiments"
        )
        
        # Model tracking
        self.current_production_model = self._get_current_production_model()
        self.session_id = f"progressive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Performance tracking
        self.training_history = []
        self.model_comparisons = []
    
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
        """Get information about current production model"""
        try:
            # Check if there's a deployed model
            deployed_dir = Path("training_data/models/production")
            if deployed_dir.exists():
                model_files = list(deployed_dir.glob("*.pt"))
                if model_files:
                    model_file = model_files[0]  # Use most recent
                    
                    # Get model metadata
                    metadata_file = model_file.with_suffix('.json')
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        return {
                            'model_path': str(model_file),
                            'metadata': metadata
                        }
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not get current production model: {e}")
            return None
    
    def _analyze_available_data(self) -> Dict[str, Any]:
        """Analyze currently available training data"""
        try:
            stats = self.dataset_manager.analyze_processed_data()
            
            # Categorize by quality
            quality_distribution = {
                'high': 0,
                'medium': 0, 
                'low': 0
            }
            
            # Count files in each quality directory
            processed_dir = Path("training_data/processed_dataset/images")
            if processed_dir.exists():
                for quality in quality_distribution.keys():
                    quality_dir = processed_dir / f"{quality}_quality"
                    if quality_dir.exists():
                        quality_distribution[quality] = len(list(quality_dir.glob("*.jpg")))
            
            return {
                'total_images': stats.total_images,
                'total_labels': stats.total_labels,
                'quality_distribution': quality_distribution,
                'sufficient_for_training': stats.total_images >= self.training_stages[0],
                'recommended_stage': self._determine_training_stage(stats.total_images)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing available data: {e}")
            return {
                'total_images': 0,
                'total_labels': 0,
                'quality_distribution': {'high': 0, 'medium': 0, 'low': 0},
                'sufficient_for_training': False,
                'recommended_stage': 0
            }
    
    def _determine_training_stage(self, total_images: int) -> int:
        """Determine appropriate training stage based on available data"""
        for i, stage_size in enumerate(self.training_stages):
            if total_images >= stage_size:
                continue
            else:
                return max(0, i - 1)  # Previous stage
        return len(self.training_stages) - 1  # Highest stage
    
    async def _create_training_dataset(self, max_images: int, 
                                     quality_filter: List[str] = None) -> str:
        """Create dataset for training"""
        if quality_filter is None:
            quality_filter = ['high', 'medium']
        
        logger.info(f"Creating training dataset with {max_images} images")
        
        try:
            version = f"progressive_{datetime.now().strftime('%Y%m%d_%H%M')}"
            
            dataset_version = self.dataset_manager.create_yolo_dataset(
                version=version,
                description=f"Progressive training dataset - {max_images} images",
                max_images=max_images,
                quality_filter=quality_filter
            )
            
            logger.info(f"Created dataset: {dataset_version.path}")
            return dataset_version.path
            
        except Exception as e:
            logger.error(f"Failed to create training dataset: {e}")
            raise
    
    async def _train_model_stage(self, dataset_path: str, stage_info: Dict[str, Any]) -> Dict[str, Any]:
        """Train model for a specific stage"""
        stage_name = stage_info['name']
        model_size = stage_info['model_size']
        epochs = stage_info['epochs']
        
        logger.info(f"Training stage: {stage_name} with {model_size}")
        
        update_training_status(
            self.session_id,
            status=Status.RUNNING,
            current_stage=stage_name,
            progress=0.0,
            total_epochs=epochs,
            start_time=datetime.now()
        )
        
        try:
            # Create training configuration
            training_config = TrainingConfig(
                model_name=f"soccer_ball_{model_size}_{stage_name}_{datetime.now().strftime('%H%M%S')}",
                base_model=f"yolov8{model_size[-1]}.pt",  # Extract size letter (n, s, m)
                dataset_path=dataset_path,
                epochs=epochs,
                batch_size=16 if model_size.endswith('n') else 8,
                img_size=640,
                learning_rate=0.001,
                weight_decay=0.0005,
                momentum=0.937,
                warmup_epochs=3,
                patience=50,
                device='auto',
                workers=4
            )
            
            # Train model with progress callback
            def progress_callback(epoch: int, metrics: Dict[str, float]):
                progress = epoch / epochs
                current_accuracy = metrics.get('mAP50', 0.0) * 100
                best_accuracy = max(metrics.get('best_mAP50', 0.0) * 100, current_accuracy)
                
                update_training_status(
                    self.session_id,
                    progress=progress,
                    epoch=epoch,
                    current_accuracy=current_accuracy,
                    best_accuracy=best_accuracy
                )
            
            # Start training
            training_result = self.trainer.train_model(
                training_config, 
                version=f"{self.session_id}_{stage_name}",
                progress_callback=progress_callback
            )
            
            # Calculate improvement over baseline
            baseline_accuracy = 0.0
            if self.current_production_model:
                baseline_accuracy = self.current_production_model.get('metadata', {}).get('accuracy', 0.0)
            
            final_accuracy = training_result.final_metrics.get('mAP50', 0.0) * 100
            improvement = final_accuracy - baseline_accuracy
            
            stage_result = {
                'stage_name': stage_name,
                'model_size': model_size,
                'model_path': training_result.model_path,
                'training_time': training_result.training_time,
                'final_accuracy': final_accuracy,
                'baseline_accuracy': baseline_accuracy,
                'improvement_percent': improvement,
                'metrics': training_result.final_metrics,
                'meets_improvement_threshold': improvement >= self.min_accuracy_improvement,
                'meets_target_accuracy': final_accuracy >= self.target_accuracy
            }
            
            logger.info(f"Stage {stage_name} completed: {final_accuracy:.2f}% accuracy (+{improvement:.2f}%)")
            
            return stage_result
            
        except Exception as e:
            error_msg = f"Training stage {stage_name} failed: {e}"
            logger.error(error_msg)
            
            update_training_status(
                self.session_id,
                status=Status.FAILED,
                error_message=error_msg
            )
            
            raise
    
    async def _compare_models(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple trained models and select the best"""
        if not model_results:
            return None
        
        logger.info(f"Comparing {len(model_results)} trained models")
        
        # Sort by accuracy (primary) and improvement (secondary)
        sorted_models = sorted(
            model_results,
            key=lambda x: (x['final_accuracy'], x['improvement_percent']),
            reverse=True
        )
        
        best_model = sorted_models[0]
        
        # Run additional validation on best model
        try:
            validation_result = await self._validate_model(best_model['model_path'])
            best_model['validation_results'] = validation_result
        except Exception as e:
            logger.warning(f"Model validation failed: {e}")
            best_model['validation_results'] = {'error': str(e)}
        
        comparison_result = {
            'best_model': best_model,
            'all_models': sorted_models,
            'selection_criteria': {
                'primary': 'final_accuracy',
                'secondary': 'improvement_percent',
                'min_improvement_threshold': self.min_accuracy_improvement
            },
            'recommendation': self._get_deployment_recommendation(best_model)
        }
        
        # Save comparison results
        comparison_file = Path(f"training_data/experiments/{self.session_id}_model_comparison.json")
        comparison_file.parent.mkdir(parents=True, exist_ok=True)
        with open(comparison_file, 'w') as f:
            json.dump(comparison_result, f, indent=2, default=str)
        
        return comparison_result
    
    async def _validate_model(self, model_path: str) -> Dict[str, Any]:
        """Validate trained model against test samples"""
        logger.info(f"Validating model: {model_path}")
        
        try:
            # Use existing sample videos for validation
            sample_videos_dir = Path("sample_videos")
            if not sample_videos_dir.exists():
                return {'error': 'No sample videos available for validation'}
            
            # Test against known samples
            validation_results = {
                'tested_videos': 0,
                'average_accuracy': 0.0,
                'individual_results': []
            }
            
            # Load the model
            from ultralytics import YOLO
            model = YOLO(model_path)
            
            # Test videos with known touch counts
            test_videos = {
                'clear_touches.mp4': 23,
                'difficult_lighting.mp4': 18,
                'multiple_players.mp4': 31
            }
            
            total_accuracy = 0.0
            tested_count = 0
            
            for video_name, expected_touches in test_videos.items():
                video_path = sample_videos_dir / video_name
                if not video_path.exists():
                    continue
                
                try:
                    # Run basic inference test (simplified)
                    # In a full implementation, this would integrate with video_processor
                    results = model(str(video_path))
                    
                    # For now, just check that model can process video
                    detected_objects = len(results[0].boxes) if results[0].boxes is not None else 0
                    
                    # Simple accuracy calculation (placeholder)
                    # In practice, you'd need full video processing pipeline
                    accuracy = max(0.0, 1.0 - abs(detected_objects - expected_touches) / expected_touches)
                    
                    validation_results['individual_results'].append({
                        'video': video_name,
                        'expected_touches': expected_touches,
                        'detected_objects': detected_objects,
                        'accuracy': accuracy
                    })
                    
                    total_accuracy += accuracy
                    tested_count += 1
                    
                except Exception as e:
                    logger.warning(f"Validation failed for {video_name}: {e}")
            
            if tested_count > 0:
                validation_results['tested_videos'] = tested_count
                validation_results['average_accuracy'] = total_accuracy / tested_count
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Model validation error: {e}")
            return {'error': str(e)}
    
    def _get_deployment_recommendation(self, model_result: Dict[str, Any]) -> str:
        """Get recommendation for model deployment"""
        accuracy = model_result['final_accuracy']
        improvement = model_result['improvement_percent']
        meets_threshold = model_result['meets_improvement_threshold']
        meets_target = model_result['meets_target_accuracy']
        
        if meets_target and meets_threshold:
            return "DEPLOY_IMMEDIATELY"
        elif meets_threshold:
            return "DEPLOY_WITH_TESTING"
        elif improvement > 0:
            return "DEPLOY_TO_STAGING"
        else:
            return "DO_NOT_DEPLOY"
    
    def _log_training_session(self, session_result: Dict[str, Any]):
        """Log training session to database"""
        try:
            db_path = Path("backend/database.db")
            if not db_path.exists():
                logger.warning("Main database not found, skipping session logging")
                return
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            best_model = session_result.get('best_model', {})
            
            cursor.execute("""
                INSERT OR REPLACE INTO training_sessions (
                    id, start_time, end_time, status, model_version,
                    dataset_size, accuracy_before, accuracy_after,
                    improvement_percent, config, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.session_id,
                session_result.get('started_at'),
                session_result.get('completed_at'),
                'COMPLETED' if best_model else 'FAILED',
                best_model.get('model_path', ''),
                session_result.get('dataset_size', 0),
                best_model.get('baseline_accuracy', 0.0),
                best_model.get('final_accuracy', 0.0),
                best_model.get('improvement_percent', 0.0),
                json.dumps(self.training_config),
                session_result.get('error_message')
            ))
            
            conn.commit()
            conn.close()
            
            logger.info("Training session logged to database")
            
        except Exception as e:
            logger.error(f"Failed to log training session: {e}")
    
    async def run_progressive_training(self, auto_deploy: bool = False,
                                     min_improvement: float = None) -> Dict[str, Any]:
        """Run the complete progressive training pipeline"""
        
        start_time = datetime.now()
        
        # Override min improvement if specified
        if min_improvement is not None:
            self.min_accuracy_improvement = min_improvement
        
        logger.info(f"🚀 Starting progressive training session: {self.session_id}")
        logger.info(f"Min improvement threshold: {self.min_accuracy_improvement}%")
        logger.info(f"Target accuracy: {self.target_accuracy}%")
        
        session_result = {
            'session_id': self.session_id,
            'started_at': start_time.isoformat(),
            'config': self.training_config,
            'stages_attempted': [],
            'stages_completed': [],
            'model_results': [],
            'best_model': None,
            'deployment_recommendation': None,
            'errors': []
        }
        
        try:
            # 1. Analyze available data
            logger.info("📊 Analyzing available training data...")
            data_analysis = self._analyze_available_data()
            session_result['data_analysis'] = data_analysis
            
            if not data_analysis['sufficient_for_training']:
                raise Exception(f"Insufficient training data: {data_analysis['total_images']} images available, minimum {self.training_stages[0]} required")
            
            logger.info(f"Available data: {data_analysis['total_images']} images")
            logger.info(f"Quality distribution: {data_analysis['quality_distribution']}")
            
            # 2. Determine training stages to run
            max_stage = data_analysis['recommended_stage']
            stages_to_run = self.training_stages[:max_stage + 1]
            
            logger.info(f"Training stages to run: {stages_to_run}")
            
            # 3. Create training datasets and run stages
            model_results = []
            
            for i, stage_size in enumerate(stages_to_run):
                stage_name = f"stage_{i+1}_{stage_size}"
                session_result['stages_attempted'].append(stage_name)
                
                try:
                    logger.info(f"🎯 Starting training stage: {stage_name}")
                    
                    # Create dataset for this stage
                    dataset_path = await self._create_training_dataset(
                        max_images=stage_size,
                        quality_filter=['high', 'medium'] if i < 2 else ['high', 'medium', 'low']
                    )
                    
                    # Define stage configuration
                    stage_configs = [
                        {'name': stage_name, 'model_size': 'yolov8n', 'epochs': 50},
                        {'name': stage_name, 'model_size': 'yolov8s', 'epochs': 75}
                    ]
                    
                    # Train multiple model sizes for this stage
                    stage_results = []
                    for config in stage_configs:
                        try:
                            result = await self._train_model_stage(dataset_path, config)
                            stage_results.append(result)
                            model_results.append(result)
                        except Exception as e:
                            error_msg = f"Failed to train {config['model_size']} for {stage_name}: {e}"
                            session_result['errors'].append(error_msg)
                            logger.error(error_msg)
                    
                    if stage_results:
                        session_result['stages_completed'].append(stage_name)
                        logger.info(f"✅ Stage {stage_name} completed with {len(stage_results)} models")
                    
                    # Check if we've reached target accuracy
                    best_stage_accuracy = max((r['final_accuracy'] for r in stage_results), default=0.0)
                    if best_stage_accuracy >= self.target_accuracy:
                        logger.info(f"🎯 Target accuracy reached: {best_stage_accuracy:.2f}%")
                        break
                
                except Exception as e:
                    error_msg = f"Stage {stage_name} failed: {e}"
                    session_result['errors'].append(error_msg)
                    logger.error(error_msg)
            
            # 4. Compare all trained models
            if model_results:
                logger.info("🔍 Comparing trained models...")
                comparison_result = await self._compare_models(model_results)
                
                session_result['model_results'] = model_results
                session_result['best_model'] = comparison_result['best_model']
                session_result['deployment_recommendation'] = comparison_result['recommendation']
                session_result['model_comparison'] = comparison_result
                
                # 5. Auto-deploy if requested and recommended
                if auto_deploy and comparison_result['recommendation'] in ['DEPLOY_IMMEDIATELY', 'DEPLOY_WITH_TESTING']:
                    logger.info("🚀 Auto-deploying best model...")
                    try:
                        deployment_result = await self._deploy_model(comparison_result['best_model'])
                        session_result['deployment_result'] = deployment_result
                    except Exception as e:
                        error_msg = f"Auto-deployment failed: {e}"
                        session_result['errors'].append(error_msg)
                        logger.error(error_msg)
            
            else:
                logger.warning("No models were successfully trained")
                session_result['best_model'] = None
            
            # Final statistics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            session_result['completed_at'] = end_time.isoformat()
            session_result['duration_seconds'] = round(duration, 2)
            session_result['success'] = len(session_result['stages_completed']) > 0
            
            # Update final training status
            final_status = Status.COMPLETED if session_result['success'] else Status.FAILED
            
            update_training_status(
                self.session_id,
                status=final_status,
                progress=1.0,
                current_stage="Training completed"
            )
            
            # Log to database
            self._log_training_session(session_result)
            
            # Save session results
            results_file = Path(f"training_data/experiments/{self.session_id}_results.json")
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(session_result, f, indent=2, default=str)
            
            # Print summary
            self._print_training_summary(session_result)
            
            return session_result
            
        except Exception as e:
            error_msg = f"Progressive training failed: {e}"
            logger.error(error_msg)
            
            session_result['error_message'] = error_msg
            session_result['success'] = False
            session_result['completed_at'] = datetime.now().isoformat()
            
            update_training_status(
                self.session_id,
                status=Status.FAILED,
                error_message=error_msg
            )
            
            raise
    
    async def _deploy_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy trained model to production"""
        # This would be implemented in deploy_improved_model.py
        # Creating placeholder here
        return {
            'deployed': True,
            'model_path': model_info['model_path'],
            'deployment_time': datetime.now().isoformat(),
            'note': 'Placeholder - implement in deploy_improved_model.py'
        }
    
    def _print_training_summary(self, session_result: Dict[str, Any]):
        """Print formatted training summary"""
        print("\n" + "="*60)
        print("🏆 PROGRESSIVE TRAINING SUMMARY")
        print("="*60)
        
        # Basic info
        print(f"\n📋 Session: {session_result['session_id']}")
        print(f"⏱️  Duration: {session_result.get('duration_seconds', 0):.1f} seconds")
        print(f"✅ Stages Completed: {len(session_result['stages_completed'])}/{len(session_result['stages_attempted'])}")
        
        # Data info
        data_analysis = session_result.get('data_analysis', {})
        print(f"📊 Training Data: {data_analysis.get('total_images', 0)} images")
        
        # Model results
        if session_result.get('best_model'):
            best = session_result['best_model']
            print(f"\n🥇 Best Model:")
            print(f"   Model: {best.get('stage_name', 'Unknown')}")
            print(f"   Accuracy: {best.get('final_accuracy', 0):.2f}%")
            print(f"   Improvement: +{best.get('improvement_percent', 0):.2f}%")
            print(f"   Recommendation: {session_result.get('deployment_recommendation', 'NONE')}")
        
        # Errors
        errors = session_result.get('errors', [])
        if errors:
            print(f"\n⚠️  Errors ({len(errors)}):")
            for error in errors[:3]:
                print(f"   • {error}")
            if len(errors) > 3:
                print(f"   ... and {len(errors) - 3} more")
        
        # Success status
        if session_result.get('success'):
            print("\n🎉 Training completed successfully!")
        else:
            print("\n❌ Training had issues")

async def main():
    parser = argparse.ArgumentParser(description="Progressive Soccer Model Training")
    parser.add_argument('--auto-deploy', action='store_true',
                       help='Automatically deploy model if it meets criteria')
    parser.add_argument('--min-improvement', type=float,
                       help='Minimum accuracy improvement required (%)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run with minimal settings for testing')
    
    args = parser.parse_args()
    
    # Adjust for quick test
    if args.quick_test:
        args.min_improvement = 1.0  # Lower threshold for testing
        logger.info("Running in quick test mode")
    
    try:
        trainer = ProgressiveTrainer()
        
        result = await trainer.run_progressive_training(
            auto_deploy=args.auto_deploy,
            min_improvement=args.min_improvement
        )
        
        logger.info("🎉 Progressive training completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Progressive training failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)