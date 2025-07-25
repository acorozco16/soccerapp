#!/usr/bin/env python3
"""
YOLOv8 Training Pipeline and Model Deployment System
Fine-tunes YOLOv8 models on soccer ball detection data
"""

import os
import json
import shutil
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    model_name: str
    base_model: str  # 'yolov8n', 'yolov8s', 'yolov8m', etc.
    dataset_path: str
    epochs: int
    batch_size: int
    img_size: int
    learning_rate: float
    weight_decay: float
    momentum: float
    warmup_epochs: int
    patience: int  # Early stopping patience
    device: str
    workers: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TrainingResults:
    model_name: str
    dataset_version: str
    training_time: float
    best_epoch: int
    final_metrics: Dict[str, float]
    model_path: str
    config_used: TrainingConfig
    created_at: datetime
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'config_used': self.config_used.to_dict(),
            'created_at': self.created_at.isoformat()
        }


@dataclass
class ModelComparison:
    baseline_model: str
    new_model: str
    baseline_metrics: Dict[str, float]
    new_metrics: Dict[str, float]
    improvement: Dict[str, float]
    test_results: Dict[str, List[float]]


class YOLOTrainer:
    def __init__(self, models_dir: str = "./models", experiments_dir: str = "./experiments"):
        self.models_dir = Path(models_dir)
        self.experiments_dir = Path(experiments_dir)
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.training_history: List[TrainingResults] = []
        self.history_file = self.models_dir / "training_history.json"
        self._load_training_history()
        
        # Default configurations for different model sizes
        self.default_configs = {
            'yolov8n': TrainingConfig(
                model_name="soccer_ball_nano",
                base_model="yolov8n.pt",
                dataset_path="",
                epochs=100,
                batch_size=32,
                img_size=640,
                learning_rate=0.01,
                weight_decay=0.0005,
                momentum=0.937,
                warmup_epochs=3,
                patience=30,
                device="auto",
                workers=8
            ),
            'yolov8s': TrainingConfig(
                model_name="soccer_ball_small",
                base_model="yolov8s.pt",
                dataset_path="",
                epochs=150,
                batch_size=24,
                img_size=640,
                learning_rate=0.01,
                weight_decay=0.0005,
                momentum=0.937,
                warmup_epochs=3,
                patience=40,
                device="auto",
                workers=8
            ),
            'yolov8m': TrainingConfig(
                model_name="soccer_ball_medium",
                base_model="yolov8m.pt",
                dataset_path="",
                epochs=200,
                batch_size=16,
                img_size=640,
                learning_rate=0.01,
                weight_decay=0.0005,
                momentum=0.937,
                warmup_epochs=5,
                patience=50,
                device="auto",
                workers=8
            )
        }
    
    def _load_training_history(self):
        """Load training history from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    for item in data.get('training_runs', []):
                        # Reconstruct TrainingConfig
                        config_data = item['config_used']
                        config = TrainingConfig(**config_data)
                        
                        # Reconstruct TrainingResults
                        result = TrainingResults(
                            model_name=item['model_name'],
                            dataset_version=item['dataset_version'],
                            training_time=item['training_time'],
                            best_epoch=item['best_epoch'],
                            final_metrics=item['final_metrics'],
                            model_path=item['model_path'],
                            config_used=config,
                            created_at=datetime.fromisoformat(item['created_at'])
                        )
                        self.training_history.append(result)
                
                logger.info(f"Loaded {len(self.training_history)} training runs")
            except Exception as e:
                logger.error(f"Error loading training history: {e}")
    
    def _save_training_history(self):
        """Save training history to file"""
        data = {
            'training_runs': [result.to_dict() for result in self.training_history],
            'total_runs': len(self.training_history),
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.history_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def check_system_requirements(self) -> Dict[str, bool]:
        """Check system requirements for training"""
        requirements = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'torch_version': torch.__version__,
            'ultralytics_available': True
        }
        
        # Check GPU memory if CUDA is available
        if requirements['cuda_available']:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            requirements['gpu_memory_gb'] = gpu_memory
            requirements['sufficient_gpu_memory'] = gpu_memory > 4  # At least 4GB
        
        logger.info(f"System requirements: {requirements}")
        return requirements
    
    def prepare_training_environment(self, config: TrainingConfig, 
                                   dataset_version: str) -> Path:
        """Prepare training environment and directories"""
        # Create experiment directory
        experiment_name = f"{config.model_name}_{dataset_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_dir = self.experiments_dir / experiment_name
        experiment_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (experiment_dir / "runs").mkdir(exist_ok=True)
        (experiment_dir / "models").mkdir(exist_ok=True)
        (experiment_dir / "plots").mkdir(exist_ok=True)
        
        # Save configuration
        config_path = experiment_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
        
        logger.info(f"Training environment prepared: {experiment_dir}")
        return experiment_dir
    
    def train_model(self, config: TrainingConfig, dataset_version: str, 
                   resume: bool = False) -> TrainingResults:
        """Train YOLOv8 model"""
        logger.info(f"Starting training: {config.model_name} on dataset {dataset_version}")
        
        # Check system requirements
        requirements = self.check_system_requirements()
        if not requirements['ultralytics_available']:
            raise RuntimeError("Ultralytics not available")
        
        # Prepare environment
        experiment_dir = self.prepare_training_environment(config, dataset_version)
        
        # Validate dataset path
        dataset_path = Path(config.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        dataset_yaml = dataset_path / "dataset.yaml"
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
        
        try:
            # Initialize model
            model = YOLO(config.base_model)
            
            # Set up training arguments
            train_args = {
                'data': str(dataset_yaml),
                'epochs': config.epochs,
                'batch': config.batch_size,
                'imgsz': config.img_size,
                'lr0': config.learning_rate,
                'weight_decay': config.weight_decay,
                'momentum': config.momentum,
                'warmup_epochs': config.warmup_epochs,
                'patience': config.patience,
                'device': config.device,
                'workers': config.workers,
                'project': str(experiment_dir / "runs"),
                'name': 'train',
                'exist_ok': True,
                'pretrained': True,
                'optimizer': 'AdamW',
                'verbose': True,
                'seed': 42,
                'deterministic': True,
                'single_cls': True,  # Single class (soccer ball)
                'rect': False,  # Rectangular training
                'cos_lr': True,  # Cosine learning rate scheduler
                'close_mosaic': 10,  # Close mosaic augmentation in last N epochs
                'resume': resume
            }
            
            # Start training
            start_time = time.time()
            logger.info(f"Training started with args: {train_args}")
            
            results = model.train(**train_args)
            
            training_time = time.time() - start_time
            
            # Get best model path
            best_model_path = experiment_dir / "runs" / "train" / "weights" / "best.pt"
            
            # Copy best model to models directory
            final_model_name = f"{config.model_name}_{dataset_version}.pt"
            final_model_path = self.models_dir / final_model_name
            shutil.copy2(best_model_path, final_model_path)
            
            # Extract final metrics
            metrics_file = experiment_dir / "runs" / "train" / "results.csv"
            final_metrics = self._extract_final_metrics(metrics_file)
            
            # Find best epoch
            best_epoch = self._find_best_epoch(metrics_file)
            
            # Create training results
            training_result = TrainingResults(
                model_name=config.model_name,
                dataset_version=dataset_version,
                training_time=training_time,
                best_epoch=best_epoch,
                final_metrics=final_metrics,
                model_path=str(final_model_path),
                config_used=config,
                created_at=datetime.now()
            )
            
            # Save results
            self.training_history.append(training_result)
            self._save_training_history()
            
            # Generate training plots
            self._generate_training_plots(experiment_dir)
            
            logger.info(f"Training completed successfully in {training_time:.1f}s")
            logger.info(f"Best model saved to: {final_model_path}")
            logger.info(f"Final metrics: {final_metrics}")
            
            return training_result
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _extract_final_metrics(self, results_csv: Path) -> Dict[str, float]:
        """Extract final metrics from training results"""
        try:
            import pandas as pd
            df = pd.read_csv(results_csv)
            
            # Get metrics from best epoch (lowest validation loss)
            best_idx = df['val/box_loss'].idxmin()
            best_row = df.iloc[best_idx]
            
            metrics = {
                'precision': float(best_row.get('metrics/precision(B)', 0)),
                'recall': float(best_row.get('metrics/recall(B)', 0)),
                'mAP50': float(best_row.get('metrics/mAP50(B)', 0)),
                'mAP50_95': float(best_row.get('metrics/mAP50-95(B)', 0)),
                'box_loss': float(best_row.get('val/box_loss', 0)),
                'cls_loss': float(best_row.get('val/cls_loss', 0)),
                'dfl_loss': float(best_row.get('val/dfl_loss', 0))
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting metrics: {e}")
            return {}
    
    def _find_best_epoch(self, results_csv: Path) -> int:
        """Find the best epoch based on validation metrics"""
        try:
            import pandas as pd
            df = pd.read_csv(results_csv)
            
            # Find epoch with best mAP50
            best_idx = df['metrics/mAP50(B)'].idxmax()
            return int(df.iloc[best_idx]['epoch']) + 1  # 1-indexed
            
        except Exception as e:
            logger.error(f"Error finding best epoch: {e}")
            return 0
    
    def _generate_training_plots(self, experiment_dir: Path):
        """Generate training visualization plots"""
        try:
            results_csv = experiment_dir / "runs" / "train" / "results.csv"
            if not results_csv.exists():
                return
            
            import pandas as pd
            df = pd.read_csv(results_csv)
            
            # Create plots directory
            plots_dir = experiment_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Plot training curves
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Training Progress', fontsize=16)
            
            # Loss curves
            axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
            axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
            axes[0, 0].set_title('Box Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # mAP curves
            axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
            axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
            axes[0, 1].set_title('Mean Average Precision')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('mAP')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Precision and Recall
            axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
            axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
            axes[1, 0].set_title('Precision and Recall')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Learning rate
            if 'lr/pg0' in df.columns:
                axes[1, 1].plot(df['epoch'], df['lr/pg0'])
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "training_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training plots saved to {plots_dir}")
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
    
    def evaluate_model(self, model_path: str, dataset_path: str, 
                      img_size: int = 640) -> Dict[str, float]:
        """Evaluate trained model on test set"""
        try:
            # Load model
            model = YOLO(model_path)
            
            # Run validation
            results = model.val(
                data=str(Path(dataset_path) / "dataset.yaml"),
                imgsz=img_size,
                batch=16,
                conf=0.001,  # Low confidence threshold for evaluation
                iou=0.6,
                max_det=300,
                half=False,
                device='auto',
                plots=True,
                save_json=True,
                verbose=True
            )
            
            # Extract metrics
            metrics = {
                'precision': float(results.box.p[0]) if len(results.box.p) > 0 else 0.0,
                'recall': float(results.box.r[0]) if len(results.box.r) > 0 else 0.0,
                'mAP50': float(results.box.map50),
                'mAP50_95': float(results.box.map),
                'fitness': float(results.fitness)
            }
            
            logger.info(f"Model evaluation complete: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    def compare_models(self, baseline_model_path: str, new_model_path: str, 
                      test_dataset_path: str) -> ModelComparison:
        """Compare two models on test dataset"""
        logger.info(f"Comparing models: {baseline_model_path} vs {new_model_path}")
        
        # Evaluate both models
        baseline_metrics = self.evaluate_model(baseline_model_path, test_dataset_path)
        new_metrics = self.evaluate_model(new_model_path, test_dataset_path)
        
        # Calculate improvements
        improvement = {}
        for metric, new_val in new_metrics.items():
            baseline_val = baseline_metrics.get(metric, 0)
            if baseline_val > 0:
                improvement[metric] = ((new_val - baseline_val) / baseline_val) * 100
            else:
                improvement[metric] = 0
        
        # Create comparison object
        comparison = ModelComparison(
            baseline_model=baseline_model_path,
            new_model=new_model_path,
            baseline_metrics=baseline_metrics,
            new_metrics=new_metrics,
            improvement=improvement,
            test_results={
                'baseline': list(baseline_metrics.values()),
                'new_model': list(new_metrics.values())
            }
        )
        
        # Save comparison results
        comparison_file = self.models_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_file, 'w') as f:
            json.dump(asdict(comparison), f, indent=2, default=str)
        
        logger.info(f"Model comparison saved to {comparison_file}")
        return comparison
    
    def deploy_best_model(self, deployment_dir: str) -> str:
        """Deploy the best performing model"""
        if not self.training_history:
            raise ValueError("No trained models available")
        
        # Find best model based on mAP50
        best_result = max(
            self.training_history, 
            key=lambda x: x.final_metrics.get('mAP50', 0)
        )
        
        deployment_path = Path(deployment_dir)
        deployment_path.mkdir(parents=True, exist_ok=True)
        
        # Copy best model
        best_model_path = deployment_path / "best_soccer_ball_model.pt"
        shutil.copy2(best_result.model_path, best_model_path)
        
        # Create deployment info
        deployment_info = {
            'model_path': str(best_model_path),
            'model_name': best_result.model_name,
            'dataset_version': best_result.dataset_version,
            'metrics': best_result.final_metrics,
            'deployed_at': datetime.now().isoformat(),
            'config': best_result.config_used.to_dict()
        }
        
        info_file = deployment_path / "deployment_info.json"
        with open(info_file, 'w') as f:
            json.dump(deployment_info, f, indent=2, default=str)
        
        logger.info(f"Best model deployed to {best_model_path}")
        logger.info(f"Model metrics: {best_result.final_metrics}")
        
        return str(best_model_path)


def main():
    """Example usage"""
    trainer = YOLOTrainer(
        models_dir="./training_data/models",
        experiments_dir="./training_data/experiments"
    )
    
    try:
        # Check system requirements
        requirements = trainer.check_system_requirements()
        print("\n=== SYSTEM REQUIREMENTS ===")
        for key, value in requirements.items():
            print(f"{key}: {value}")
        
        # Configure training
        config = trainer.default_configs['yolov8s']
        config.dataset_path = "./training_data/datasets/yolo_v20240101_1200"  # Update with actual path
        config.epochs = 50  # Reduce for testing
        
        # Train model
        print("\n=== STARTING TRAINING ===")
        result = trainer.train_model(config, "20240101_1200")
        
        print(f"\n=== TRAINING COMPLETE ===")
        print(f"Model: {result.model_name}")
        print(f"Training time: {result.training_time:.1f}s")
        print(f"Best epoch: {result.best_epoch}")
        print(f"Final metrics: {result.final_metrics}")
        
        # Deploy best model
        deployed_model = trainer.deploy_best_model("./deployed_model")
        print(f"\nBest model deployed to: {deployed_model}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")


if __name__ == "__main__":
    main()