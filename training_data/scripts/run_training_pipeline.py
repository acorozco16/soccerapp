#!/usr/bin/env python3
"""
Soccer Training Data Collection and Model Training Pipeline
Orchestrates the entire pipeline from YouTube scraping to model deployment
"""

import sys
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))

from scrapers.youtube_scraper import YouTubeSoccerScraper
from processors.frame_processor import FrameProcessor
from datasets.dataset_manager import DatasetManager
from models.yolo_trainer import YOLOTrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(self, base_dir: str = "./training_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.scraper = YouTubeSoccerScraper(
            output_dir=str(self.base_dir / "scraped_data")
        )
        
        self.processor = FrameProcessor(
            scraped_data_dir=str(self.base_dir / "scraped_data"),
            output_dir=str(self.base_dir / "processed_dataset")
        )
        
        self.dataset_manager = DatasetManager(
            processed_data_dir=str(self.base_dir / "processed_dataset"),
            datasets_dir=str(self.base_dir / "datasets")
        )
        
        self.trainer = YOLOTrainer(
            models_dir=str(self.base_dir / "models"),
            experiments_dir=str(self.base_dir / "experiments")
        )
    
    async def run_data_collection(self, max_videos: int = 100, 
                                max_videos_per_term: int = 15) -> dict:
        """Run YouTube scraping and frame extraction"""
        logger.info("Starting data collection phase...")
        
        try:
            # Scrape YouTube videos
            scraping_summary = await self.scraper.scrape_soccer_videos(
                max_videos_per_term=max_videos_per_term,
                max_total_videos=max_videos
            )
            
            logger.info(f"Data collection complete: {scraping_summary}")
            return scraping_summary
            
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            raise
    
    def run_frame_processing(self, max_frames: int = None) -> dict:
        """Process scraped frames and create annotations"""
        logger.info("Starting frame processing phase...")
        
        try:
            processing_summary = self.processor.process_all_frames(
                max_frames=max_frames
            )
            
            logger.info(f"Frame processing complete: {processing_summary}")
            return processing_summary
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            raise
    
    def run_dataset_creation(self, version: str = None, 
                           max_images: int = 1000,
                           quality_filter: list = None) -> dict:
        """Create YOLO dataset from processed frames"""
        logger.info("Starting dataset creation phase...")
        
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M")
        
        if quality_filter is None:
            quality_filter = ['high', 'medium']  # Skip low quality by default
        
        try:
            # Analyze current data
            stats = self.dataset_manager.analyze_processed_data()
            logger.info(f"Dataset analysis: {stats.total_images} images, {stats.total_labels} labels")
            
            # Create YOLO dataset
            dataset_version = self.dataset_manager.create_yolo_dataset(
                version=version,
                description=f"Soccer ball detection dataset - auto-generated {datetime.now()}",
                max_images=max_images,
                quality_filter=quality_filter
            )
            
            dataset_summary = {
                'version': dataset_version.version,
                'total_images': dataset_version.total_images,
                'train_images': dataset_version.train_images,
                'val_images': dataset_version.val_images,
                'test_images': dataset_version.test_images,
                'quality_distribution': dataset_version.quality_distribution,
                'diversity_score': dataset_version.diversity_metrics.get('diversity_score', 0),
                'path': dataset_version.path
            }
            
            logger.info(f"Dataset creation complete: {dataset_summary}")
            return dataset_summary
            
        except Exception as e:
            logger.error(f"Dataset creation failed: {e}")
            raise
    
    def run_model_training(self, dataset_path: str, dataset_version: str,
                          model_size: str = 'yolov8s', epochs: int = 100) -> dict:
        """Train YOLOv8 model on created dataset"""
        logger.info(f"Starting model training phase with {model_size}...")
        
        try:
            # Check system requirements
            requirements = self.trainer.check_system_requirements()
            logger.info(f"System requirements: {requirements}")
            
            # Get training configuration
            config = self.trainer.default_configs[model_size].copy() if hasattr(self.trainer.default_configs[model_size], 'copy') else self.trainer.default_configs[model_size]
            
            # Create new config with updated parameters
            training_config = TrainingConfig(
                model_name=f"soccer_ball_{model_size}_{dataset_version}",
                base_model=config.base_model,
                dataset_path=dataset_path,
                epochs=epochs,
                batch_size=config.batch_size,
                img_size=config.img_size,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
                warmup_epochs=config.warmup_epochs,
                patience=config.patience,
                device=config.device,
                workers=config.workers
            )
            
            # Train model
            training_result = self.trainer.train_model(training_config, dataset_version)
            
            training_summary = {
                'model_name': training_result.model_name,
                'dataset_version': training_result.dataset_version,
                'training_time': training_result.training_time,
                'best_epoch': training_result.best_epoch,
                'final_metrics': training_result.final_metrics,
                'model_path': training_result.model_path
            }
            
            logger.info(f"Model training complete: {training_summary}")
            return training_summary
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def run_model_evaluation(self, model_path: str, test_dataset_path: str) -> dict:
        """Evaluate trained model performance"""
        logger.info("Starting model evaluation...")
        
        try:
            metrics = self.trainer.evaluate_model(model_path, test_dataset_path)
            
            evaluation_summary = {
                'model_path': model_path,
                'test_dataset': test_dataset_path,
                'evaluation_metrics': metrics,
                'evaluated_at': datetime.now().isoformat()
            }
            
            logger.info(f"Model evaluation complete: {evaluation_summary}")
            return evaluation_summary
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise
    
    def deploy_model(self, deployment_dir: str = "./deployed_model") -> str:
        """Deploy the best trained model"""
        logger.info("Deploying best model...")
        
        try:
            deployed_model_path = self.trainer.deploy_best_model(deployment_dir)
            
            logger.info(f"Model deployed successfully to: {deployed_model_path}")
            return deployed_model_path
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            raise
    
    async def run_full_pipeline(self, config: dict) -> dict:
        """Run the complete training pipeline"""
        logger.info("Starting full training pipeline...")
        
        pipeline_results = {
            'started_at': datetime.now().isoformat(),
            'config': config
        }
        
        try:
            # Phase 1: Data Collection
            if config.get('run_data_collection', True):
                scraping_results = await self.run_data_collection(
                    max_videos=config.get('max_videos', 100),
                    max_videos_per_term=config.get('max_videos_per_term', 15)
                )
                pipeline_results['data_collection'] = scraping_results
            
            # Phase 2: Frame Processing
            if config.get('run_frame_processing', True):
                processing_results = self.run_frame_processing(
                    max_frames=config.get('max_frames', None)
                )
                pipeline_results['frame_processing'] = processing_results
            
            # Phase 3: Dataset Creation
            if config.get('run_dataset_creation', True):
                dataset_results = self.run_dataset_creation(
                    version=config.get('dataset_version'),
                    max_images=config.get('max_images', 1000),
                    quality_filter=config.get('quality_filter', ['high', 'medium'])
                )
                pipeline_results['dataset_creation'] = dataset_results
                
                # Use created dataset for training
                dataset_path = dataset_results['path']
                dataset_version = dataset_results['version']
            else:
                dataset_path = config.get('dataset_path')
                dataset_version = config.get('dataset_version', 'external')
            
            # Phase 4: Model Training
            if config.get('run_model_training', True):
                training_results = self.run_model_training(
                    dataset_path=dataset_path,
                    dataset_version=dataset_version,
                    model_size=config.get('model_size', 'yolov8s'),
                    epochs=config.get('epochs', 100)
                )
                pipeline_results['model_training'] = training_results
                model_path = training_results['model_path']
            else:
                model_path = config.get('model_path')
            
            # Phase 5: Model Evaluation
            if config.get('run_evaluation', True) and model_path:
                evaluation_results = self.run_model_evaluation(
                    model_path=model_path,
                    test_dataset_path=dataset_path
                )
                pipeline_results['evaluation'] = evaluation_results
            
            # Phase 6: Model Deployment
            if config.get('deploy_model', True):
                deployed_model = self.deploy_model(
                    deployment_dir=config.get('deployment_dir', './deployed_model')
                )
                pipeline_results['deployed_model'] = deployed_model
            
            pipeline_results['completed_at'] = datetime.now().isoformat()
            pipeline_results['status'] = 'success'
            
            logger.info("Full pipeline completed successfully!")
            return pipeline_results
            
        except Exception as e:
            pipeline_results['error'] = str(e)
            pipeline_results['status'] = 'failed'
            pipeline_results['failed_at'] = datetime.now().isoformat()
            logger.error(f"Pipeline failed: {e}")
            raise


def create_config_from_args(args) -> dict:
    """Create configuration from command line arguments"""
    return {
        'run_data_collection': args.collect_data,
        'run_frame_processing': args.process_frames,
        'run_dataset_creation': args.create_dataset,
        'run_model_training': args.train_model,
        'run_evaluation': args.evaluate,
        'deploy_model': args.deploy,
        'max_videos': args.max_videos,
        'max_videos_per_term': args.max_videos_per_term,
        'max_frames': args.max_frames,
        'max_images': args.max_images,
        'quality_filter': args.quality_filter,
        'model_size': args.model_size,
        'epochs': args.epochs,
        'dataset_version': args.dataset_version,
        'dataset_path': args.dataset_path,
        'deployment_dir': args.deployment_dir
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Soccer Training Data Collection and Model Training Pipeline"
    )
    
    # Phase control
    parser.add_argument('--collect-data', action='store_true', default=True,
                       help='Run data collection from YouTube')
    parser.add_argument('--process-frames', action='store_true', default=True,
                       help='Process frames and create annotations')
    parser.add_argument('--create-dataset', action='store_true', default=True,
                       help='Create YOLO training dataset')
    parser.add_argument('--train-model', action='store_true', default=True,
                       help='Train YOLOv8 model')
    parser.add_argument('--evaluate', action='store_true', default=True,
                       help='Evaluate trained model')
    parser.add_argument('--deploy', action='store_true', default=True,
                       help='Deploy best model')
    
    # Data collection parameters
    parser.add_argument('--max-videos', type=int, default=50,
                       help='Maximum number of videos to download')
    parser.add_argument('--max-videos-per-term', type=int, default=10,
                       help='Maximum videos per search term')
    
    # Processing parameters
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process (None for all)')
    parser.add_argument('--max-images', type=int, default=500,
                       help='Maximum images in final dataset')
    parser.add_argument('--quality-filter', nargs='+', 
                       default=['high', 'medium'],
                       choices=['high', 'medium', 'low'],
                       help='Quality levels to include')
    
    # Training parameters
    parser.add_argument('--model-size', choices=['yolov8n', 'yolov8s', 'yolov8m'],
                       default='yolov8s', help='YOLOv8 model size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    
    # Optional paths
    parser.add_argument('--dataset-version', type=str, default=None,
                       help='Dataset version identifier')
    parser.add_argument('--dataset-path', type=str, default=None,
                       help='Path to existing dataset (skip creation)')
    parser.add_argument('--deployment-dir', type=str, default='./deployed_model',
                       help='Directory for model deployment')
    
    # Pipeline control
    parser.add_argument('--quick-test', action='store_true',
                       help='Run with minimal data for testing')
    
    args = parser.parse_args()
    
    # Adjust parameters for quick test
    if args.quick_test:
        args.max_videos = 5
        args.max_videos_per_term = 2
        args.max_frames = 50
        args.max_images = 20
        args.epochs = 5
        logger.info("Running in quick test mode with minimal data")
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Initialize and run pipeline
    pipeline = TrainingPipeline()
    
    try:
        results = await pipeline.run_full_pipeline(config)
        
        print("\n" + "="*50)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        if 'data_collection' in results:
            dc = results['data_collection']
            print(f"\nData Collection:")
            print(f"  Videos processed: {dc.get('total_videos_processed', 0)}")
            print(f"  Frames extracted: {dc.get('total_frames_extracted', 0)}")
        
        if 'frame_processing' in results:
            fp = results['frame_processing']
            print(f"\nFrame Processing:")
            print(f"  Annotations created: {fp.get('total_annotations', 0)}")
            print(f"  Bounding boxes: {fp.get('total_bounding_boxes', 0)}")
        
        if 'dataset_creation' in results:
            dc = results['dataset_creation']
            print(f"\nDataset Creation:")
            print(f"  Version: {dc.get('version')}")
            print(f"  Total images: {dc.get('total_images')}")
            print(f"  Train/Val/Test: {dc.get('train_images')}/{dc.get('val_images')}/{dc.get('test_images')}")
        
        if 'model_training' in results:
            mt = results['model_training']
            print(f"\nModel Training:")
            print(f"  Model: {mt.get('model_name')}")
            print(f"  Training time: {mt.get('training_time', 0):.1f}s")
            print(f"  Best epoch: {mt.get('best_epoch')}")
            
            metrics = mt.get('final_metrics', {})
            print(f"  Final mAP@0.5: {metrics.get('mAP50', 0):.3f}")
            print(f"  Final precision: {metrics.get('precision', 0):.3f}")
            print(f"  Final recall: {metrics.get('recall', 0):.3f}")
        
        if 'deployed_model' in results:
            print(f"\nModel Deployment:")
            print(f"  Deployed to: {results['deployed_model']}")
        
        print(f"\nTotal pipeline time: {results.get('completed_at')} - {results.get('started_at')}")
        
    except Exception as e:
        print(f"\nPIPELINE FAILED: {e}")
        logger.error(f"Pipeline execution failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)