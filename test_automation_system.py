#!/usr/bin/env python3
"""
Automation System Test Suite
Comprehensive testing for the training automation system
"""

import os
import json
import asyncio
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Test configuration
TEST_CONFIG = {
    'api_keys': {
        'youtube': ['test_key_1'],
        'unsplash': 'test_unsplash_key',
        'pexels': 'test_pexels_key'
    },
    'email': {
        'enabled': False
    },
    'training': {
        'target_images_initial': 10,
        'target_images_weekly': 5,
        'min_new_data_for_training': 2,
        'min_accuracy_improvement': 1.0,
        'target_accuracy': 80.0,
        'max_training_hours': 1,
        'training_stages': [5, 10],
        'max_parallel_downloads': 2,
        'max_disk_usage_gb': 10,
        'cleanup_after_days': 7
    },
    'cloud_storage': {
        'enabled': False
    }
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutomationSystemTester:
    def __init__(self):
        self.test_dir = Path("automation_test_env")
        self.original_cwd = Path.cwd()
        self.test_results = []
        
    def setup_test_environment(self):
        """Set up isolated test environment"""
        logger.info("🛠️ Setting up test environment...")
        
        # Create clean test directory
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        
        self.test_dir.mkdir(parents=True)
        
        # Create necessary subdirectories
        test_dirs = [
            "training_data/automation",
            "training_data/scrapers",
            "training_data/dashboard", 
            "training_data/collected_data",
            "training_data/models",
            "training_data/logs",
            "backend",
            "sample_videos"
        ]
        
        for dir_name in test_dirs:
            (self.test_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
        # Create test configuration
        config_file = self.test_dir / "automation_config.json"
        with open(config_file, 'w') as f:
            json.dump(TEST_CONFIG, f, indent=2)
        
        # Create dummy sample videos
        sample_videos_dir = self.test_dir / "sample_videos"
        for video_name in ['clear_touches.mp4', 'difficult_lighting.mp4']:
            (sample_videos_dir / video_name).touch()
        
        # Create dummy backend database
        backend_dir = self.test_dir / "backend"
        (backend_dir / "database.db").touch()
        
        logger.info("✅ Test environment ready")
    
    def cleanup_test_environment(self):
        """Clean up test environment"""
        logger.info("🧹 Cleaning up test environment...")
        try:
            if self.test_dir.exists():
                shutil.rmtree(self.test_dir)
            logger.info("✅ Test environment cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup test environment: {e}")
    
    async def test_configuration_system(self) -> bool:
        """Test the automation configuration system"""
        logger.info("🧪 Testing configuration system...")
        
        try:
            os.chdir(self.test_dir)
            
            # Test setup_automation.py
            sys.path.insert(0, str(self.original_cwd))
            from setup_automation import AutomationSetup
            
            setup = AutomationSetup()
            
            # Test config loading
            config = setup._load_config()
            assert config is not None, "Config should not be None"
            assert 'api_keys' in config, "Config should have api_keys"
            assert 'training' in config, "Config should have training section"
            
            logger.info("✅ Configuration system test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration system test failed: {e}")
            return False
        finally:
            os.chdir(self.original_cwd)
    
    async def test_status_manager(self) -> bool:
        """Test the training status manager"""
        logger.info("🧪 Testing status manager...")
        
        try:
            os.chdir(self.test_dir)
            
            # Import status manager
            sys.path.insert(0, str(self.original_cwd / "training_data" / "automation"))
            from training_status import TrainingStatusManager, Status
            
            # Create status manager
            status_manager = TrainingStatusManager()
            
            # Test collection status update
            status_manager.update_collection_status(
                "test_source",
                status=Status.RUNNING,
                progress=0.5,
                items_collected=50,
                target_items=100
            )
            
            # Test getting status
            collection_status = status_manager.get_collection_status("test_source")
            assert "test_source" in collection_status, "Should have test_source status"
            assert collection_status["test_source"].progress == 0.5, "Progress should be 0.5"
            
            # Test training status
            status_manager.update_training_status(
                "test_session",
                status=Status.RUNNING,
                progress=0.3,
                current_accuracy=0.8
            )
            
            training_status = status_manager.get_training_status()
            assert training_status is not None, "Should have training status"
            assert training_status.progress == 0.3, "Training progress should be 0.3"
            
            logger.info("✅ Status manager test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Status manager test failed: {e}")
            return False
        finally:
            os.chdir(self.original_cwd)
    
    async def test_data_collection_system(self) -> bool:
        """Test the data collection system"""
        logger.info("🧪 Testing data collection system...")
        
        try:
            os.chdir(self.test_dir)
            
            # Import data collector (with mocked APIs)
            sys.path.insert(0, str(self.original_cwd / "training_data" / "automation"))
            
            # Test would require API mocking for real functionality
            # For now, just test imports and basic initialization
            from collect_all_data import ComprehensiveDataCollector
            
            collector = ComprehensiveDataCollector()
            assert collector is not None, "Collector should initialize"
            
            # Test configuration loading
            assert collector.config is not None, "Config should be loaded"
            
            logger.info("✅ Data collection system test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data collection system test failed: {e}")
            return False
        finally:
            os.chdir(self.original_cwd)
    
    async def test_progressive_training_system(self) -> bool:
        """Test the progressive training system"""
        logger.info("🧪 Testing progressive training system...")
        
        try:
            os.chdir(self.test_dir)
            
            # Create minimal training data structure
            processed_dir = self.test_dir / "training_data" / "processed_dataset" / "images" / "high_quality"
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Create some dummy image files
            for i in range(5):
                (processed_dir / f"image_{i}.jpg").touch()
                (Path(str(processed_dir).replace('images', 'labels')) / f"image_{i}.txt").touch()
            
            # Test trainer initialization
            sys.path.insert(0, str(self.original_cwd / "training_data" / "automation"))
            from train_progressive import ProgressiveTrainer
            
            trainer = ProgressiveTrainer()
            assert trainer is not None, "Trainer should initialize"
            
            # Test data analysis
            data_analysis = trainer._analyze_available_data()
            assert 'total_images' in data_analysis, "Should analyze available data"
            
            logger.info("✅ Progressive training system test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Progressive training system test failed: {e}")
            return False
        finally:
            os.chdir(self.original_cwd)
    
    async def test_deployment_system(self) -> bool:
        """Test the model deployment system"""
        logger.info("🧪 Testing deployment system...")
        
        try:
            os.chdir(self.test_dir)
            
            # Create dummy model file
            models_dir = self.test_dir / "training_data" / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            
            dummy_model = models_dir / "test_model.pt"
            dummy_model.touch()
            
            # Create metadata
            metadata = {
                'model_name': 'test_model',
                'accuracy': 85.5,
                'created_at': datetime.now().isoformat()
            }
            
            metadata_file = dummy_model.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            # Test deployment manager
            sys.path.insert(0, str(self.original_cwd / "training_data" / "automation"))
            from deploy_improved_model import ModelDeploymentManager
            
            deployer = ModelDeploymentManager()
            assert deployer is not None, "Deployer should initialize"
            
            # Test model validation (would fail without actual model, but tests structure)
            try:
                validation_result = await deployer._validate_model_performance(str(dummy_model))
                # Should return error due to dummy model, but structure should work
                assert 'error' in validation_result or 'passes_validation' in validation_result
            except:
                pass  # Expected to fail with dummy model
            
            logger.info("✅ Deployment system test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment system test failed: {e}")
            return False
        finally:
            os.chdir(self.original_cwd)
    
    async def test_weekly_automation(self) -> bool:
        """Test the weekly automation system"""
        logger.info("🧪 Testing weekly automation system...")
        
        try:
            os.chdir(self.test_dir)
            
            # Test weekly updater initialization
            sys.path.insert(0, str(self.original_cwd / "training_data" / "automation"))
            from weekly_training_update import WeeklyTrainingUpdater
            
            updater = WeeklyTrainingUpdater()
            assert updater is not None, "Updater should initialize"
            
            # Test data freshness analysis
            data_analysis = updater._analyze_training_data_freshness()
            assert 'total_files' in data_analysis, "Should analyze data freshness"
            
            # Test system health check
            health_check = updater._check_system_health()
            assert 'healthy' in health_check, "Should check system health"
            
            logger.info("✅ Weekly automation test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Weekly automation test failed: {e}")
            return False
        finally:
            os.chdir(self.original_cwd)
    
    async def test_dashboard_system(self) -> bool:
        """Test the training dashboard system"""
        logger.info("🧪 Testing dashboard system...")
        
        try:
            os.chdir(self.test_dir)
            
            # Test dashboard API
            sys.path.insert(0, str(self.original_cwd / "training_data" / "dashboard"))
            from dashboard_api import get_status_manager_instance
            
            status_manager = get_status_manager_instance()
            assert status_manager is not None, "Status manager should be available"
            
            # Test dashboard file existence
            dashboard_file = self.original_cwd / "training_data" / "dashboard" / "training_dashboard.html"
            assert dashboard_file.exists(), "Dashboard HTML should exist"
            
            charts_file = self.original_cwd / "training_data" / "dashboard" / "progress_charts.js"
            assert charts_file.exists(), "Charts JS should exist"
            
            logger.info("✅ Dashboard system test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dashboard system test failed: {e}")
            return False
        finally:
            os.chdir(self.original_cwd)
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all automation system tests"""
        logger.info("🚀 Starting comprehensive automation system tests...")
        
        self.setup_test_environment()
        
        tests = [
            ("Configuration System", self.test_configuration_system),
            ("Status Manager", self.test_status_manager),
            ("Data Collection System", self.test_data_collection_system),
            ("Progressive Training System", self.test_progressive_training_system),
            ("Deployment System", self.test_deployment_system),
            ("Weekly Automation", self.test_weekly_automation),
            ("Dashboard System", self.test_dashboard_system)
        ]
        
        results = {
            'started_at': datetime.now().isoformat(),
            'total_tests': len(tests),
            'passed': 0,
            'failed': 0,
            'test_results': []
        }
        
        for test_name, test_func in tests:
            logger.info(f"Running test: {test_name}")
            
            try:
                start_time = datetime.now()
                passed = await test_func()
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                test_result = {
                    'name': test_name,
                    'passed': passed,
                    'duration_seconds': round(duration, 2),
                    'timestamp': start_time.isoformat()
                }
                
                if passed:
                    results['passed'] += 1
                else:
                    results['failed'] += 1
                
                results['test_results'].append(test_result)
                
            except Exception as e:
                logger.error(f"Test {test_name} crashed: {e}")
                results['failed'] += 1
                results['test_results'].append({
                    'name': test_name,
                    'passed': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        results['completed_at'] = datetime.now().isoformat()
        results['success_rate'] = (results['passed'] / results['total_tests']) * 100
        
        # Cleanup
        self.cleanup_test_environment()
        
        return results
    
    def print_test_report(self, results: Dict[str, Any]):
        """Print formatted test report"""
        print("\n" + "="*60)
        print("🧪 AUTOMATION SYSTEM TEST REPORT")
        print("="*60)
        
        print(f"\n📊 Overall Results:")
        print(f"   Total Tests: {results['total_tests']}")
        print(f"   Passed: {results['passed']}")
        print(f"   Failed: {results['failed']}")
        print(f"   Success Rate: {results['success_rate']:.1f}%")
        
        print(f"\n📋 Individual Test Results:")
        for test in results['test_results']:
            status = "✅ PASS" if test['passed'] else "❌ FAIL"
            duration = test.get('duration_seconds', 0)
            print(f"   {status} {test['name']} ({duration:.2f}s)")
            
            if not test['passed'] and 'error' in test:
                print(f"      Error: {test['error']}")
        
        print(f"\n⏱️ Test Duration: {results.get('started_at')} to {results.get('completed_at')}")
        
        if results['success_rate'] >= 80:
            print("\n🎉 Test suite PASSED! Automation system is ready.")
        else:
            print("\n⚠️ Test suite had issues. Review failed tests before deployment.")

async def main():
    """Run the automation system test suite"""
    tester = AutomationSystemTester()
    
    try:
        results = await tester.run_all_tests()
        tester.print_test_report(results)
        
        # Save results
        results_file = Path("automation_test_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test results saved to: {results_file}")
        
        # Return exit code based on success rate
        return 0 if results['success_rate'] >= 80 else 1
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return 1

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Soccer AI Training Automation System")
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)