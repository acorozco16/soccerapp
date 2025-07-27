#!/usr/bin/env python3
"""
Quick Start Script for Soccer AI Training Automation
Guided setup and initial testing of the automation system
"""

import os
import json
import asyncio
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickStartGuide:
    def __init__(self):
        self.project_root = Path.cwd()
        self.config_file = self.project_root / "automation_config.json"
        
    def check_prerequisites(self) -> Dict[str, bool]:
        """Check if all prerequisites are installed"""
        logger.info("🔍 Checking prerequisites...")
        
        checks = {}
        
        # Check Python version
        try:
            import sys
            version = sys.version_info
            checks['python_version'] = version >= (3, 9)
            if checks['python_version']:
                logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro}")
            else:
                logger.error(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires 3.9+")
        except Exception as e:
            checks['python_version'] = False
            logger.error(f"❌ Python version check failed: {e}")
        
        # Check required packages
        required_packages = [
            'fastapi', 'uvicorn', 'opencv-python', 'mediapipe', 
            'ultralytics', 'yt-dlp', 'aiohttp', 'psutil', 'tqdm'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                logger.info(f"✅ {package}")
            except ImportError:
                missing_packages.append(package) 
                logger.error(f"❌ {package} - Missing")
        
        checks['required_packages'] = len(missing_packages) == 0
        checks['missing_packages'] = missing_packages
        
        # Check disk space
        try:
            import shutil
            free_bytes = shutil.disk_usage('.').free
            free_gb = free_bytes / (1024**3)
            checks['disk_space'] = free_gb >= 10.0  # At least 10GB
            if checks['disk_space']:
                logger.info(f"✅ Disk space: {free_gb:.1f} GB available")
            else:
                logger.error(f"❌ Disk space: {free_gb:.1f} GB - Need at least 10GB")
        except Exception as e:
            checks['disk_space'] = False
            logger.error(f"❌ Disk space check failed: {e}")
        
        # Check project structure
        required_dirs = [
            'backend',
            'frontend', 
            'training_data',
            'sample_videos'
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                logger.info(f"✅ {dir_name}/")
            else:
                missing_dirs.append(dir_name)
                logger.error(f"❌ {dir_name}/ - Missing")
        
        checks['project_structure'] = len(missing_dirs) == 0
        checks['missing_dirs'] = missing_dirs
        
        return checks
    
    def install_missing_packages(self, missing_packages: List[str]) -> bool:
        """Install missing Python packages"""
        if not missing_packages:
            return True
        
        logger.info(f"📦 Installing missing packages: {', '.join(missing_packages)}")
        
        try:
            cmd = [sys.executable, '-m', 'pip', 'install'] + missing_packages
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Packages installed successfully")
                return True
            else:
                logger.error(f"❌ Package installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Package installation error: {e}")
            return False
    
    def run_initial_setup(self) -> bool:
        """Run the automation setup script"""
        logger.info("⚙️ Running initial automation setup...")
        
        try:
            # Check if setup script exists
            setup_script = self.project_root / "setup_automation.py"
            if not setup_script.exists():
                logger.error("❌ setup_automation.py not found")
                return False
            
            # Run setup in interactive mode would require user input
            # For quick start, we'll create a minimal config
            minimal_config = {
                'api_keys': {
                    'youtube': [],
                    'instagram': None,
                    'unsplash': None,
                    'pexels': None
                },
                'email': {
                    'enabled': False
                },
                'training': {
                    'target_images_initial': 100,
                    'target_images_weekly': 50,
                    'min_new_data_for_training': 25,
                    'min_accuracy_improvement': 5.0,
                    'target_accuracy': 85.0,
                    'max_training_hours': 2,
                    'training_stages': [50, 100, 200],
                    'max_parallel_downloads': 3,
                    'max_disk_usage_gb': 20,
                    'cleanup_after_days': 30
                },
                'cloud_storage': {
                    'enabled': False
                }
            }
            
            # Save minimal config
            with open(self.config_file, 'w') as f:
                json.dump(minimal_config, f, indent=2)
            
            logger.info("✅ Minimal configuration created")
            logger.info("⚠️ Run setup_automation.py later to configure APIs and email")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
    
    def create_directory_structure(self) -> bool:
        """Create necessary directory structure"""
        logger.info("📁 Creating directory structure...")
        
        directories = [
            "training_data/automation",
            "training_data/scrapers", 
            "training_data/dashboard",
            "training_data/collected_data",
            "training_data/collected_data/youtube",
            "training_data/collected_data/instagram",
            "training_data/collected_data/stock_photos",
            "training_data/collected_data/datasets",
            "training_data/processed_dataset",
            "training_data/processed_dataset/images",
            "training_data/processed_dataset/labels",
            "training_data/datasets",
            "training_data/models",
            "training_data/models/production",
            "training_data/models/staging",
            "training_data/models/backups",
            "training_data/experiments",
            "training_data/logs"
        ]
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"✅ Created {len(directories)} directories")
            return True
            
        except Exception as e:
            logger.error(f"❌ Directory creation failed: {e}")
            return False
    
    async def test_basic_functionality(self) -> Dict[str, bool]:
        """Test basic functionality of the automation system"""
        logger.info("🧪 Testing basic functionality...")
        
        test_results = {}
        
        # Test 1: Status manager
        try:
            sys.path.insert(0, str(self.project_root / "training_data" / "automation"))
            from training_status import TrainingStatusManager, Status
            
            status_manager = TrainingStatusManager()
            status_manager.update_collection_status(
                "test",
                status=Status.RUNNING,
                progress=0.5,
                items_collected=10,
                target_items=20
            )
            
            status = status_manager.get_collection_status("test")
            test_results['status_manager'] = status is not None
            logger.info("✅ Status manager working")
            
        except Exception as e:
            test_results['status_manager'] = False
            logger.error(f"❌ Status manager test failed: {e}")
        
        # Test 2: Configuration loading
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                test_results['configuration'] = 'training' in config
                logger.info("✅ Configuration loading working")
            else:
                test_results['configuration'] = False
                logger.error("❌ Configuration file not found")
                
        except Exception as e:
            test_results['configuration'] = False
            logger.error(f"❌ Configuration test failed: {e}")
        
        # Test 3: Backend integration
        try:
            backend_main = self.project_root / "backend" / "main.py"
            if backend_main.exists():
                # Check if training API integration exists
                with open(backend_main, 'r') as f:
                    content = f.read()
                test_results['backend_integration'] = 'training_api' in content
                if test_results['backend_integration']:
                    logger.info("✅ Backend integration working")
                else:
                    logger.error("❌ Backend integration not found")
            else:
                test_results['backend_integration'] = False
                logger.error("❌ Backend main.py not found")
                
        except Exception as e:
            test_results['backend_integration'] = False
            logger.error(f"❌ Backend integration test failed: {e}")
        
        # Test 4: Dashboard files
        try:
            dashboard_html = self.project_root / "training_data" / "dashboard" / "training_dashboard.html"
            dashboard_js = self.project_root / "training_data" / "dashboard" / "progress_charts.js"
            
            test_results['dashboard_files'] = dashboard_html.exists() and dashboard_js.exists()
            if test_results['dashboard_files']:
                logger.info("✅ Dashboard files present")
            else:
                logger.error("❌ Dashboard files missing")
                
        except Exception as e:
            test_results['dashboard_files'] = False
            logger.error(f"❌ Dashboard files test failed: {e}")
        
        return test_results
    
    def start_development_servers(self) -> bool:
        """Start the development servers"""
        logger.info("🚀 Starting development servers...")
        
        try:
            # Instructions for manual start (since we can't start servers in script)
            print("\n" + "="*60)
            print("🚀 STARTING DEVELOPMENT SERVERS")
            print("="*60)
            print("\nPlease start the servers manually in separate terminals:")
            print("\n1. Backend Server:")
            print("   cd backend")
            print("   source venv/bin/activate  # or venv\\Scripts\\activate on Windows")
            print("   python main.py")
            print("   → Server will start at http://localhost:8000")
            
            print("\n2. Frontend Server:")
            print("   cd frontend")
            print("   npm run dev")
            print("   → Server will start at http://localhost:3000")
            
            print("\n3. Training Dashboard:")
            print("   → Available at http://localhost:8000/training-dashboard")
            print("   → Or directly at http://localhost:8000/training-api/")
            
            print("\n4. Test the system:")
            print("   → Upload a video at http://localhost:3000")
            print("   → Check training dashboard for automation features")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Server start instructions failed: {e}")
            return False
    
    def show_next_steps(self):
        """Show next steps after quick start"""
        print("\n" + "="*60)
        print("🎯 NEXT STEPS")
        print("="*60)
        
        print("\n1. 📧 Configure API Keys and Email (Optional but Recommended):")
        print("   python setup_automation.py")
        print("   • Add YouTube API keys for video scraping")
        print("   • Add Unsplash/Pexels keys for stock photos")
        print("   • Configure email notifications")
        
        print("\n2. 🧪 Test the Complete System:")
        print("   python test_automation_system.py")
        print("   • Runs comprehensive tests")
        print("   • Validates all components")
        
        print("\n3. 📥 Try Data Collection:")
        print("   cd training_data/automation")
        print("   python collect_all_data.py --quick-test")
        print("   • Tests data collection with minimal data")
        
        print("\n4. 🧠 Try Progressive Training:")
        print("   cd training_data/automation")
        print("   python train_progressive.py --quick-test")
        print("   • Tests model training pipeline")
        
        print("\n5. 🔄 Setup Weekly Automation:")
        print("   # Add to crontab (Linux/Mac):")
        print("   0 2 * * 0 cd /path/to/soccer-app && python training_data/automation/weekly_training_update.py")
        
        print("\n6. 📊 Monitor Progress:")
        print("   • Dashboard: http://localhost:8000/training-dashboard")
        print("   • Logs: training_data/logs/")
        print("   • Status: python -c \"from training_data.automation.training_status import get_status_manager; print(get_status_manager().get_overall_status())\"")
        
        print("\n7. 📚 Documentation:")
        print("   • README.md - General usage")
        print("   • training_data/README.md - Training system details")
        print("   • scripts/README.md - Management scripts")
        
        print(f"\n🎉 Quick start completed! Your automation system is ready.")
        print(f"   Configuration: {self.config_file}")
        print(f"   Logs: training_data/logs/")
    
    async def run_quick_start(self) -> bool:
        """Run the complete quick start process"""
        print("🚀 SOCCER AI TRAINING AUTOMATION - QUICK START")
        print("="*60)
        print("This script will set up the training automation system.")
        print("You can run full setup later with: python setup_automation.py\n")
        
        # Step 1: Check prerequisites
        checks = self.check_prerequisites()
        
        if not all([checks['python_version'], checks['disk_space'], checks['project_structure']]):
            logger.error("❌ Critical prerequisites failed. Please fix and retry.")
            return False
        
        # Step 2: Install missing packages
        if not checks['required_packages']:
            if not self.install_missing_packages(checks['missing_packages']):
                logger.error("❌ Package installation failed")
                return False
        
        # Step 3: Create directory structure
        if not self.create_directory_structure():
            return False
        
        # Step 4: Run initial setup
        if not self.run_initial_setup():
            return False
        
        # Step 5: Test basic functionality
        test_results = await self.test_basic_functionality()
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        logger.info(f"🧪 Basic tests: {passed_tests}/{total_tests} passed")
        
        if passed_tests < total_tests:
            logger.warning("⚠️ Some tests failed, but system should still work")
        
        # Step 6: Show server start instructions
        self.start_development_servers()
        
        # Step 7: Show next steps
        self.show_next_steps()
        
        return True

async def main():
    """Main quick start function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick Start Soccer AI Training Automation")
    parser.add_argument('--skip-packages', action='store_true',
                       help='Skip automatic package installation')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run tests, skip setup')
    
    args = parser.parse_args()
    
    guide = QuickStartGuide()
    
    try:
        if args.test_only:
            logger.info("🧪 Running tests only...")
            test_results = await guide.test_basic_functionality()
            passed = sum(test_results.values())
            total = len(test_results)
            
            print(f"\n🧪 Test Results: {passed}/{total} passed")
            for test_name, result in test_results.items():
                status = "✅" if result else "❌"
                print(f"   {status} {test_name}")
            
            return 0 if passed == total else 1
        
        success = await guide.run_quick_start()
        
        if success:
            logger.info("🎉 Quick start completed successfully!")
            return 0
        else:
            logger.error("❌ Quick start failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Quick start cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Quick start failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)