#!/usr/bin/env python3
"""
COMPREHENSIVE MEDIAPIPE STABILITY DEPLOYMENT
============================================

This script implements all MediaPipe fixes on the DigitalOcean server.
Run this as root on the server: python3 deploy_mediapipe_fixes.py

FIXES INCLUDED:
1. Environment diagnosis and hardening
2. Dependency version locking
3. Pose detector stability improvements
4. Resource pooling implementation
5. Fallback strategies
6. Monitoring and logging enhancements
"""

import os
import sys
import subprocess
import shutil
import logging
from pathlib import Path
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/mediapipe_deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MediaPipeDeployment:
    """Comprehensive MediaPipe stability deployment"""
    
    def __init__(self):
        self.server_path = "/root/soccerapp/backend"
        self.venv_path = "/root/soccerapp/venv"
        self.backup_path = "/root/soccerapp/backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        self.deployment_log = []
    
    def log_step(self, message: str, success: bool = True):
        """Log deployment step"""
        status = "✅" if success else "❌"
        log_message = f"{status} {message}"
        logger.info(log_message)
        self.deployment_log.append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "success": success
        })
        print(log_message)
    
    def pre_deployment_backup(self):
        """Create backup before deployment"""
        try:
            self.log_step("Creating pre-deployment backup...")
            os.makedirs(self.backup_path, exist_ok=True)
            
            # Backup critical files
            files_to_backup = [
                "video_processor.py",
                "requirements.txt",
                "main.py",
                "supabase_client.py"
            ]
            
            for file in files_to_backup:
                src = os.path.join(self.server_path, file)
                dst = os.path.join(self.backup_path, file)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    self.log_step(f"Backed up {file}")
            
            self.log_step(f"Backup created at {self.backup_path}")
            
        except Exception as e:
            self.log_step(f"Backup failed: {e}", False)
            raise
    
    def stop_services(self):
        """Stop running services"""
        try:
            self.log_step("Stopping services...")
            
            # Kill any running uvicorn processes
            subprocess.run(["pkill", "-f", "uvicorn"], timeout=10)
            self.log_step("Stopped uvicorn processes")
            
        except Exception as e:
            self.log_step(f"Service stop warning: {e}", False)
    
    def update_system_dependencies(self):
        """Update system-level dependencies"""
        try:
            self.log_step("Updating system dependencies...")
            
            # Update package list
            subprocess.run(["apt-get", "update"], check=True, timeout=300)
            
            # Install required system libraries
            required_libs = [
                "libglib2.0-0",
                "libgstreamer1.0-0", 
                "libgtk-3-0",
                "libgl1-mesa-glx",
                "libglib2.0-dev",
                "python3-dev",
                "build-essential"
            ]
            
            cmd = ["apt-get", "install", "-y"] + required_libs
            subprocess.run(cmd, check=True, timeout=600)
            
            self.log_step("System dependencies updated")
            
        except Exception as e:
            self.log_step(f"System dependency update failed: {e}", False)
            raise
    
    def setup_python_environment(self):
        """Setup Python virtual environment with stable packages"""
        try:
            self.log_step("Setting up Python environment...")
            
            # Activate virtual environment
            activate_script = os.path.join(self.venv_path, "bin", "activate")
            if not os.path.exists(activate_script):
                self.log_step("Virtual environment not found, creating new one...")
                subprocess.run([sys.executable, "-m", "venv", self.venv_path], check=True)
            
            # Python executable in venv
            python_exec = os.path.join(self.venv_path, "bin", "python3")
            pip_exec = os.path.join(self.venv_path, "bin", "pip")
            
            # Upgrade pip first
            subprocess.run([pip_exec, "install", "--upgrade", "pip"], check=True, timeout=180)
            
            # Clear any conflicting packages
            packages_to_remove = ["mediapipe", "opencv-python", "protobuf", "numpy"]
            for package in packages_to_remove:
                try:
                    subprocess.run([pip_exec, "uninstall", "-y", package], timeout=60)
                except:
                    pass  # Package might not be installed
            
            # Install stable requirements
            stable_requirements = self._get_stable_requirements()
            
            # Write requirements to temp file
            req_file = "/tmp/stable_requirements.txt"
            with open(req_file, 'w') as f:
                f.write(stable_requirements)
            
            # Install with specific flags for stability
            subprocess.run([
                pip_exec, "install", 
                "-r", req_file,
                "--no-cache-dir",
                "--force-reinstall"
            ], check=True, timeout=1800)  # 30 minutes timeout
            
            self.log_step("Python environment setup complete")
            
        except Exception as e:
            self.log_step(f"Python environment setup failed: {e}", False)
            raise
    
    def _get_stable_requirements(self) -> str:
        """Get stable requirements content"""
        return """mediapipe==0.10.7
opencv-python==4.8.1.78
numpy==1.24.3
protobuf==3.20.3
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
supabase==2.0.2
torch==2.1.0
torchvision==0.16.0
ultralytics==8.0.196
filterpy==1.4.5
scipy==1.11.4
requests==2.31.0
aiofiles==23.2.1
python-dotenv==1.0.0
structlog==23.2.0
psutil==5.9.6
Pillow==10.0.1
imageio==2.31.5
imageio-ffmpeg==0.4.9"""
    
    def deploy_video_processor_fixes(self):
        """Deploy the enhanced video processor with stability fixes"""
        try:
            self.log_step("Deploying video processor fixes...")
            
            video_processor_path = os.path.join(self.server_path, "video_processor.py")
            
            # Read current video processor
            with open(video_processor_path, 'r') as f:
                current_content = f.read()
            
            # Apply the stability fixes
            enhanced_content = self._get_enhanced_video_processor(current_content)
            
            # Write enhanced version
            with open(video_processor_path, 'w') as f:
                f.write(enhanced_content)
            
            self.log_step("Video processor fixes deployed")
            
        except Exception as e:
            self.log_step(f"Video processor deployment failed: {e}", False)
            raise
    
    def _get_enhanced_video_processor(self, current_content: str) -> str:
        """Generate enhanced video processor with stability fixes"""
        
        # Add imports at the top
        enhanced_imports = """
import threading
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any
"""
        
        # Add the PoseDetectorPool class
        pose_pool_class = """
class PoseDetectorPool:
    \"\"\"Singleton pool for MediaPipe pose detectors to prevent initialization conflicts\"\"\"
    
    _instance = None
    _lock = threading.Lock()
    _pose_detector = None
    _mp_pose = None
    _initialization_lock = threading.Lock()
    _last_used = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._initialize_mediapipe()
    
    def _initialize_mediapipe(self):
        \"\"\"Safe MediaPipe initialization with retry logic\"\"\"
        max_retries = 3
        for attempt in range(max_retries):
            try:
                import mediapipe as mp
                self._mp_pose = mp.solutions.pose
                
                self._pose_detector = self._mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self._last_used = time.time()
                logger.info(f"MediaPipe pose detector initialized successfully (attempt {attempt + 1})")
                return
                
            except Exception as e:
                logger.error(f"MediaPipe initialization attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to initialize MediaPipe after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)
    
    @contextmanager
    def get_pose_detector(self):
        \"\"\"Thread-safe context manager for pose detector access\"\"\"
        with self._initialization_lock:
            try:
                if (self._pose_detector is None or 
                    (self._last_used and time.time() - self._last_used > 300)):
                    logger.info("Reinitializing pose detector due to timeout or None state")
                    self._cleanup_detector()
                    self._initialize_mediapipe()
                
                self._last_used = time.time()
                yield self._pose_detector, self._mp_pose
                
            except Exception as e:
                logger.error(f"Error in pose detector context: {e}")
                try:
                    self._cleanup_detector()
                    self._initialize_mediapipe()
                    yield self._pose_detector, self._mp_pose
                except Exception as retry_error:
                    logger.error(f"Pose detector recovery failed: {retry_error}")
                    yield None, None
    
    def _cleanup_detector(self):
        \"\"\"Safe cleanup of pose detector\"\"\"
        if self._pose_detector is not None:
            try:
                self._pose_detector.close()
            except Exception as e:
                logger.warning(f"Error closing pose detector: {e}")
            finally:
                self._pose_detector = None

# Global pose detector pool instance
pose_pool = PoseDetectorPool()
"""
        
        # Build the enhanced content
        enhanced_content = current_content
        
        # Add imports after existing imports
        import_insertion_point = enhanced_content.find("import logging")
        if import_insertion_point != -1:
            enhanced_content = (enhanced_content[:import_insertion_point] + 
                               enhanced_imports + "\n" +
                               enhanced_content[import_insertion_point:])
        
        # Add pose pool class after imports
        class_insertion_point = enhanced_content.find("class VideoProcessor:")
        if class_insertion_point != -1:
            enhanced_content = (enhanced_content[:class_insertion_point] + 
                               pose_pool_class + "\n\n" +
                               enhanced_content[class_insertion_point:])
        
        # Replace the orientation detection method
        old_method_start = enhanced_content.find("def _detect_video_orientation(self, frame: np.ndarray) -> str:")
        if old_method_start != -1:
            old_method_end = enhanced_content.find("return \"normal\"", old_method_start) + len("return \"normal\"")
            
            new_method = """def _detect_video_orientation(self, frame: np.ndarray) -> str:
        \"\"\"Ultra-stable orientation detection with fallback strategies\"\"\"
        try:
            with pose_pool.get_pose_detector() as (pose_detector, mp_pose):
                if pose_detector is None or mp_pose is None:
                    logger.warning("Pose detector unavailable, using fallback")
                    return self._detect_orientation_fallback(frame)
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = pose_detector.process(rgb_frame)
                
                if (pose_results and hasattr(pose_results, 'pose_landmarks') and 
                    pose_results.pose_landmarks and pose_results.pose_landmarks.landmark):
                    
                    landmarks = pose_results.pose_landmarks.landmark
                    if len(landmarks) < 33:
                        return "normal"
                    
                    try:
                        nose_y = landmarks[0].y
                        left_ankle_y = landmarks[27].y
                        right_ankle_y = landmarks[28].y
                        avg_foot_y = (left_ankle_y + right_ankle_y) / 2
                        
                        if nose_y > avg_foot_y + 0.15:
                            return "upside_down"
                            
                        return "normal"
                    except (IndexError, AttributeError):
                        return "normal"
                else:
                    return self._detect_orientation_fallback(frame)
                    
        except Exception as e:
            logger.warning(f"Orientation detection failed: {e}")
            return self._detect_orientation_fallback(frame)
    
    def _detect_orientation_fallback(self, frame: np.ndarray) -> str:
        \"\"\"Pose-free orientation detection using image analysis\"\"\"
        try:
            height, width = frame.shape[:2]
            if height > width * 1.5:
                return "rotated_left"
            return "normal"
        except Exception:
            return "normal\""""
            
            enhanced_content = (enhanced_content[:old_method_start] + 
                               new_method + 
                               enhanced_content[old_method_end:])
        
        return enhanced_content
    
    def verify_deployment(self):
        """Verify the deployment is working"""
        try:
            self.log_step("Verifying deployment...")
            
            python_exec = os.path.join(self.venv_path, "bin", "python3")
            
            # Test MediaPipe import
            test_script = """
import mediapipe as mp
import cv2
import numpy as np

# Test pose detector initialization
pose = mp.solutions.pose.Pose()
print("✅ MediaPipe pose detector initialized")

# Test with dummy frame
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
rgb_frame = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB)
results = pose.process(rgb_frame)
print("✅ MediaPipe processing works")

pose.close()
print("✅ MediaPipe cleanup works")
print("✅ ALL TESTS PASSED")
"""
            
            with open("/tmp/test_mediapipe.py", "w") as f:
                f.write(test_script)
            
            result = subprocess.run([python_exec, "/tmp/test_mediapipe.py"], 
                                   capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.log_step("MediaPipe verification successful")
                self.log_step(f"Test output: {result.stdout}")
            else:
                self.log_step(f"MediaPipe verification failed: {result.stderr}", False)
                raise Exception(f"Verification failed: {result.stderr}")
            
        except Exception as e:
            self.log_step(f"Verification failed: {e}", False)
            raise
    
    def start_services(self):
        """Start services after deployment"""
        try:
            self.log_step("Starting services...")
            
            # Change to backend directory
            os.chdir(self.server_path)
            
            # Start uvicorn in background
            python_exec = os.path.join(self.venv_path, "bin", "python3")
            
            self.log_step("Services ready to start")
            self.log_step("To start server: cd /root/soccerapp/backend && source ../venv/bin/activate && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000")
            
        except Exception as e:
            self.log_step(f"Service start failed: {e}", False)
    
    def generate_deployment_report(self):
        """Generate deployment report"""
        report = f"""
🚀 MEDIAPIPE STABILITY DEPLOYMENT REPORT
========================================

Deployment Time: {datetime.now().isoformat()}
Server Path: {self.server_path}
Backup Path: {self.backup_path}

DEPLOYMENT STEPS:
{chr(10).join([f"  {log['message']} {'✅' if log['success'] else '❌'}" for log in self.deployment_log])}

NEXT STEPS:
1. Start the server: cd {self.server_path} && source {self.venv_path}/bin/activate && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
2. Test with a juggling video
3. Monitor logs for stability: tail -f /var/log/mediapipe_deployment.log
4. If issues persist, restore from backup: {self.backup_path}

MONITORING:
- Check pose detector errors: grep "MediaPipe" /var/log/mediapipe_deployment.log
- Monitor memory usage: watch -n 5 free -h
- Check processes: ps aux | grep uvicorn

"""
        
        report_file = "/root/soccerapp/deployment_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(report)
        self.log_step(f"Deployment report saved to {report_file}")
    
    def deploy(self):
        """Main deployment process"""
        try:
            print("🚀 Starting MediaPipe Stability Deployment...")
            
            self.pre_deployment_backup()
            self.stop_services()
            self.update_system_dependencies()
            self.setup_python_environment()
            self.deploy_video_processor_fixes()
            self.verify_deployment()
            self.start_services()
            self.generate_deployment_report()
            
            print("🎉 DEPLOYMENT SUCCESSFUL!")
            print("MediaPipe stability fixes have been deployed.")
            
        except Exception as e:
            self.log_step(f"DEPLOYMENT FAILED: {e}", False)
            print(f"❌ DEPLOYMENT FAILED: {e}")
            print(f"Backup available at: {self.backup_path}")
            raise

def main():
    """Main deployment function"""
    if os.geteuid() != 0:
        print("❌ This script must be run as root")
        sys.exit(1)
    
    deployment = MediaPipeDeployment()
    deployment.deploy()

if __name__ == "__main__":
    main()