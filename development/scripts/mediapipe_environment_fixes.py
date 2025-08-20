#!/usr/bin/env python3
"""
MediaPipe Environment Diagnostic and Fix Script
Run this on DigitalOcean server to diagnose and fix environment issues
"""

import subprocess
import sys
import os
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class MediaPipeEnvironmentFixer:
    """Comprehensive MediaPipe environment diagnostic and fixing"""
    
    def __init__(self):
        self.issues_found = []
        self.fixes_applied = []
    
    def diagnose_environment(self) -> Dict[str, any]:
        """Comprehensive environment diagnosis"""
        print("🔍 Diagnosing MediaPipe Environment...")
        
        results = {
            "python_version": self._check_python_version(),
            "dependencies": self._check_dependencies(),
            "system_libs": self._check_system_libraries(),
            "memory": self._check_memory(),
            "gpu": self._check_gpu_support(),
            "permissions": self._check_permissions(),
            "conflicts": self._check_version_conflicts()
        }
        
        return results
    
    def _check_python_version(self) -> Dict[str, str]:
        """Check Python version compatibility"""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        # MediaPipe requires Python 3.7-3.11
        compatible = 3.7 <= version.major + version.minor/10 <= 3.11
        
        result = {
            "version": version_str,
            "compatible": compatible,
            "recommendation": "OK" if compatible else "Update to Python 3.8-3.11"
        }
        
        if not compatible:
            self.issues_found.append(f"Python {version_str} not compatible with MediaPipe")
            
        return result
    
    def _check_dependencies(self) -> Dict[str, any]:
        """Check critical dependency versions"""
        dependencies = {
            "mediapipe": None,
            "opencv-python": None,
            "numpy": None,
            "protobuf": None,
            "attrs": None
        }
        
        for package in dependencies.keys():
            try:
                result = subprocess.run([sys.executable, "-c", f"import {package.replace('-', '_')}; print({package.replace('-', '_')}.__version__)"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    dependencies[package] = result.stdout.strip()
                else:
                    dependencies[package] = "NOT_INSTALLED"
                    self.issues_found.append(f"{package} not installed")
            except Exception as e:
                dependencies[package] = f"ERROR: {e}"
                self.issues_found.append(f"Error checking {package}: {e}")
        
        # Check for known problematic combinations
        self._check_dependency_conflicts(dependencies)
        
        return dependencies
    
    def _check_dependency_conflicts(self, deps: Dict[str, str]):
        """Check for known problematic dependency combinations"""
        
        # Protobuf version conflicts (very common with MediaPipe)
        protobuf_version = deps.get("protobuf", "")
        if protobuf_version and protobuf_version != "NOT_INSTALLED":
            try:
                major_version = int(protobuf_version.split('.')[0])
                if major_version >= 4:
                    self.issues_found.append("Protobuf 4.x can cause MediaPipe issues, recommend 3.20.x")
            except ValueError:
                pass
        
        # OpenCV conflicts
        opencv_version = deps.get("opencv-python", "")
        if opencv_version and "4.5" in opencv_version:
            self.issues_found.append("OpenCV 4.5.x has known MediaPipe compatibility issues")
    
    def _check_system_libraries(self) -> Dict[str, bool]:
        """Check required system libraries"""
        required_libs = [
            "libglib2.0-0",
            "libgstreamer1.0-0",
            "libgtk-3-0",
            "libgl1-mesa-glx",
            "libglib2.0-dev"
        ]
        
        lib_status = {}
        for lib in required_libs:
            try:
                result = subprocess.run(["dpkg", "-l", lib], 
                                      capture_output=True, text=True, timeout=5)
                lib_status[lib] = result.returncode == 0
                if result.returncode != 0:
                    self.issues_found.append(f"Missing system library: {lib}")
            except Exception:
                lib_status[lib] = False
                self.issues_found.append(f"Cannot check system library: {lib}")
        
        return lib_status
    
    def _check_memory(self) -> Dict[str, any]:
        """Check available memory"""
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            
            total_mem = None
            available_mem = None
            
            for line in meminfo.split('\n'):
                if line.startswith('MemTotal:'):
                    total_mem = int(line.split()[1]) * 1024  # Convert KB to bytes
                elif line.startswith('MemAvailable:'):
                    available_mem = int(line.split()[1]) * 1024
            
            # MediaPipe typically needs 500MB+ per pose detector
            sufficient = available_mem and available_mem > 1024 * 1024 * 1024  # 1GB
            
            if not sufficient:
                self.issues_found.append("Insufficient memory for stable MediaPipe operation")
            
            return {
                "total_gb": round(total_mem / (1024**3), 2) if total_mem else None,
                "available_gb": round(available_mem / (1024**3), 2) if available_mem else None,
                "sufficient": sufficient
            }
            
        except Exception as e:
            self.issues_found.append(f"Cannot check memory: {e}")
            return {"error": str(e)}
    
    def _check_gpu_support(self) -> Dict[str, any]:
        """Check GPU support (optional but can cause issues)"""
        try:
            # Check if CUDA is available
            cuda_result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
            has_cuda = cuda_result.returncode == 0
            
            # Check if MediaPipe is trying to use GPU
            gpu_status = {
                "cuda_available": has_cuda,
                "recommendation": "Force CPU mode for stability" if has_cuda else "CPU mode (good for stability)"
            }
            
            if has_cuda:
                self.issues_found.append("GPU detected - consider forcing CPU mode for MediaPipe stability")
                
            return gpu_status
            
        except Exception:
            return {"cuda_available": False, "recommendation": "CPU mode (good for stability)"}
    
    def _check_permissions(self) -> Dict[str, bool]:
        """Check file and directory permissions"""
        paths_to_check = [
            "/tmp",
            os.path.expanduser("~/.cache"),
            "/dev/shm"
        ]
        
        permissions = {}
        for path in paths_to_check:
            try:
                test_file = os.path.join(path, "mediapipe_test")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                permissions[path] = True
            except Exception:
                permissions[path] = False
                self.issues_found.append(f"No write permission to {path}")
        
        return permissions
    
    def _check_version_conflicts(self) -> List[str]:
        """Check for known version conflicts"""
        conflicts = []
        
        try:
            # Test MediaPipe import
            result = subprocess.run([sys.executable, "-c", 
                                   "import mediapipe as mp; pose = mp.solutions.pose.Pose(); pose.close(); print('OK')"], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                conflicts.append(f"MediaPipe import/initialization failed: {result.stderr}")
                self.issues_found.append("MediaPipe cannot initialize properly")
            
        except subprocess.TimeoutExpired:
            conflicts.append("MediaPipe initialization timeout (>30s)")
            self.issues_found.append("MediaPipe initialization is too slow")
        except Exception as e:
            conflicts.append(f"Cannot test MediaPipe: {e}")
        
        return conflicts
    
    def apply_fixes(self) -> List[str]:
        """Apply automatic fixes for detected issues"""
        print("🔧 Applying fixes...")
        
        fixes = []
        
        # Fix 1: Install missing system libraries
        missing_libs = []
        system_libs = self._check_system_libraries()
        for lib, installed in system_libs.items():
            if not installed:
                missing_libs.append(lib)
        
        if missing_libs:
            try:
                cmd = ["sudo", "apt-get", "update", "&&", "sudo", "apt-get", "install", "-y"] + missing_libs
                subprocess.run(cmd, check=True, timeout=300)
                fixes.append(f"Installed system libraries: {', '.join(missing_libs)}")
            except Exception as e:
                fixes.append(f"Failed to install system libraries: {e}")
        
        # Fix 2: Pin dependency versions
        try:
            known_stable_versions = [
                "mediapipe==0.10.7",
                "opencv-python==4.8.1.78", 
                "protobuf==3.20.3",
                "numpy==1.24.3"
            ]
            
            for package in known_stable_versions:
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                 check=True, timeout=180)
                    fixes.append(f"Installed/pinned {package}")
                except Exception as e:
                    fixes.append(f"Failed to install {package}: {e}")
                    
        except Exception as e:
            fixes.append(f"Error during dependency fixing: {e}")
        
        # Fix 3: Clear caches
        try:
            cache_dirs = [
                os.path.expanduser("~/.cache/pip"),
                "/tmp/mediapipe_*",
                "/var/tmp/mediapipe_*"
            ]
            
            for cache_dir in cache_dirs:
                subprocess.run(["rm", "-rf", cache_dir], timeout=30)
            fixes.append("Cleared MediaPipe and pip caches")
        except Exception as e:
            fixes.append(f"Cache clearing failed: {e}")
        
        self.fixes_applied = fixes
        return fixes
    
    def generate_report(self) -> str:
        """Generate comprehensive diagnostic report"""
        results = self.diagnose_environment()
        
        report = """
🔍 MEDIAPIPE ENVIRONMENT DIAGNOSTIC REPORT
============================================

PYTHON VERSION:
{python_version}

DEPENDENCIES:
{dependencies}

SYSTEM LIBRARIES:
{system_libs}

MEMORY STATUS:
{memory}

GPU STATUS:
{gpu}

PERMISSIONS:
{permissions}

VERSION CONFLICTS:
{conflicts}

ISSUES FOUND:
{issues}

RECOMMENDED FIXES:
{recommendations}
""".format(
            python_version=self._format_dict(results["python_version"]),
            dependencies=self._format_dict(results["dependencies"]),
            system_libs=self._format_dict(results["system_libs"]),
            memory=self._format_dict(results["memory"]),
            gpu=self._format_dict(results["gpu"]),
            permissions=self._format_dict(results["permissions"]),
            conflicts='\n'.join(results["conflicts"]) if results["conflicts"] else "None detected",
            issues='\n'.join([f"❌ {issue}" for issue in self.issues_found]) if self.issues_found else "✅ No major issues detected",
            recommendations=self._generate_recommendations()
        )
        
        return report
    
    def _format_dict(self, d: Dict) -> str:
        """Format dictionary for report"""
        return '\n'.join([f"  {k}: {v}" for k, v in d.items()])
    
    def _generate_recommendations(self) -> str:
        """Generate specific recommendations based on findings"""
        recommendations = []
        
        if "protobuf" in str(self.issues_found):
            recommendations.append("🔧 pip install protobuf==3.20.3")
        
        if "memory" in str(self.issues_found):
            recommendations.append("🔧 Increase server memory or implement resource pooling")
        
        if "MediaPipe import" in str(self.issues_found):
            recommendations.append("🔧 Reinstall MediaPipe with: pip uninstall mediapipe && pip install mediapipe==0.10.7")
        
        if "GPU" in str(self.issues_found):
            recommendations.append("🔧 Force CPU mode by setting CUDA_VISIBLE_DEVICES=''")
        
        if not recommendations:
            recommendations.append("✅ Environment looks good - focus on application-level fixes")
        
        return '\n'.join(recommendations)

def main():
    """Main diagnostic and fix routine"""
    fixer = MediaPipeEnvironmentFixer()
    
    print("Starting MediaPipe Environment Analysis...")
    report = fixer.generate_report()
    print(report)
    
    if fixer.issues_found:
        response = input("\nApply automatic fixes? (y/n): ")
        if response.lower() == 'y':
            fixes = fixer.apply_fixes()
            print("\nFixes Applied:")
            for fix in fixes:
                print(f"✅ {fix}")
        else:
            print("Manual intervention required. See recommendations above.")
    else:
        print("✅ Environment looks stable!")

if __name__ == "__main__":
    main()