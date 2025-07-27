#!/usr/bin/env python3
"""
Setup Automation System
One-time configuration for comprehensive training automation
"""

import os
import json
import sys
import sqlite3
import smtplib
from pathlib import Path
from typing import Dict, Any, Optional
from email.mime.text import MimeText
import requests

class AutomationSetup:
    def __init__(self):
        self.config_file = Path("automation_config.json")
        self.config = {}
        self.setup_status = {}
        
    def load_existing_config(self) -> Dict[str, Any]:
        """Load existing configuration if available"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Could not load existing config: {e}")
        return {}
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"✅ Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"❌ Failed to save config: {e}")
    
    def setup_api_keys(self):
        """Configure API keys for data sources"""
        print("\n🔑 API KEYS CONFIGURATION")
        print("=" * 50)
        
        api_keys = {}
        
        # YouTube API Keys (multiple for parallel processing)
        print("\n📺 YouTube Data API (for video scraping)")
        print("Get keys from: https://console.cloud.google.com/apis/credentials")
        
        youtube_keys = []
        while len(youtube_keys) < 3:
            key = input(f"YouTube API Key #{len(youtube_keys) + 1} (or 'skip' to continue): ").strip()
            if key.lower() == 'skip' and len(youtube_keys) > 0:
                break
            elif key.lower() == 'skip':
                print("⚠️ At least one YouTube API key is recommended")
                continue
            elif key:
                # Test the key
                if self.test_youtube_api(key):
                    youtube_keys.append(key)
                    print(f"✅ YouTube API Key #{len(youtube_keys)} verified")
                else:
                    print("❌ Invalid YouTube API key, please try again")
        
        api_keys['youtube'] = youtube_keys
        
        # Instagram Basic Display API
        print("\n📸 Instagram Basic Display API (for hashtag scraping)")
        print("Get from: https://developers.facebook.com/docs/instagram-basic-display-api")
        
        instagram_token = input("Instagram Access Token (or 'skip'): ").strip()
        if instagram_token.lower() != 'skip' and instagram_token:
            if self.test_instagram_api(instagram_token):
                api_keys['instagram'] = instagram_token
                print("✅ Instagram API verified")
            else:
                print("❌ Instagram API key invalid")
        
        # Stock Photo APIs
        print("\n🖼️ Stock Photo APIs")
        
        # Unsplash
        unsplash_key = input("Unsplash Access Key (or 'skip'): ").strip()
        if unsplash_key.lower() != 'skip' and unsplash_key:
            if self.test_unsplash_api(unsplash_key):
                api_keys['unsplash'] = unsplash_key
                print("✅ Unsplash API verified")
            else:
                print("❌ Unsplash API key invalid")
        
        # Pexels
        pexels_key = input("Pexels API Key (or 'skip'): ").strip()
        if pexels_key.lower() != 'skip' and pexels_key:
            if self.test_pexels_api(pexels_key):
                api_keys['pexels'] = pexels_key
                print("✅ Pexels API verified")
            else:
                print("❌ Pexels API key invalid")
        
        self.config['api_keys'] = api_keys
        self.setup_status['api_keys'] = True
    
    def setup_email_notifications(self):
        """Configure email notifications"""
        print("\n📧 EMAIL NOTIFICATIONS SETUP")
        print("=" * 50)
        
        use_email = input("Enable email notifications? (y/n): ").strip().lower()
        if use_email != 'y':
            self.config['email'] = {'enabled': False}
            return
        
        email_config = {'enabled': True}
        
        print("\nEmail Configuration:")
        email_config['smtp_server'] = input("SMTP Server (e.g., smtp.gmail.com): ").strip()
        email_config['smtp_port'] = int(input("SMTP Port (587 for TLS, 465 for SSL): ").strip() or "587")
        email_config['username'] = input("Email Username: ").strip()
        email_config['password'] = input("Email Password/App Password: ").strip()
        email_config['from_email'] = input("From Email Address: ").strip()
        email_config['to_emails'] = input("To Email Addresses (comma-separated): ").strip().split(',')
        email_config['to_emails'] = [email.strip() for email in email_config['to_emails']]
        
        # Test email configuration
        if self.test_email_config(email_config):
            self.config['email'] = email_config
            print("✅ Email configuration verified")
        else:
            print("❌ Email configuration failed, disabling notifications")
            self.config['email'] = {'enabled': False}
        
        self.setup_status['email'] = True
    
    def setup_training_parameters(self):
        """Configure training parameters and thresholds"""
        print("\n🎯 TRAINING PARAMETERS SETUP")
        print("=" * 50)
        
        training_config = {}
        
        print("\nData Collection Targets:")
        training_config['target_images_initial'] = int(input("Initial data collection target (default 5000): ").strip() or "5000")
        training_config['target_images_weekly'] = int(input("Weekly data collection target (default 1000): ").strip() or "1000")
        training_config['min_new_data_for_training'] = int(input("Min new images to trigger training (default 500): ").strip() or "500")
        
        print("\nModel Performance Thresholds:")
        training_config['min_accuracy_improvement'] = float(input("Min accuracy improvement for deployment (% default 5): ").strip() or "5")
        training_config['target_accuracy'] = float(input("Target model accuracy (% default 90): ").strip() or "90")
        training_config['min_confidence_score'] = float(input("Min confidence score threshold (default 0.7): ").strip() or "0.7")
        
        print("\nTraining Schedule:")
        training_config['max_training_hours'] = int(input("Max training time per session (hours, default 4): ").strip() or "4")
        training_config['training_stages'] = [
            int(x.strip()) for x in 
            input("Training stages - image counts (default: 100,500,1000,5000): ").strip().split(',')
            if x.strip()
        ] or [100, 500, 1000, 5000]
        
        print("\nResource Management:")
        training_config['max_parallel_downloads'] = int(input("Max parallel downloads (default 5): ").strip() or "5")
        training_config['max_disk_usage_gb'] = int(input("Max disk usage for training data (GB, default 50): ").strip() or "50")
        training_config['cleanup_after_days'] = int(input("Clean up old data after days (default 30): ").strip() or "30")
        
        self.config['training'] = training_config
        self.setup_status['training_params'] = True
    
    def setup_cloud_storage(self):
        """Configure optional cloud storage"""
        print("\n☁️ CLOUD STORAGE SETUP (Optional)")
        print("=" * 50)
        
        use_cloud = input("Enable cloud storage backup? (y/n): ").strip().lower()
        if use_cloud != 'y':
            self.config['cloud_storage'] = {'enabled': False}
            return
        
        cloud_config = {'enabled': True}
        
        provider = input("Cloud provider (aws/gcp/azure): ").strip().lower()
        
        if provider == 'aws':
            cloud_config['provider'] = 'aws'
            cloud_config['aws_access_key'] = input("AWS Access Key ID: ").strip()
            cloud_config['aws_secret_key'] = input("AWS Secret Access Key: ").strip()
            cloud_config['aws_bucket'] = input("S3 Bucket Name: ").strip()
            cloud_config['aws_region'] = input("AWS Region (default us-east-1): ").strip() or "us-east-1"
        else:
            print("❌ Only AWS S3 supported currently")
            cloud_config['enabled'] = False
        
        self.config['cloud_storage'] = cloud_config
        self.setup_status['cloud_storage'] = True
    
    def setup_directories(self):
        """Create necessary directory structure"""
        print("\n📁 DIRECTORY STRUCTURE SETUP")
        print("=" * 50)
        
        directories = [
            "training_data/automation",
            "training_data/dashboard", 
            "training_data/scrapers",
            "training_data/collected_data",
            "training_data/collected_data/youtube",
            "training_data/collected_data/instagram", 
            "training_data/collected_data/stock_photos",
            "training_data/collected_data/datasets",
            "training_data/models/backups",
            "training_data/models/staging",
            "training_data/models/production",
            "training_data/logs",
            "training_data/notifications"
        ]
        
        created_dirs = []
        for directory in directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(directory)
        
        if created_dirs:
            print(f"✅ Created {len(created_dirs)} directories")
        else:
            print("✅ All directories already exist")
        
        self.setup_status['directories'] = True
    
    def setup_database_extensions(self):
        """Add training automation tables to existing database"""
        print("\n🗄️ DATABASE EXTENSIONS SETUP")
        print("=" * 50)
        
        db_path = Path("backend/database.db")
        if not db_path.exists():
            print("❌ Main database not found. Please run the main app first.")
            return False
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Training sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id TEXT PRIMARY KEY,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT,
                    model_version TEXT,
                    dataset_size INTEGER,
                    accuracy_before REAL,
                    accuracy_after REAL,
                    improvement_percent REAL,
                    config TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Data collection sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_collection_sessions (
                    id TEXT PRIMARY KEY,
                    source TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT,
                    images_collected INTEGER,
                    videos_collected INTEGER,
                    total_size_mb REAL,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Model deployments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_deployments (
                    id TEXT PRIMARY KEY,
                    model_version TEXT,
                    deployment_time TIMESTAMP,
                    previous_model TEXT,
                    status TEXT,
                    rollback_time TIMESTAMP,
                    performance_metrics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # System health logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_health_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    component TEXT,
                    status TEXT,
                    metrics TEXT,
                    alerts TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            
            print("✅ Database extensions created successfully")
            self.setup_status['database'] = True
            return True
            
        except Exception as e:
            print(f"❌ Database setup failed: {e}")
            return False
    
    def test_youtube_api(self, api_key: str) -> bool:
        """Test YouTube API key"""
        try:
            url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q=soccer&type=video&maxResults=1&key={api_key}"
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def test_instagram_api(self, access_token: str) -> bool:
        """Test Instagram API token"""
        try:
            url = f"https://graph.instagram.com/me?fields=id,username&access_token={access_token}"
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def test_unsplash_api(self, access_key: str) -> bool:
        """Test Unsplash API key"""
        try:
            headers = {"Authorization": f"Client-ID {access_key}"}
            url = "https://api.unsplash.com/search/photos?query=soccer&per_page=1"
            response = requests.get(url, headers=headers, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def test_pexels_api(self, api_key: str) -> bool:
        """Test Pexels API key"""
        try:
            headers = {"Authorization": api_key}
            url = "https://api.pexels.com/v1/search?query=soccer&per_page=1"
            response = requests.get(url, headers=headers, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def test_email_config(self, email_config: Dict[str, Any]) -> bool:
        """Test email configuration"""
        try:
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            
            # Send test email
            msg = MimeText("Soccer App automation setup test email")
            msg['Subject'] = "Soccer App - Setup Test"
            msg['From'] = email_config['from_email']
            msg['To'] = email_config['to_emails'][0]
            
            server.send_message(msg)
            server.quit()
            
            print("📧 Test email sent successfully")
            return True
        except Exception as e:
            print(f"❌ Email test failed: {e}")
            return False
    
    def validate_setup(self) -> bool:
        """Validate the complete setup"""
        print("\n✅ SETUP VALIDATION")
        print("=" * 50)
        
        validation_results = {}
        
        # Check API keys
        api_keys = self.config.get('api_keys', {})
        validation_results['youtube_api'] = len(api_keys.get('youtube', [])) > 0
        validation_results['data_sources'] = len([k for k in api_keys.keys() if api_keys[k]]) > 0
        
        # Check email
        validation_results['email'] = self.config.get('email', {}).get('enabled', False)
        
        # Check training params
        validation_results['training_params'] = 'training' in self.config
        
        # Check directories
        validation_results['directories'] = self.setup_status.get('directories', False)
        
        # Check database
        validation_results['database'] = self.setup_status.get('database', False)
        
        # Print validation results
        for component, status in validation_results.items():
            status_icon = "✅" if status else "⚠️"
            print(f"{status_icon} {component.replace('_', ' ').title()}: {'OK' if status else 'Not configured'}")
        
        # Overall status
        required_components = ['data_sources', 'training_params', 'directories', 'database']
        all_required = all(validation_results[comp] for comp in required_components)
        
        if all_required:
            print("\n🎉 Setup completed successfully!")
            print("You can now run: python training_data/automation/collect_all_data.py")
            return True
        else:
            print("\n⚠️ Setup incomplete. Some required components are missing.")
            return False
    
    def run_setup(self):
        """Run the complete setup process"""
        print("🚀 SOCCER APP TRAINING AUTOMATION SETUP")
        print("=" * 60)
        print("This will configure your system for automated training and monitoring.")
        print("You can re-run this script to update configuration.\n")
        
        # Load existing config
        self.config = self.load_existing_config()
        
        try:
            # Setup steps
            self.setup_directories()
            self.setup_database_extensions()
            self.setup_api_keys()
            self.setup_training_parameters()
            self.setup_email_notifications()
            self.setup_cloud_storage()
            
            # Save configuration
            self.save_config()
            
            # Validate setup
            success = self.validate_setup()
            
            if success:
                print("\n🎯 NEXT STEPS:")
                print("1. Run initial data collection: python training_data/automation/collect_all_data.py --target-images 5000")
                print("2. Start progressive training: python training_data/automation/train_progressive.py")
                print("3. Access training dashboard: http://localhost:8000/training-dashboard")
                print("4. Set up weekly automation: python training_data/automation/weekly_training_update.py")
            
            return success
            
        except KeyboardInterrupt:
            print("\n\n⏹️ Setup cancelled by user")
            return False
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            return False

def main():
    setup = AutomationSetup()
    success = setup.run_setup()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()