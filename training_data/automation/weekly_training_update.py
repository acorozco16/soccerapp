#!/usr/bin/env python3
"""
Weekly Training Update System
Automated maintenance, data collection, training, and deployment
"""

import os
import json
import asyncio
import time
import smtplib
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# Add required paths
import sys
sys.path.append(str(Path(__file__).parent))

from collect_all_data import ComprehensiveDataCollector
from train_progressive import ProgressiveTrainer
from deploy_improved_model import ModelDeploymentManager
from training_status import get_status_manager, update_collection_status, Status

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_data/logs/weekly_updates.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WeeklyTrainingUpdater:
    def __init__(self, config_file: str = "automation_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.status_manager = get_status_manager()
        
        # Training configuration
        self.training_config = self.config.get('training', {})
        self.email_config = self.config.get('email', {})
        
        # Update schedule settings
        self.min_new_data_for_training = self.training_config.get('min_new_data_for_training', 500)
        self.target_weekly_images = self.training_config.get('target_images_weekly', 1000)
        self.min_improvement_threshold = self.training_config.get('min_accuracy_improvement', 5.0)
        
        # Tracking
        self.update_session_id = f"weekly_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.update_results = {
            'session_id': self.update_session_id,
            'started_at': None,
            'completed_at': None,
            'phases_completed': [],
            'phases_failed': [],
            'data_collection_results': None,
            'training_results': None,
            'deployment_results': None,
            'notifications_sent': [],
            'errors': []
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.config_file.exists():
            raise Exception(f"Configuration file not found: {self.config_file}")
        
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load configuration: {e}")
    
    def _analyze_training_data_freshness(self) -> Dict[str, Any]:
        """Analyze how much new training data is available"""
        try:
            collected_data_dir = Path("training_data/collected_data")
            
            # Get current data statistics
            current_stats = {
                'total_files': 0,
                'new_files_this_week': 0,
                'by_source': {},
                'oldest_file': None,
                'newest_file': None
            }
            
            if not collected_data_dir.exists():
                return current_stats
            
            week_ago = datetime.now() - timedelta(days=7)
            
            # Analyze each source
            for source_dir in collected_data_dir.iterdir():
                if not source_dir.is_dir():
                    continue
                
                source_stats = {
                    'total_files': 0,
                    'new_files': 0,
                    'newest_file': None
                }
                
                for file_path in source_dir.rglob("*"):
                    if file_path.is_file() and not file_path.name.endswith('.json'):
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        
                        source_stats['total_files'] += 1
                        current_stats['total_files'] += 1
                        
                        if file_time > week_ago:
                            source_stats['new_files'] += 1
                            current_stats['new_files_this_week'] += 1
                        
                        # Track newest file
                        if (source_stats['newest_file'] is None or 
                            file_time > source_stats['newest_file']):
                            source_stats['newest_file'] = file_time
                        
                        # Track overall newest/oldest
                        if (current_stats['newest_file'] is None or 
                            file_time > current_stats['newest_file']):
                            current_stats['newest_file'] = file_time
                        
                        if (current_stats['oldest_file'] is None or 
                            file_time < current_stats['oldest_file']):
                            current_stats['oldest_file'] = file_time
                
                current_stats['by_source'][source_dir.name] = source_stats
            
            # Determine if training is needed
            current_stats['training_recommended'] = (
                current_stats['new_files_this_week'] >= self.min_new_data_for_training
            )
            
            current_stats['data_collection_needed'] = (
                current_stats['new_files_this_week'] < self.target_weekly_images
            )
            
            return current_stats
            
        except Exception as e:
            logger.error(f"Error analyzing training data freshness: {e}")
            return {'error': str(e)}
    
    def _get_last_training_session(self) -> Optional[Dict[str, Any]]:
        """Get information about the last training session"""
        try:
            # Check database for last training session
            db_path = Path("backend/database.db")
            if db_path.exists():
                import sqlite3
                
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, start_time, end_time, status, accuracy_after, improvement_percent
                    FROM training_sessions
                    ORDER BY start_time DESC
                    LIMIT 1
                """)
                
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    return {
                        'session_id': result[0],
                        'start_time': result[1],
                        'end_time': result[2],
                        'status': result[3],
                        'final_accuracy': result[4] or 0,
                        'improvement': result[5] or 0,
                        'days_ago': (datetime.now() - datetime.fromisoformat(result[1])).days
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting last training session: {e}")
            return None
    
    def _get_current_model_performance(self) -> Dict[str, Any]:
        """Get current production model performance"""
        try:
            # Check for current production model
            production_dir = Path("training_data/models/production")
            if not production_dir.exists():
                return {'has_model': False}
            
            model_files = list(production_dir.glob("*.pt"))
            if not model_files:
                return {'has_model': False}
            
            # Get most recent model
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            
            # Load metadata
            metadata_file = latest_model.with_suffix('.json')
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            return {
                'has_model': True,
                'model_path': str(latest_model),
                'deployed_at': metadata.get('deployed_at'),
                'accuracy': metadata.get('accuracy', 0),
                'deployment_type': metadata.get('deployment_type', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error getting current model performance: {e}")
            return {'has_model': False, 'error': str(e)}
    
    async def _run_weekly_data_collection(self) -> Dict[str, Any]:
        """Run weekly data collection if needed"""
        logger.info("🔍 Analyzing data collection needs...")
        
        data_analysis = self._analyze_training_data_freshness()
        
        if data_analysis.get('error'):
            raise Exception(f"Data analysis failed: {data_analysis['error']}")
        
        if not data_analysis.get('data_collection_needed', False):
            logger.info(f"✅ Sufficient new data available: {data_analysis['new_files_this_week']} files this week")
            return {
                'skipped': True,
                'reason': 'Sufficient data available',
                'current_stats': data_analysis
            }
        
        logger.info(f"📥 Starting weekly data collection (target: {self.target_weekly_images} images)...")
        
        collector = ComprehensiveDataCollector()
        collection_result = await collector.collect_all_data(
            target_images=self.target_weekly_images,
            enable_sources=['youtube', 'stock_photos', 'instagram']  # Skip large datasets for weekly updates
        )
        
        return {
            'executed': True,
            'collection_result': collection_result,
            'previous_stats': data_analysis
        }
    
    async def _run_conditional_training(self) -> Dict[str, Any]:
        """Run training if sufficient new data is available"""
        logger.info("🧠 Analyzing training needs...")
        
        data_analysis = self._analyze_training_data_freshness()
        last_training = self._get_last_training_session()
        
        # Check if training is needed
        reasons_to_skip = []
        
        if not data_analysis.get('training_recommended', False):
            reasons_to_skip.append(f"Insufficient new data: {data_analysis.get('new_files_this_week', 0)} < {self.min_new_data_for_training}")
        
        if last_training and last_training['days_ago'] < 3:
            reasons_to_skip.append(f"Recent training session: {last_training['days_ago']} days ago")
        
        if reasons_to_skip:
            logger.info(f"⏸️ Skipping training: {'; '.join(reasons_to_skip)}")
            return {
                'skipped': True,
                'reasons': reasons_to_skip,
                'data_analysis': data_analysis,
                'last_training': last_training
            }
        
        logger.info("🚀 Starting progressive training...")
        
        trainer = ProgressiveTrainer()
        training_result = await trainer.run_progressive_training(
            auto_deploy=False,  # Don't auto-deploy from weekly updates
            min_improvement=self.min_improvement_threshold
        )
        
        return {
            'executed': True,
            'training_result': training_result,
            'data_analysis': data_analysis,
            'last_training': last_training
        }
    
    async def _run_conditional_deployment(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy model if training produced good results"""
        if not training_results.get('executed', False):
            logger.info("⏸️ Skipping deployment: No training was performed")
            return {
                'skipped': True,
                'reason': 'No training performed'
            }
        
        training_result = training_results.get('training_result', {})
        best_model = training_result.get('best_model')
        
        if not best_model:
            logger.info("⏸️ Skipping deployment: No successful model from training")
            return {
                'skipped': True,
                'reason': 'No successful model produced'
            }
        
        # Check if model meets deployment criteria
        improvement = best_model.get('improvement_percent', 0)
        accuracy = best_model.get('final_accuracy', 0)
        
        if improvement < self.min_improvement_threshold:
            logger.info(f"⏸️ Skipping deployment: Improvement {improvement:.2f}% < {self.min_improvement_threshold}%")
            return {
                'skipped': True,
                'reason': f'Insufficient improvement: {improvement:.2f}%',
                'model_info': best_model
            }
        
        logger.info(f"🚀 Deploying model with {improvement:.2f}% improvement...")
        
        deployer = ModelDeploymentManager()
        deployment_result = await deployer.deploy_model(
            model_path=best_model['model_path'],
            test_first=True,
            rollback_on_failure=True
        )
        
        return {
            'executed': True,
            'deployment_result': deployment_result,
            'model_info': best_model
        }
    
    def _generate_update_report(self) -> str:
        """Generate comprehensive update report"""
        results = self.update_results
        
        # Calculate duration
        if results['started_at'] and results['completed_at']:
            start_time = datetime.fromisoformat(results['started_at'])
            end_time = datetime.fromisoformat(results['completed_at'])
            duration = (end_time - start_time).total_seconds() / 60  # minutes
        else:
            duration = 0
        
        report = f"""
🤖 SOCCER AI WEEKLY TRAINING UPDATE REPORT
{'='*60}

📋 Session: {results['session_id']}
📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
⏱️ Duration: {duration:.1f} minutes
✅ Phases Completed: {len(results['phases_completed'])}
❌ Phases Failed: {len(results['phases_failed'])}

"""
        
        # Data Collection Summary
        data_results = results.get('data_collection_results')
        if data_results:
            if data_results.get('skipped'):
                report += f"""
📥 DATA COLLECTION: SKIPPED
   Reason: {data_results.get('reason', 'Unknown')}
   Current files this week: {data_results.get('current_stats', {}).get('new_files_this_week', 0)}
"""
            else:
                collection_data = data_results.get('collection_result', {})
                report += f"""
📥 DATA COLLECTION: COMPLETED
   Total items collected: {collection_data.get('total_items_collected', 0)}
   Sources completed: {len(collection_data.get('sources_completed', []))}
   Success rate: {(len(collection_data.get('sources_completed', [])) / max(len(collection_data.get('sources_attempted', [])), 1)) * 100:.1f}%
"""
        
        # Training Summary
        training_results = results.get('training_results')
        if training_results:
            if training_results.get('skipped'):
                reasons = '; '.join(training_results.get('reasons', []))
                report += f"""
🧠 MODEL TRAINING: SKIPPED
   Reasons: {reasons}
"""
            else:
                training_data = training_results.get('training_result', {})
                best_model = training_data.get('best_model', {})
                report += f"""
🧠 MODEL TRAINING: COMPLETED
   Best model accuracy: {best_model.get('final_accuracy', 0):.2f}%
   Improvement: +{best_model.get('improvement_percent', 0):.2f}%
   Training stages: {len(training_data.get('stages_completed', []))}
   Model recommendation: {training_data.get('deployment_recommendation', 'Unknown')}
"""
        
        # Deployment Summary
        deployment_results = results.get('deployment_results')
        if deployment_results:
            if deployment_results.get('skipped'):
                report += f"""
🚀 MODEL DEPLOYMENT: SKIPPED
   Reason: {deployment_results.get('reason', 'Unknown')}
"""
            else:
                deployment_data = deployment_results.get('deployment_result', {})
                report += f"""
🚀 MODEL DEPLOYMENT: COMPLETED
   Status: {deployment_data.get('final_status', 'Unknown')}
   Model path: {deployment_data.get('model_path', 'Unknown')}
   Deployment type: {deployment_data.get('deployment_type', 'Unknown')}
"""
        
        # Errors
        if results['errors']:
            report += f"""
⚠️ ERRORS ENCOUNTERED:
"""
            for error in results['errors'][:5]:  # Show first 5 errors
                report += f"   • {error}\n"
        
        # Current System Status
        current_model = self._get_current_model_performance()
        report += f"""
📊 CURRENT SYSTEM STATUS:
   Production model: {'Yes' if current_model.get('has_model') else 'No'}
"""
        if current_model.get('has_model'):
            report += f"""   Model accuracy: {current_model.get('accuracy', 0):.2f}%
   Deployed: {current_model.get('deployed_at', 'Unknown')}
"""
        
        # Recommendations
        report += f"""
💡 RECOMMENDATIONS:
"""
        
        if not current_model.get('has_model'):
            report += "   • No production model deployed - consider manual training\n"
        
        if len(results['errors']) > 0:
            report += "   • Check error logs and resolve issues before next update\n"
        
        if results.get('data_collection_results', {}).get('skipped'):
            report += "   • Monitor data collection sources for issues\n"
        
        data_analysis = training_results.get('data_analysis', {}) if training_results else {}
        if data_analysis.get('new_files_this_week', 0) < 100:
            report += "   • Low data collection rate - check scraper configurations\n"
        
        report += f"""
📈 NEXT SCHEDULED UPDATE: {(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')}

{'='*60}
Generated by Soccer AI Training Automation System
"""
        
        return report
    
    async def _send_email_notification(self, subject: str, body: str) -> bool:
        """Send email notification"""
        if not self.email_config.get('enabled', False):
            logger.info("📧 Email notifications disabled")
            return False
        
        try:
            msg = MimeMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(self.email_config['to_emails'])
            msg['Subject'] = subject
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"📧 Email notification sent: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def _check_system_health(self) -> Dict[str, Any]:
        """Check system health before starting updates"""
        try:
            import psutil
            
            # Check disk space
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            
            # Check memory
            memory = psutil.virtual_memory()
            available_memory_gb = memory.available / (1024**3)
            
            health_check = {
                'healthy': True,
                'warnings': [],
                'errors': [],
                'disk_free_gb': round(free_gb, 2),
                'memory_available_gb': round(available_memory_gb, 2)
            }
            
            # Check minimum requirements
            if free_gb < 5.0:
                health_check['errors'].append(f"Low disk space: {free_gb:.1f} GB free")
                health_check['healthy'] = False
            elif free_gb < 10.0:
                health_check['warnings'].append(f"Disk space getting low: {free_gb:.1f} GB free")
            
            if available_memory_gb < 1.0:
                health_check['errors'].append(f"Low memory: {available_memory_gb:.1f} GB available")
                health_check['healthy'] = False
            elif available_memory_gb < 2.0:
                health_check['warnings'].append(f"Memory getting low: {available_memory_gb:.1f} GB available")
            
            return health_check
            
        except Exception as e:
            return {
                'healthy': False,
                'errors': [f"Health check failed: {e}"],
                'warnings': []
            }
    
    async def run_weekly_update(self, force: bool = False) -> Dict[str, Any]:
        """Run the complete weekly update process"""
        self.update_results['started_at'] = datetime.now().isoformat()
        
        logger.info(f"🚀 Starting weekly training update: {self.update_session_id}")
        
        try:
            # Phase 1: System Health Check
            logger.info("🏥 Checking system health...")
            health_check = self._check_system_health()
            
            if not health_check['healthy'] and not force:
                error_msg = f"System health check failed: {health_check['errors']}"
                logger.error(error_msg)
                self.update_results['errors'].append(error_msg)
                raise Exception(error_msg)
            
            if health_check['warnings']:
                for warning in health_check['warnings']:
                    logger.warning(warning)
            
            self.update_results['phases_completed'].append('health_check')
            
            # Phase 2: Data Collection
            try:
                logger.info("📥 Phase 2: Weekly data collection...")
                data_results = await self._run_weekly_data_collection()
                self.update_results['data_collection_results'] = data_results
                self.update_results['phases_completed'].append('data_collection')
            except Exception as e:
                error_msg = f"Data collection phase failed: {e}"
                logger.error(error_msg)
                self.update_results['errors'].append(error_msg)
                self.update_results['phases_failed'].append('data_collection')
            
            # Phase 3: Model Training
            try:
                logger.info("🧠 Phase 3: Conditional model training...")
                training_results = await self._run_conditional_training()
                self.update_results['training_results'] = training_results
                self.update_results['phases_completed'].append('training')
            except Exception as e:
                error_msg = f"Training phase failed: {e}"
                logger.error(error_msg)
                self.update_results['errors'].append(error_msg)
                self.update_results['phases_failed'].append('training')
                training_results = {'executed': False, 'error': str(e)}
            
            # Phase 4: Model Deployment
            try:
                logger.info("🚀 Phase 4: Conditional model deployment...")
                deployment_results = await self._run_conditional_deployment(training_results)
                self.update_results['deployment_results'] = deployment_results
                self.update_results['phases_completed'].append('deployment')
            except Exception as e:
                error_msg = f"Deployment phase failed: {e}"
                logger.error(error_msg)
                self.update_results['errors'].append(error_msg)
                self.update_results['phases_failed'].append('deployment')
            
            # Phase 5: Generate Report and Send Notifications
            self.update_results['completed_at'] = datetime.now().isoformat()
            
            logger.info("📊 Generating update report...")
            report = self._generate_update_report()
            
            # Save report
            report_file = Path(f"training_data/logs/weekly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, 'w') as f:
                f.write(report)
            
            # Send email notification
            subject = f"Soccer AI Weekly Update - {datetime.now().strftime('%Y-%m-%d')}"
            if await self._send_email_notification(subject, report):
                self.update_results['notifications_sent'].append('email')
            
            # Print report
            print(report)
            
            # Save results
            results_file = Path(f"training_data/logs/weekly_update_{self.update_session_id}.json")
            with open(results_file, 'w') as f:
                json.dump(self.update_results, f, indent=2, default=str)
            
            logger.info("✅ Weekly update completed successfully!")
            
            return self.update_results
            
        except Exception as e:
            error_msg = f"Weekly update failed: {e}"
            logger.error(error_msg)
            
            self.update_results['error_message'] = error_msg
            self.update_results['completed_at'] = datetime.now().isoformat()
            
            # Send error notification
            error_subject = f"Soccer AI Weekly Update FAILED - {datetime.now().strftime('%Y-%m-%d')}"
            error_report = f"""
❌ WEEKLY UPDATE FAILED

Session: {self.update_session_id}
Error: {error_msg}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Phases completed: {self.update_results['phases_completed']}
Phases failed: {self.update_results['phases_failed']}

Please check the logs and resolve the issue.
"""
            
            await self._send_email_notification(error_subject, error_report)
            
            raise

async def main():
    parser = argparse.ArgumentParser(description="Weekly Soccer AI Training Update")
    parser.add_argument('--force', action='store_true',
                       help='Force update even if system health checks fail')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without executing')
    parser.add_argument('--report-only', action='store_true',
                       help='Generate and send report only')
    
    args = parser.parse_args()
    
    try:
        updater = WeeklyTrainingUpdater()
        
        if args.dry_run:
            logger.info("🔍 DRY RUN MODE - Analyzing what would be done...")
            
            # Analyze current state
            data_analysis = updater._analyze_training_data_freshness()
            last_training = updater._get_last_training_session()
            current_model = updater._get_current_model_performance()
            
            print("\n📊 WEEKLY UPDATE ANALYSIS")
            print("=" * 50)
            print(f"New data this week: {data_analysis.get('new_files_this_week', 0)}")
            print(f"Data collection needed: {data_analysis.get('data_collection_needed', False)}")
            print(f"Training recommended: {data_analysis.get('training_recommended', False)}")
            print(f"Last training: {last_training['days_ago'] if last_training else 'Never'} days ago")
            print(f"Current model: {'Available' if current_model.get('has_model') else 'None'}")
            
            return 0
        
        if args.report_only:
            logger.info("📊 Generating status report only...")
            # This would generate a status report without running updates
            report = updater._generate_update_report()
            print(report)
            return 0
        
        # Run full weekly update
        result = await updater.run_weekly_update(force=args.force)
        
        logger.info("🎉 Weekly update process completed!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Weekly update process failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)