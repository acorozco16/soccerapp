# Management Scripts

Administrative scripts for the Soccer Video Analysis app maintenance, backup, and cleanup operations.

## 📁 Scripts Overview

### 🧹 cleanup.py
**Purpose**: Clean up old files, failed videos, and temporary data

**Usage**:
```bash
# Show what would be cleaned (dry run)
python scripts/cleanup.py --dry-run

# Clean videos older than 7 days
python scripts/cleanup.py --days-old 7

# Clean only failed videos
python scripts/cleanup.py --failed-only

# Clean only temporary files
python scripts/cleanup.py --temp-only

# Show storage statistics
python scripts/cleanup.py --stats
```

**Features**:
- Remove videos older than specified days
- Clean failed video processing attempts
- Remove orphaned files and temporary data
- Disk space analysis and reporting
- Safe dry-run mode for testing

---

### 💾 backup.py
**Purpose**: Create and restore backups of database and video data

**Usage**:
```bash
# Create database-only backup
python scripts/backup.py --create

# Create backup including video files (large)
python scripts/backup.py --create --include-videos

# List available backups
python scripts/backup.py --list

# Restore database from backup
python scripts/backup.py --restore backups/soccer_backup_20241225_120000.zip

# Restore including video files
python scripts/backup.py --restore backup.zip --restore-videos

# Export metadata to JSON
python scripts/backup.py --export-metadata video_metadata.json
```

**Features**:
- Database backup and restore
- Optional video file inclusion
- Backup compression (ZIP format)
- Metadata export for analysis
- Version tracking and history

---

### 🔧 maintenance.py
**Purpose**: Routine maintenance, optimization, and health monitoring

**Usage**:
```bash
# Run all maintenance tasks
python scripts/maintenance.py --all

# Optimize database performance
python scripts/maintenance.py --optimize-db

# Check file integrity
python scripts/maintenance.py --check-integrity

# Generate usage report (30 days)
python scripts/maintenance.py --usage-report 30

# System health check
python scripts/maintenance.py --health-check
```

**Features**:
- Database optimization (VACUUM, ANALYZE)
- File integrity verification
- Usage statistics and reporting
- System health monitoring
- Performance metrics analysis

## 🚀 Quick Start Guide

### Daily Maintenance
```bash
# Basic health check and cleanup
python scripts/maintenance.py --health-check
python scripts/cleanup.py --failed-only --dry-run
```

### Weekly Maintenance
```bash
# Full maintenance cycle
python scripts/maintenance.py --all
python scripts/cleanup.py --days-old 14
python scripts/backup.py --create
```

### Monthly Maintenance
```bash
# Deep maintenance and backup
python scripts/maintenance.py --usage-report 30
python scripts/cleanup.py --days-old 30
python scripts/backup.py --create --include-videos
```

## 📊 Monitoring Commands

### Storage Analysis
```bash
# Check current storage usage
python scripts/cleanup.py --stats

# Find large files and directories
du -sh uploads/* | sort -hr
```

### Database Health
```bash
# Quick database stats
python scripts/maintenance.py --optimize-db

# Detailed integrity check
python scripts/maintenance.py --check-integrity
```

### System Performance
```bash
# Full health report
python scripts/maintenance.py --health-check

# Usage patterns
python scripts/maintenance.py --usage-report 7
```

## 🔄 Automation Setup

### Cron Jobs (Linux/Mac)
```bash
# Edit crontab
crontab -e

# Add these lines for automated maintenance:

# Daily cleanup at 2 AM
0 2 * * * cd /path/to/soccer-app && python scripts/cleanup.py --failed-only

# Weekly maintenance on Sunday at 3 AM
0 3 * * 0 cd /path/to/soccer-app && python scripts/maintenance.py --all

# Monthly backup on 1st day at 4 AM
0 4 1 * * cd /path/to/soccer-app && python scripts/backup.py --create
```

### GitHub Actions (CI/CD)
```yaml
# .github/workflows/maintenance.yml
name: Weekly Maintenance
on:
  schedule:
    - cron: '0 2 * * 0'  # Sunday 2 AM
jobs:
  maintenance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run maintenance
        run: python scripts/maintenance.py --all
```

## 🛠 Configuration

### Environment Variables
```bash
# Backup location
export BACKUP_DIR="/path/to/backups"

# Cleanup settings
export CLEANUP_DAYS=14
export MAX_LOG_SIZE_MB=100

# Database settings
export DB_OPTIMIZE_INTERVAL=7  # days
```

### Script Settings
Scripts use these default settings (configurable via command line):

- **Cleanup**: 7 days retention, failed videos only
- **Backup**: Database only (videos optional)
- **Maintenance**: 30-day usage reports
- **Health**: All system checks enabled

## 📈 Performance Tips

### For Large Installations
```bash
# Process cleanup in batches
python scripts/cleanup.py --days-old 7 --dry-run | head -100

# Optimize during low usage periods
python scripts/maintenance.py --optimize-db

# Use incremental backups
python scripts/backup.py --create  # Database only
```

### Storage Optimization
```bash
# Check largest directories first
du -sh uploads/*/ | sort -hr

# Clean oldest files first
python scripts/cleanup.py --days-old 30

# Compress old backups
gzip backups/*.zip
```

## 🚨 Troubleshooting

### Common Issues

**1. Permission Errors**
```bash
# Fix script permissions
chmod +x scripts/*.py

# Fix directory permissions
chmod -R 755 uploads/
```

**2. Database Locked**
```bash
# Check for running processes
ps aux | grep python

# Wait for processes to complete or restart app
```

**3. Disk Space Full**
```bash
# Emergency cleanup
python scripts/cleanup.py --days-old 1 --temp-only

# Find large files
find uploads/ -size +100M -ls
```

**4. Backup Restoration Issues**
```bash
# Verify backup integrity
unzip -t backup.zip

# Restore to temporary location first
python scripts/backup.py --restore backup.zip --restore-videos
```

### Logs and Debugging
```bash
# Check script logs
tail -f maintenance.log

# Verbose output
python scripts/maintenance.py --health-check -v

# Debug mode
export DEBUG=1
python scripts/cleanup.py --stats
```

## 📋 Best Practices

### Safety Guidelines
1. **Always use --dry-run first** for cleanup operations
2. **Create backups before major maintenance**
3. **Test restore procedures regularly**
4. **Monitor disk space and performance**
5. **Keep maintenance logs**

### Scheduling Recommendations
- **Health checks**: Daily
- **Failed video cleanup**: Daily
- **Full maintenance**: Weekly
- **Database optimization**: Weekly
- **Backups**: Weekly (database), Monthly (with videos)
- **Old file cleanup**: Monthly

### Security Considerations
- Backup files may contain sensitive data
- Secure backup storage location
- Regular backup testing
- Access control for maintenance scripts

---

## 📞 Support

For issues with these scripts:
1. Check the troubleshooting section above
2. Review log files for error details  
3. Test with --dry-run flags first
4. Verify file permissions and disk space
5. Create an issue with logs and error messages

**Happy maintaining!** 🔧✨