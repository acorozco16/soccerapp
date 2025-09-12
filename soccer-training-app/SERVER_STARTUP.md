# Soccer Training API Server Startup Instructions

## Server Details
- **IP Address**: 147.182.224.87
- **Domain**: soccertrainingapp.org
- **Server Location**: `/root/soccerapp/backend/`
- **Virtual Environment**: `/root/soccerapp/venv/`

## Startup Commands

```bash
# SSH into DigitalOcean droplet
ssh root@147.182.224.87

# Navigate to app directory
cd /root/soccerapp

# Activate virtual environment
source venv/bin/activate

# Navigate to backend
cd backend

# Install missing dependencies (if needed)
pip install PyJWT

# Start the server
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

# Or run in background
nohup python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 &
```

## Common Issues & Fixes

### Missing JWT Module
```bash
pip install PyJWT
```

### Missing Other Dependencies
```bash
pip install -r requirements.txt
```

### Check Server Status
```bash
curl https://soccertrainingapp.org/health
```

### View Running Processes
```bash
ps aux | grep python
```

### Kill Server Process
```bash
pkill -f uvicorn
```

## Server Architecture
- **Framework**: FastAPI with Uvicorn
- **Database**: SQLite (`/root/soccerapp/soccer_analysis.db`)
- **Uploads**: `/root/soccerapp/uploads/`
- **Port**: 8000 (proxied through nginx to 80/443)