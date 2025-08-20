# 🧭 Project Navigation Guide

## 📁 **Quick File Finder**

### **🚀 Getting Started**
```bash
# Main applications
backend/main.py                    # Start FastAPI backend server
frontend/package.json             # Next.js web app
mobile/soccer-training-app/        # React Native mobile app
```

### **📚 Documentation (All in /docs/)**
```bash
docs/README.md                     # Project overview  
docs/AUTOMATION_GUIDE.md           # ML training automation
docs/Backend_Architecture_Guide.md # Complete backend guide
docs/deploy.md                     # Deployment instructions
docs/digitalocean\ 10:6.rtf       # Conversation history
```

### **🤖 AI/ML Pipeline (All in /ai-training/)**
```bash
ai-training/training_data/         # Data collection & processing
ai-training/models/               # Trained model files
ai-training/train_*.py            # Training scripts
ai-training/Train_*.ipynb         # Jupyter notebooks
```

### **🛠 Scripts by Function**
```bash
# Data Collection
scripts/data-collection/collect_*.py
scripts/data-collection/create_*.py

# Analysis & Testing  
scripts/analysis/analyze_*.py
scripts/analysis/diagnose_*.py
scripts/analysis/test_*.py

# Setup & Configuration
scripts/setup/setup_*.py
scripts/setup/quick_*.py
```

### **🚢 Deployment (All in /deployment/)**
```bash
# Docker
deployment/docker/Dockerfile
deployment/docker/docker-compose.yml

# Cloud Platforms
deployment/cloud/railway.toml      # Railway deployment
deployment/cloud/render.yaml       # Render deployment  
deployment/cloud/vercel.json       # Vercel deployment

# Dependencies
deployment/requirements/requirements.txt
deployment/requirements/requirements_*.txt
```

### **💾 Data & Storage (All in /data/)**
```bash
data/databases/soccer_analysis.db  # Main SQLite database
data/models/yolov8n.pt             # YOLO model file
data/sample_videos/                # Test videos
data/uploads/                      # Video uploads directory
data/runs/                         # Training run outputs
```

## 🔍 **Common Tasks**

### **Start Development**
```bash
# Backend
cd backend && python main.py

# Frontend  
cd frontend && npm run dev

# Mobile
cd mobile/soccer-training-app && npm start
```

### **Run Analysis**
```bash
# Test current model
python scripts/analysis/test_current_model.py

# Analyze performance
python scripts/analysis/analyze_current_performance.py
```

### **Train New Model**
```bash
# Quick training
python ai-training/train_soccer_model.py

# Progressive training
python ai-training/automation/train_progressive.py
```

### **Deploy**
```bash
# Check deployment configs
ls deployment/cloud/

# Docker deployment
cd deployment/docker && docker-compose up
```

## 📂 **Before Reorganization (77+ root files) → After (Clean Structure)**

### **Benefits of New Structure:**
✅ **Easy Navigation** - Know exactly where everything is  
✅ **Logical Grouping** - Related files together  
✅ **Clear Separation** - Dev vs deployment vs documentation  
✅ **Professional Structure** - Industry-standard organization  
✅ **Future-Proof** - Easy to add new components  

### **For New Claude Code Sessions:**
1. **Start here**: `README.md` - Project overview
2. **Core apps**: `backend/`, `frontend/`, `mobile/`  
3. **Documentation**: `docs/` folder - All guides & references
4. **Need to deploy**: `deployment/` folder - All configs
5. **AI/ML work**: `ai-training/` folder - Complete ML pipeline

This structure makes it **10x easier** for any developer (including future Claude Code sessions) to understand and work with the project.