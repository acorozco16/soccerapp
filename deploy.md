# Deployment Guide

Complete deployment guide for the Soccer Ball Touch Counter application.

## 🌐 Architecture Overview

- **Frontend**: Next.js deployed on Vercel
- **Backend**: FastAPI deployed on Railway
- **Database**: SQLite (file-based, attached to Railway volume)
- **File Storage**: Railway volume for video uploads

## 🚀 Quick Deployment

### 1. Vercel Frontend Deployment

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy frontend
cd frontend
vercel --prod
```

During deployment:
- Set `NEXT_PUBLIC_API_URL` to your Railway backend URL
- Vercel will automatically detect Next.js and configure build settings

### 2. Railway Backend Deployment

1. **Connect Repository**:
   - Go to [Railway](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository

2. **Configure Environment Variables**:
   ```
   PORT=8000
   ENVIRONMENT=production
   DATABASE_URL=sqlite:///./production.db
   UPLOAD_DIR=/app/uploads
   MAX_FILE_SIZE=104857600
   ```

3. **Add Volume for Storage**:
   - In Railway dashboard, go to Variables tab
   - Add a volume mount: `/app/uploads` (5GB recommended)

4. **Custom Start Command**:
   ```
   cd backend && python main.py
   ```

## 📋 Detailed Setup Instructions

### Frontend (Vercel)

#### Prerequisites
- Vercel account
- GitHub repository connected

#### Environment Variables
Set in Vercel dashboard:
```env
NEXT_PUBLIC_API_URL=https://your-app-name.railway.app
```

#### Build Settings
Vercel auto-detects Next.js, but verify:
- **Framework Preset**: Next.js
- **Build Command**: `npm run build`
- **Output Directory**: `.next`
- **Root Directory**: `frontend`

#### Custom Domains
1. Add domain in Vercel dashboard
2. Update DNS records as instructed
3. SSL certificates are automatic

### Backend (Railway)

#### Prerequisites
- Railway account
- GitHub repository

#### Deployment Configuration
1. **Service Settings**:
   - **Source**: GitHub repository
   - **Root Directory**: `/` (whole repo)
   - **Build Command**: `cd backend && pip install -r requirements.txt`
   - **Start Command**: `cd backend && python main.py`

2. **Environment Variables**:
   ```env
   PORT=8000
   ENVIRONMENT=production
   PYTHONPATH=/app/backend
   DATABASE_URL=sqlite:///./production.db
   UPLOAD_DIR=/app/uploads
   MAX_FILE_SIZE=104857600
   DEBUG=false
   ```

3. **Volume Configuration**:
   - **Mount Path**: `/app/uploads`
   - **Size**: 5GB (adjustable based on usage)

4. **Health Check**:
   - **Path**: `/health`
   - **Timeout**: 30 seconds

#### Resource Requirements
- **Memory**: 2GB (for video processing)
- **CPU**: 1000m (1 core)
- **Storage**: 5GB volume for uploads

### Database Setup

The app uses SQLite for simplicity. For production scale:

#### SQLite (Default)
- Automatic setup on first run
- Stored in Railway volume
- Suitable for moderate usage

#### PostgreSQL (Optional Upgrade)
```bash
# Add Railway PostgreSQL addon
railway add postgresql

# Update environment variables
DATABASE_URL=postgresql://user:pass@host:port/db
```

## 🔒 Security Considerations

### Environment Variables
Never commit sensitive data:
```bash
# Use .env.example as template
cp .env.example .env.local
# Add your actual values to .env.local
```

### CORS Configuration
Update `backend/main.py` for production:
```python
origins = [
    "https://your-frontend-domain.vercel.app",
    "https://your-custom-domain.com"
]
```

### File Upload Security
- Max file size: 100MB (configurable)
- Allowed formats: MP4, MOV only
- Virus scanning recommended for production

## 📊 Monitoring & Logging

### Railway Logs
```bash
# View logs
railway logs

# Follow logs in real-time
railway logs --follow
```

### Vercel Analytics
- Enable in Vercel dashboard
- Monitor performance and usage

### Custom Monitoring
Add to `backend/main.py`:
```python
import logging
logging.basicConfig(level=logging.INFO)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}
```

## 🎛 Configuration

### Performance Tuning

#### Video Processing Optimization
```env
# Reduce quality for faster processing
VIDEO_QUALITY_TARGET=480
FRAME_SKIP_RATIO=3

# Adjust detection confidence
BALL_DETECTION_CONFIDENCE=0.6
POSE_DETECTION_CONFIDENCE=0.8
```

#### Memory Management
```python
# In video_processor.py
# Reduce batch sizes for limited memory
MAX_CONCURRENT_VIDEOS = 2
FRAME_BATCH_SIZE = 10
```

### Scaling Considerations

#### Horizontal Scaling
- Railway supports multiple instances
- Use Redis for session storage if needed
- Consider file storage service (AWS S3)

#### Cost Optimization
- Use Railway's sleep mode for development
- Implement video cleanup after processing
- Monitor usage and adjust resources

## 🚦 Testing Deployment

### Pre-deployment Checklist
- [ ] Environment variables configured
- [ ] CORS origins updated
- [ ] File upload limits set
- [ ] Health checks working
- [ ] Database connection tested

### Post-deployment Testing
```bash
# Test API health
curl https://your-backend.railway.app/health

# Test file upload
curl -X POST -F "file=@test.mp4" \
     https://your-backend.railway.app/upload

# Test frontend
curl https://your-frontend.vercel.app
```

### Load Testing
```bash
# Install artillery for load testing
npm install -g artillery

# Test with concurrent uploads
artillery io test-load-upload.yaml
```

## 🛠 Troubleshooting

### Common Issues

#### Build Failures
- Check Node.js/Python versions
- Verify dependency versions
- Review build logs

#### Memory Issues
- Reduce video quality settings
- Increase Railway memory allocation
- Implement video streaming for large files

#### CORS Errors
- Verify frontend URL in backend CORS config
- Check environment variable values
- Test with browser dev tools

#### Database Errors
- Verify volume mount path
- Check file permissions
- Monitor disk space usage

### Debug Mode
```env
# Enable debug mode
DEBUG=true
LOG_LEVEL=DEBUG
```

## 📈 Production Optimization

### CDN Setup
- Use Vercel's built-in CDN for frontend
- Consider CloudFlare for additional caching

### Database Optimization
```sql
-- Add indexes for better performance
CREATE INDEX idx_videos_created_at ON videos(created_at);
CREATE INDEX idx_videos_status ON videos(status);
```

### Caching Strategy
```python
# Add Redis caching for results
REDIS_URL=redis://redis:6379
CACHE_TTL=3600  # 1 hour
```

## 💰 Cost Estimation

### Vercel (Frontend)
- **Hobby Plan**: Free for personal projects
- **Pro Plan**: $20/month for teams
- **Enterprise**: Custom pricing

### Railway (Backend)
- **Developer Plan**: $5/month
- **Team Plan**: $20/month
- **Usage-based pricing** for compute and storage

### Total Monthly Cost
- **Development**: ~$0-10/month
- **Small Business**: ~$25-50/month
- **Enterprise**: Custom pricing

## 🔄 Continuous Deployment

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Railway
        run: railway deploy
```

### Automatic Deployments
- Vercel: Auto-deploys on git push
- Railway: Auto-deploys on git push to main branch

---

## 📞 Support

For deployment issues:
1. Check logs in respective platforms
2. Review environment variables
3. Test locally first
4. Check platform status pages
5. Contact platform support if needed

**Ready to deploy your soccer analysis app!** 🚀