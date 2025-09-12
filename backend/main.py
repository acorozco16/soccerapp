from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
from datetime import datetime, timedelta
import asyncio
from typing import Optional

# Configure logging first
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from database import init_db, SessionLocal
from models import Video, ProcessingStatus
from health import get_health_status
from drill_api import drill_router
from auth_api import auth_router

# Conditional import for video processor (only needed for actual video processing)
try:
    from video_processor import VideoProcessor
    VIDEO_PROCESSING_ENABLED = True
except ImportError as e:
    logger.warning(f"Video processing disabled - missing dependencies: {e}")
    VideoProcessor = None
    VIDEO_PROCESSING_ENABLED = False

# Initialize FastAPI app
app = FastAPI(title="Soccer Video Analysis API", version="1.0.0")

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_db()

# Include drill analysis router
app.include_router(drill_router)

# Include authentication router
app.include_router(auth_router)

# Configure upload paths
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
RAW_DIR = UPLOAD_DIR / "raw"
PROCESSED_DIR = UPLOAD_DIR / "processed"
FRAMES_DIR = UPLOAD_DIR / "frames"

# Create directories
for dir_path in [RAW_DIR, PROCESSED_DIR, FRAMES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configuration
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {".mp4", ".mov"}
MIN_DURATION = 10  # seconds
MAX_DURATION = 300  # 5 minutes


@app.get("/")
async def root():
    return {"message": "Soccer Video Analysis API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "soccer-video-api"
    }


@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Generate unique video ID
    video_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    file_path = RAW_DIR / f"{video_id}{file_ext}"
    
    # Save uploaded file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")
    
    # Validate file size
    if file_path.stat().st_size > MAX_FILE_SIZE:
        file_path.unlink()
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB"
        )
    
    # Create database entry
    db = SessionLocal()
    try:
        video = Video(
            id=video_id,
            filename=file.filename,
            file_path=str(file_path),
            status=ProcessingStatus.UPLOADED,
            upload_time=datetime.utcnow()
        )
        db.add(video)
        db.commit()
    except Exception as e:
        logger.error(f"Database error: {e}")
        file_path.unlink()
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        db.close()
    
    # Start processing in background
    background_tasks.add_task(process_video, video_id, str(file_path))
    
    return {
        "video_id": video_id,
        "status": "uploaded",
        "message": "Video uploaded successfully. Processing will begin shortly."
    }


async def process_video(video_id: str, file_path: str):
    """Background task to process uploaded video"""
    if not VIDEO_PROCESSING_ENABLED:
        logger.error(f"Video processing disabled for {video_id} - missing dependencies")
        return
        
    processor = VideoProcessor()
    db = SessionLocal()
    
    try:
        # Update status to processing
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            logger.error(f"Video {video_id} not found in database")
            return
            
        video.status = ProcessingStatus.PROCESSING
        video.process_start_time = datetime.utcnow()
        db.commit()
        
        # Process the video
        logger.info(f"Starting processing for video {video_id}")
        results = await processor.analyze_video(file_path, video_id)
        
        # Save results
        results_path = PROCESSED_DIR / f"{video_id}_analysis.json"
        import json
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Update database with results
        video.status = ProcessingStatus.COMPLETED
        video.process_end_time = datetime.utcnow()
        video.results_path = str(results_path)
        video.total_touches = results.get("total_ball_touches", 0)
        video.confidence_score = results.get("confidence_score", 0.0)
        db.commit()
        
        logger.info(f"Completed processing for video {video_id}")
        
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {e}")
        video = db.query(Video).filter(Video.id == video_id).first()
        if video:
            video.status = ProcessingStatus.ERROR
            video.error_message = str(e)
            db.commit()
    finally:
        db.close()


@app.get("/status/{video_id}")
async def get_status(video_id: str):
    db = SessionLocal()
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        response = {
            "video_id": video_id,
            "status": video.status.value,
            "upload_time": video.upload_time.isoformat(),
        }
        
        if video.status == ProcessingStatus.PROCESSING:
            # Check for timeout (3 minutes)
            if video.process_start_time:
                elapsed = datetime.utcnow() - video.process_start_time
                if elapsed > timedelta(minutes=3):
                    video.status = ProcessingStatus.ERROR
                    video.error_message = "Processing timeout"
                    db.commit()
                    response["status"] = ProcessingStatus.ERROR.value
                    response["error"] = "Processing timeout"
                else:
                    response["processing_time"] = elapsed.total_seconds()
        
        elif video.status == ProcessingStatus.ERROR:
            response["error"] = video.error_message
            
        elif video.status == ProcessingStatus.COMPLETED:
            response["results_available"] = True
            if video.process_start_time and video.process_end_time:
                response["processing_time"] = (
                    video.process_end_time - video.process_start_time
                ).total_seconds()
        
        return response
        
    finally:
        db.close()


@app.get("/results/{video_id}")
async def get_results(video_id: str):
    db = SessionLocal()
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
            
        if video.status != ProcessingStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Video processing not completed. Current status: {video.status.value}"
            )
        
        # Load results from file
        if not video.results_path or not Path(video.results_path).exists():
            raise HTTPException(status_code=404, detail="Results file not found")
            
        import json
        with open(video.results_path, "r") as f:
            results = json.load(f)
            
        return results
        
    finally:
        db.close()


@app.get("/frame/{video_id}/{frame_name}")
async def get_frame(video_id: str, frame_name: str):
    """Serve annotated debug frames"""
    frame_path = FRAMES_DIR / video_id / frame_name
    
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
    
    return FileResponse(frame_path, media_type="image/jpeg")


@app.delete("/video/{video_id}")
async def delete_video(video_id: str):
    """Delete video and all associated data"""
    db = SessionLocal()
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Delete files
        paths_to_delete = [
            Path(video.file_path),
            FRAMES_DIR / video_id,
        ]
        if video.results_path:
            paths_to_delete.append(Path(video.results_path))
            
        for path in paths_to_delete:
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        
        # Delete from database
        db.delete(video)
        db.commit()
        
        return {"message": "Video deleted successfully"}
        
    finally:
        db.close()


@app.get("/videos")
async def list_videos(limit: int = 10):
    """List recent video analyses"""
    db = SessionLocal()
    try:
        videos = db.query(Video).order_by(
            Video.upload_time.desc()
        ).limit(limit).all()
        
        return [
            {
                "video_id": v.id,
                "filename": v.filename,
                "status": v.status.value,
                "upload_time": v.upload_time.isoformat(),
                "total_touches": v.total_touches,
                "confidence_score": v.confidence_score,
            }
            for v in videos
        ]
        
    finally:
        db.close()


@app.get("/metadata/{video_id}")
async def get_metadata(video_id: str):
    """Get video metadata"""
    db = SessionLocal()
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Extract metadata using imageio
        import imageio
        reader = imageio.get_reader(video.file_path)
        metadata = reader.get_meta_data()
        
        return {
            "video_id": video_id,
            "filename": video.filename,
            "duration": metadata.get("duration", 0),
            "fps": metadata.get("fps", 0),
            "size": [metadata.get("size", [0, 0])[0], metadata.get("size", [0, 0])[1]],
            "format": Path(video.file_path).suffix,
        }
        
    except Exception as e:
        logger.error(f"Error getting metadata: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract metadata")
    finally:
        db.close()


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and deployment"""
    return get_health_status()


# Training Dashboard Integration
@app.get("/training-dashboard")
async def training_dashboard():
    """Serve the training dashboard"""
    try:
        from training_data.dashboard.dashboard_api import training_api
        
        # Import and include the training API routes
        app.include_router(training_api, tags=["Training Automation"])
        
        # Redirect to the dashboard
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/training-api/")
        
    except ImportError as e:
        logger.warning(f"Training dashboard not available: {e}")
        return {"error": "Training dashboard not available", "message": str(e)}

# Include training API routes if available
# TEMPORARILY DISABLED - Path issues with logs directory
# try:
#     from training_data.dashboard.dashboard_api import training_api
#     app.include_router(training_api, tags=["Training Automation"])
#     logger.info("Training automation API enabled at /training-api")
# except ImportError as e:
#     logger.warning(f"Training automation API not available: {e}")


if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("ENVIRONMENT", "development") == "development"
    
    uvicorn.run(app, host="0.0.0.0", port=port, reload=reload)