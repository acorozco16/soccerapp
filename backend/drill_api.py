"""
Drill Analysis API Endpoints
Integrates the drill framework with FastAPI
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Depends
from pathlib import Path
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
import logging

from database import SessionLocal
from models import Video, ProcessingStatus
from drill_analyzer import drill_registry, DrillType
from unified_processor import UnifiedVideoProcessor
from auth import get_current_user, get_optional_user
from supabase_client import create_drill_attempt, update_user_progress

logger = logging.getLogger(__name__)

# Create router for drill endpoints
drill_router = APIRouter(prefix="/drill", tags=["Drill Analysis"])

# Initialize unified processor lazily to avoid import issues
processor = None

def get_processor():
    """Get or create unified processor"""
    global processor
    if processor is None:
        processor = UnifiedVideoProcessor()
    return processor

# Configure paths (same as main.py)
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
RAW_DIR = UPLOAD_DIR / "raw"
PROCESSED_DIR = UPLOAD_DIR / "processed"


@drill_router.get("/available")
async def list_available_drills():
    """List all available drill types"""
    try:
        drills = drill_registry.list_drills()
        return {
            "drills": drills,
            "total_count": len(drills)
        }
    except Exception as e:
        logger.error(f"Error listing drills: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve drill list")


@drill_router.get("/info/{drill_type}")
async def get_drill_info(drill_type: str):
    """Get detailed information about a specific drill"""
    try:
        drill_info = get_processor().get_drill_info(drill_type)
        if not drill_info:
            raise HTTPException(
                status_code=404, 
                detail=f"Drill type '{drill_type}' not found"
            )
        
        return drill_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting drill info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve drill information")


@drill_router.post("/analyze")
async def analyze_drill(
    background_tasks: BackgroundTasks,
    drill_type: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload video and analyze for specific drill type
    
    This creates a new analysis request for a specific drill.
    Multiple drill types can be analyzed from the same video.
    """
    
    # Validate drill type
    try:
        DrillType(drill_type)
    except ValueError:
        available_drills = [dt.value for dt in DrillType]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid drill type '{drill_type}'. Available: {available_drills}"
        )
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
        
    file_ext = Path(file.filename).suffix.lower()
    ALLOWED_EXTENSIONS = {".mp4", ".mov"}
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Generate unique analysis ID
    analysis_id = f"{drill_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    file_path = RAW_DIR / f"{analysis_id}{file_ext}"
    
    # Save uploaded file
    try:
        with open(file_path, "wb") as buffer:
            import shutil
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")
    
    # Validate file size
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    if file_path.stat().st_size > MAX_FILE_SIZE:
        file_path.unlink()
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB"
        )
    
    # Create database entry with drill type
    db = SessionLocal()
    try:
        video = Video(
            id=analysis_id,
            filename=f"{drill_type}_{file.filename}",
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
    
    # Start drill analysis in background
    background_tasks.add_task(process_drill_analysis, analysis_id, str(file_path), drill_type, current_user["id"])
    
    return {
        "analysis_id": analysis_id,
        "drill_type": drill_type,
        "status": "uploaded",
        "message": f"Video uploaded for {drill_type} analysis. Processing will begin shortly."
    }


@drill_router.post("/analyze/{video_id}")
async def analyze_existing_video(
    video_id: str,
    drill_type: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Analyze an existing video for a different drill type
    
    This allows analyzing the same video for multiple drills
    without re-uploading.
    """
    
    # Validate drill type
    try:
        DrillType(drill_type)
    except ValueError:
        available_drills = [dt.value for dt in DrillType]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid drill type '{drill_type}'. Available: {available_drills}"
        )
    
    # Check if original video exists
    db = SessionLocal()
    try:
        original_video = db.query(Video).filter(Video.id == video_id).first()
        if not original_video:
            raise HTTPException(status_code=404, detail="Original video not found")
        
        if not Path(original_video.file_path).exists():
            raise HTTPException(status_code=404, detail="Original video file not found")
        
        # Create new analysis entry
        analysis_id = f"{drill_type}_{video_id}_{uuid.uuid4().hex[:6]}"
        
        drill_video = Video(
            id=analysis_id,
            filename=f"{drill_type}_{original_video.filename}",
            file_path=original_video.file_path,  # Reuse same file
            status=ProcessingStatus.UPLOADED,
            upload_time=datetime.utcnow()
        )
        db.add(drill_video)
        db.commit()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        db.close()
    
    # Start drill analysis
    background_tasks.add_task(process_drill_analysis, analysis_id, original_video.file_path, drill_type, current_user["id"])
    
    return {
        "analysis_id": analysis_id,
        "original_video_id": video_id,
        "drill_type": drill_type,
        "status": "queued",
        "message": f"Started {drill_type} analysis for existing video"
    }


@drill_router.get("/status/{analysis_id}")
async def get_drill_status(analysis_id: str):
    """Get status of drill analysis"""
    db = SessionLocal()
    try:
        analysis = db.query(Video).filter(Video.id == analysis_id).first()
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Extract drill type from filename or ID
        drill_type = analysis.filename.split("_")[0] if "_" in analysis.filename else "unknown"
        
        response = {
            "analysis_id": analysis_id,
            "drill_type": drill_type,
            "status": analysis.status.value,
            "upload_time": analysis.upload_time.isoformat(),
        }
        
        if analysis.status == ProcessingStatus.PROCESSING:
            if analysis.process_start_time:
                from datetime import timedelta
                elapsed = datetime.utcnow() - analysis.process_start_time
                if elapsed > timedelta(minutes=5):  # Longer timeout for drill analysis
                    analysis.status = ProcessingStatus.ERROR
                    analysis.error_message = "Processing timeout"
                    db.commit()
                    response["status"] = ProcessingStatus.ERROR.value
                    response["error"] = "Processing timeout"
                else:
                    response["processing_time"] = elapsed.total_seconds()
        
        elif analysis.status == ProcessingStatus.ERROR:
            response["error"] = analysis.error_message
            
        elif analysis.status == ProcessingStatus.COMPLETED:
            response["results_available"] = True
            if analysis.process_start_time and analysis.process_end_time:
                response["processing_time"] = (
                    analysis.process_end_time - analysis.process_start_time
                ).total_seconds()
        
        return response
        
    finally:
        db.close()


@drill_router.get("/results/{analysis_id}")
async def get_drill_results(analysis_id: str):
    """Get drill analysis results"""
    db = SessionLocal()
    try:
        analysis = db.query(Video).filter(Video.id == analysis_id).first()
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
            
        if analysis.status != ProcessingStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Analysis not completed. Current status: {analysis.status.value}"
            )
        
        # Load results from file
        if not analysis.results_path or not Path(analysis.results_path).exists():
            raise HTTPException(status_code=404, detail="Results file not found")
            
        with open(analysis.results_path, "r") as f:
            results = json.load(f)
            
        return results
        
    finally:
        db.close()


async def process_drill_analysis(analysis_id: str, file_path: str, drill_type: str, user_id: str):
    """Background task to process drill analysis"""
    db = SessionLocal()
    
    try:
        # Update status to processing
        analysis = db.query(Video).filter(Video.id == analysis_id).first()
        if not analysis:
            logger.error(f"Analysis {analysis_id} not found in database")
            return
            
        analysis.status = ProcessingStatus.PROCESSING
        analysis.process_start_time = datetime.utcnow()
        db.commit()
        
        # Process the drill analysis
        logger.info(f"Starting {drill_type} analysis for {analysis_id}")
        
        try:
            results = get_processor().analyze_drill(file_path, drill_type, analysis_id)
        except Exception as e:
            logger.error(f"Drill analysis failed for {analysis_id}: {e}")
            raise
        
        # Save results
        results_path = PROCESSED_DIR / f"{analysis_id}_drill_analysis.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Update database with results
        analysis.status = ProcessingStatus.COMPLETED
        analysis.process_end_time = datetime.utcnow()
        analysis.results_path = str(results_path)
        
        # Extract key metrics for database storage
        analysis.total_touches = results["results"]["count_detected"]
        analysis.confidence_score = results["results"]["confidence"]
        
        db.commit()
        
        # Save drill attempt to Supabase
        try:
            create_drill_attempt(
                user_id=user_id,
                drill_type=drill_type,
                results=results,
                video_filename=Path(file_path).name
            )
            
            # Update user progress if benchmark was met
            if results["results"].get("benchmark_met", False):
                score = results["results"]["count_detected"]
                update_user_progress(user_id, drill_type, score)
            
            logger.info(f"Saved drill attempt to Supabase for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to save to Supabase: {e}")
            # Don't fail the entire analysis if Supabase save fails
        
        logger.info(f"Completed {drill_type} analysis for {analysis_id}")
        
    except Exception as e:
        logger.error(f"Error processing drill analysis {analysis_id}: {e}")
        analysis = db.query(Video).filter(Video.id == analysis_id).first()
        if analysis:
            analysis.status = ProcessingStatus.ERROR
            analysis.error_message = str(e)
            db.commit()
    finally:
        db.close()


@drill_router.get("/benchmark/{drill_type}")
async def get_drill_benchmark(drill_type: str):
    """Get benchmark criteria for a specific drill"""
    try:
        drill_enum = DrillType(drill_type)
        config = drill_registry.get_config(drill_enum)
        
        if not config:
            raise HTTPException(status_code=404, detail=f"Drill '{drill_type}' not found")
        
        return {
            "drill_type": drill_type,
            "name": config.name,
            "success_criteria": config.success_criteria,
            "time_window": config.time_window,
            "min_reps": config.min_reps,  
            "max_reps": config.max_reps,
            "per_foot": config.per_foot,
            "pattern_based": config.pattern_based
        }
        
    except ValueError:
        available_drills = [dt.value for dt in DrillType]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid drill type '{drill_type}'. Available: {available_drills}"
        )
    except Exception as e:
        logger.error(f"Error getting benchmark: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve benchmark")


# Convenience endpoint for frontend
@drill_router.get("/types")
async def get_drill_types():
    """Get simple list of drill type IDs for frontend dropdowns"""
    return {
        "drill_types": [dt.value for dt in DrillType],
        "count": len(list(DrillType))
    }