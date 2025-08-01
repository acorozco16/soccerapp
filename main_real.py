from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import json
import uuid
from datetime import datetime
import tempfile
import asyncio

# Import analyzers
from analyzers.juggling_analyzer import JugglingAnalyzer
from analyzers.bell_touches_analyzer import BellTouchesAnalyzer
from analyzers.inside_outside_analyzer import InsideOutsideAnalyzer
from analyzers.sole_rolls_analyzer import SoleRollsAnalyzer
from analyzers.outside_foot_push_analyzer import OutsideFootPushAnalyzer
from analyzers.v_cuts_analyzer import VCutsAnalyzer
from analyzers.croquetas_analyzer import CroquetasAnalyzer
from analyzers.triangles_analyzer import TrianglesAnalyzer

app = FastAPI(title="Soccer Training API - Real Analysis", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzers
ANALYZERS = {
    "juggling": JugglingAnalyzer(),
    "bell_touches": BellTouchesAnalyzer(),
    "inside_outside": InsideOutsideAnalyzer(),
    "sole_rolls": SoleRollsAnalyzer(),
    "outside_foot_push": OutsideFootPushAnalyzer(),
    "v_cuts": VCutsAnalyzer(),
    "croquetas": CroquetasAnalyzer(),
    "triangles": TrianglesAnalyzer()
}

# In-memory storage (replace with Supabase later)
drill_attempts = {}
user_progress = {}
analysis_jobs = {}  # Track ongoing analysis

# Models
class DrillAnalysis(BaseModel):
    drill_type: str
    total_touches: int
    duration: float
    touches_per_second: float
    confidence: float
    timestamp: str
    analysis_id: str
    user_id: str

# Mock auth
def get_current_user(authorization: Optional[str] = None):
    return {"id": "user_123", "email": "test@example.com", "full_name": "Test User"}

@app.get("/")
async def root():
    return {"message": "Soccer Training API - Real Analysis", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "soccer-training-api-real"}

# Auth endpoints (keep mock)
@app.post("/auth/login")
async def login():
    return {
        "access_token": "mock_token_123",
        "user": {
            "id": "user_123",
            "email": "test@example.com",
            "full_name": "Test User"
        }
    }

@app.post("/auth/register") 
async def register():
    return {
        "access_token": "mock_token_456",
        "user": {
            "id": "user_456",
            "email": "newuser@example.com",
            "full_name": "New User"
        }
    }

@app.get("/drill/available")
async def available_drills():
    return {
        "drills": [
            {"id": "juggling", "name": "Juggling", "description": "Keep the ball up"},
            {"id": "bell_touches", "name": "Bell Touches", "description": "Touch with inside of both feet"},
            {"id": "inside_outside", "name": "Inside-Outside", "description": "Touch with inside then outside"},
            {"id": "sole_rolls", "name": "Sole Rolls", "description": "Roll ball with sole of foot"},
            {"id": "outside_foot_push", "name": "Outside Foot Push", "description": "Push ball with outside of foot"},
            {"id": "v_cuts", "name": "V Cuts", "description": "Cut the ball in V motion"},
            {"id": "croquetas", "name": "Croquetas", "description": "Pull ball across body"},
            {"id": "triangles", "name": "Triangles", "description": "Move ball in triangle pattern"}
        ]
    }

# Process video in background
async def process_video_analysis(
    analysis_id: str,
    video_path: str,
    drill_type: str,
    user_id: str
):
    try:
        # Update status to processing
        analysis_jobs[analysis_id] = {
            "status": "processing",
            "progress": 0,
            "drill_type": drill_type
        }
        
        # Get the appropriate analyzer
        analyzer = ANALYZERS.get(drill_type)
        if not analyzer:
            raise ValueError(f"Unknown drill type: {drill_type}")
        
        # Run analysis (this takes 20-30 seconds)
        print(f"Starting analysis for {drill_type} - {analysis_id}")
        result = analyzer.analyze_video(video_path)
        print(f"Analysis complete for {analysis_id}: {result}")
        
        # Create analysis record
        analysis = DrillAnalysis(
            drill_type=drill_type,
            total_touches=result.get("total_ball_touches", 0),
            duration=result.get("total_frames", 900) / 30.0,  # Assume 30fps
            touches_per_second=result.get("total_ball_touches", 0) / (result.get("total_frames", 900) / 30.0),
            confidence=result.get("confidence_score", 0.0),
            timestamp=datetime.utcnow().isoformat(),
            analysis_id=analysis_id,
            user_id=user_id
        )
        
        # Store results
        if user_id not in drill_attempts:
            drill_attempts[user_id] = []
        drill_attempts[user_id].append(analysis.dict())
        
        # Update user progress
        update_user_progress(user_id, drill_type, result.get("total_ball_touches", 0))
        
        # Update job status
        analysis_jobs[analysis_id] = {
            "status": "completed",
            "progress": 100,
            "results": analysis.dict()
        }
        
    except Exception as e:
        print(f"Analysis failed for {analysis_id}: {str(e)}")
        analysis_jobs[analysis_id] = {
            "status": "failed",
            "error": str(e)
        }
    finally:
        # Clean up video file
        if os.path.exists(video_path):
            os.unlink(video_path)

@app.post("/drill/analyze")
async def analyze_drill(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    drill_type: str = "juggling",
    current_user: dict = Depends(get_current_user)
):
    """Upload video and start real analysis"""
    analysis_id = str(uuid.uuid4())
    
    # Validate drill type
    if drill_type not in ANALYZERS:
        raise HTTPException(status_code=400, detail=f"Invalid drill type: {drill_type}")
    
    # Save video temporarily
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, f"{analysis_id}.mp4")
    
    try:
        # Write video file
        content = await file.read()
        with open(video_path, "wb") as f:
            f.write(content)
        
        # Start background analysis
        background_tasks.add_task(
            process_video_analysis,
            analysis_id,
            video_path,
            drill_type,
            current_user["id"]
        )
        
        # Return immediate response
        analysis_jobs[analysis_id] = {
            "status": "processing",
            "progress": 0,
            "drill_type": drill_type
        }
        
        return {
            "analysis_id": analysis_id,
            "status": "processing",
            "message": "Video analysis started. Check status endpoint for progress."
        }
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(video_path):
            os.unlink(video_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drill/status/{analysis_id}")
async def get_drill_status(analysis_id: str):
    """Get real-time analysis status"""
    job = analysis_jobs.get(analysis_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return {
        "analysis_id": analysis_id,
        "status": job["status"],
        "progress": job.get("progress", 0),
        "drill_type": job.get("drill_type")
    }

@app.get("/drill/results/{analysis_id}")
async def get_drill_results(
    analysis_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get analysis results"""
    job = analysis_jobs.get(analysis_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Analysis status: {job['status']}")
    
    return job["results"]

def update_user_progress(user_id: str, drill_type: str, score: int):
    """Update user's personal best"""
    if user_id not in user_progress:
        user_progress[user_id] = {}
    
    if drill_type not in user_progress[user_id]:
        user_progress[user_id][drill_type] = {
            "personal_best": score,
            "total_attempts": 1,
            "last_attempt": datetime.utcnow().isoformat()
        }
    else:
        current = user_progress[user_id][drill_type]
        current["personal_best"] = max(current["personal_best"], score)
        current["total_attempts"] += 1
        current["last_attempt"] = datetime.utcnow().isoformat()

@app.get("/user/drill-history")
async def get_drill_history(
    drill_type: Optional[str] = None,
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """Get user's drill history"""
    user_attempts = drill_attempts.get(current_user["id"], [])
    
    if drill_type:
        user_attempts = [a for a in user_attempts if a["drill_type"] == drill_type]
    
    user_attempts.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return {
        "attempts": user_attempts[:limit],
        "total": len(user_attempts)
    }

@app.get("/user/stats")
async def get_user_stats(current_user: dict = Depends(get_current_user)):
    """Get user statistics"""
    user_attempts = drill_attempts.get(current_user["id"], [])
    user_pb = user_progress.get(current_user["id"], {})
    
    # Real calculations
    total_sessions = len(user_attempts)
    
    # Sessions this week
    from datetime import timedelta
    week_ago = datetime.utcnow() - timedelta(days=7)
    sessions_this_week = sum(
        1 for a in user_attempts 
        if datetime.fromisoformat(a["timestamp"]) > week_ago
    )
    
    # Calculate improvement
    overall_improvement = 0.0
    for drill_type, progress in user_pb.items():
        drill_specific = [a for a in user_attempts if a["drill_type"] == drill_type]
        if len(drill_specific) > 1:
            first = drill_specific[-1]["total_touches"]
            last = drill_specific[0]["total_touches"]
            if first > 0:
                improvement = ((last - first) / first) * 100
                overall_improvement += improvement
    
    if user_pb:
        overall_improvement /= len(user_pb)
    
    return {
        "total_sessions": total_sessions,
        "sessions_this_week": sessions_this_week,
        "overall_improvement": round(overall_improvement, 1),
        "drills": user_pb
    }