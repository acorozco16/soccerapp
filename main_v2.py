from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import json
import uuid
from datetime import datetime
import aiofiles
import tempfile

app = FastAPI(title="Soccer Training API v2", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock database (in production, use Supabase)
# For now, store in memory
drill_attempts = {}
user_progress = {}

# Pydantic models
class DrillAnalysis(BaseModel):
    drill_type: str
    total_touches: int
    duration: float
    touches_per_second: float
    confidence: float
    timestamp: str
    analysis_id: str
    user_id: str

class UserStats(BaseModel):
    total_sessions: int
    sessions_this_week: int
    overall_improvement: float
    drills: Dict[str, Dict[str, Any]]

# Mock user authentication (simplified for now)
def get_current_user(authorization: Optional[str] = None):
    # In production, verify JWT token
    # For now, return mock user
    return {"id": "user_123", "email": "test@example.com", "full_name": "Test User"}

@app.get("/")
async def root():
    return {"message": "Soccer Training API v2", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "soccer-training-api-v2"}

# Auth endpoints (keep mock for now)
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

# Video upload and analysis endpoint
@app.post("/drill/analyze")
async def analyze_drill(
    file: UploadFile = File(...),
    drill_type: str = "juggling",
    current_user: dict = Depends(get_current_user)
):
    """
    Upload video and analyze drill performance
    """
    analysis_id = str(uuid.uuid4())
    
    # Create temporary file
    temp_file = None
    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        content = await file.read()
        await aiofiles.open(temp_file.name, 'wb').write(content)
        
        # Mock analysis results (in production, call real analyzer)
        # Simulate different results based on drill type
        mock_results = {
            "juggling": {"touches": 45, "confidence": 0.92},
            "bell_touches": {"touches": 32, "confidence": 0.88},
            "inside_outside": {"touches": 28, "confidence": 0.90},
            "sole_rolls": {"touches": 24, "confidence": 0.85},
            "outside_foot_push": {"touches": 30, "confidence": 0.87},
            "v_cuts": {"touches": 22, "confidence": 0.89},
            "croquetas": {"touches": 26, "confidence": 0.91},
            "triangles": {"touches": 20, "confidence": 0.86}
        }
        
        result = mock_results.get(drill_type, {"touches": 25, "confidence": 0.85})
        
        # Create analysis result
        analysis = DrillAnalysis(
            drill_type=drill_type,
            total_touches=result["touches"],
            duration=30.0,  # Mock 30 second video
            touches_per_second=result["touches"] / 30.0,
            confidence=result["confidence"],
            timestamp=datetime.utcnow().isoformat(),
            analysis_id=analysis_id,
            user_id=current_user["id"]
        )
        
        # Store in mock database
        if current_user["id"] not in drill_attempts:
            drill_attempts[current_user["id"]] = []
        drill_attempts[current_user["id"]].append(analysis.dict())
        
        # Update user progress
        update_user_progress(current_user["id"], drill_type, result["touches"])
        
        # Return immediate response
        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "results": analysis.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

def update_user_progress(user_id: str, drill_type: str, score: int):
    """Update user's personal best if applicable"""
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

@app.get("/drill/status/{analysis_id}")
async def get_drill_status(analysis_id: str):
    """Get status of drill analysis"""
    # In production, check real status
    # For now, always return completed
    return {
        "analysis_id": analysis_id,
        "status": "completed",
        "progress": 100
    }

@app.get("/drill/results/{analysis_id}")
async def get_drill_results(
    analysis_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get results of a specific drill analysis"""
    user_attempts = drill_attempts.get(current_user["id"], [])
    
    for attempt in user_attempts:
        if attempt["analysis_id"] == analysis_id:
            return attempt
    
    raise HTTPException(status_code=404, detail="Analysis not found")

@app.get("/user/drill-history")
async def get_drill_history(
    drill_type: Optional[str] = None,
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """Get user's drill attempt history"""
    user_attempts = drill_attempts.get(current_user["id"], [])
    
    # Filter by drill type if specified
    if drill_type:
        user_attempts = [a for a in user_attempts if a["drill_type"] == drill_type]
    
    # Sort by timestamp (newest first)
    user_attempts.sort(key=lambda x: x["timestamp"], reverse=True)
    
    # Limit results
    return {
        "attempts": user_attempts[:limit],
        "total": len(user_attempts)
    }

@app.get("/user/stats")
async def get_user_stats(current_user: dict = Depends(get_current_user)):
    """Get user statistics and progress"""
    user_attempts = drill_attempts.get(current_user["id"], [])
    user_pb = user_progress.get(current_user["id"], {})
    
    # Calculate stats
    total_sessions = len(user_attempts)
    
    # Sessions this week (mock calculation)
    sessions_this_week = min(total_sessions, 12)  # Mock data
    
    # Overall improvement (mock calculation)
    overall_improvement = 35.0 if total_sessions > 0 else 0.0
    
    # Drill-specific stats
    drill_stats = {}
    for drill_type, progress in user_pb.items():
        drill_stats[drill_type] = {
            "personal_best": progress["personal_best"],
            "total_attempts": progress["total_attempts"],
            "last_attempt": progress["last_attempt"]
        }
    
    return UserStats(
        total_sessions=total_sessions,
        sessions_this_week=sessions_this_week,
        overall_improvement=overall_improvement,
        drills=drill_stats
    ).dict()

@app.get("/user/progress/{drill_type}")
async def get_drill_progress(
    drill_type: str,
    current_user: dict = Depends(get_current_user)
):
    """Get progress for a specific drill"""
    progress = user_progress.get(current_user["id"], {}).get(drill_type)
    
    if not progress:
        return {
            "drill_type": drill_type,
            "personal_best": 0,
            "total_attempts": 0,
            "improvement": 0.0
        }
    
    # Get all attempts for this drill
    user_attempts = drill_attempts.get(current_user["id"], [])
    drill_attempts_list = [a for a in user_attempts if a["drill_type"] == drill_type]
    
    # Calculate improvement (compare first vs last attempt)
    improvement = 0.0
    if len(drill_attempts_list) > 1:
        first_score = drill_attempts_list[-1]["total_touches"]  # Oldest
        last_score = drill_attempts_list[0]["total_touches"]   # Newest
        if first_score > 0:
            improvement = ((last_score - first_score) / first_score) * 100
    
    return {
        "drill_type": drill_type,
        "personal_best": progress["personal_best"],
        "total_attempts": progress["total_attempts"],
        "improvement": improvement,
        "history": drill_attempts_list[:10]  # Last 10 attempts
    }