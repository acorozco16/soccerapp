from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="Soccer Training API - Simple", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Soccer Training API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "soccer-training-api"}

# Mock auth endpoints for testing
@app.post("/auth/login")
async def login():
    return {
        "access_token": "mock_token_123",
        "user": {
            "id": "mock_user_123",
            "email": "test@example.com",
            "full_name": "Test User"
        }
    }

@app.post("/auth/register") 
async def register():
    return {
        "access_token": "mock_token_456",
        "user": {
            "id": "mock_user_456",
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

@app.post("/drill/analyze")
async def analyze_drill():
    return {
        "analysis_id": "mock_analysis_123",
        "status": "completed",
        "results": {
            "total_touches": 25,
            "duration": 30,
            "touches_per_second": 0.83,
            "confidence": 0.95
        }
    }