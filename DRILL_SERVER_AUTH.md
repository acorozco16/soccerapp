# Drill Server Authentication Setup

## Token Exchange Endpoint

Add this endpoint to your drill server to exchange Supabase tokens for drill server tokens:

### Python/FastAPI Implementation

```python
from fastapi import HTTPException, Depends
from pydantic import BaseModel
import jwt
import requests
from datetime import datetime, timedelta

class TokenExchange(BaseModel):
    supabase_token: str

# Your Supabase configuration
SUPABASE_URL = "https://nxumfeldylzpqwqlvszz.supabase.co"
SUPABASE_JWT_SECRET = "your-supabase-jwt-secret"  # Get from Supabase Dashboard -> Settings -> API
DRILL_SERVER_JWT_SECRET = "your-drill-server-secret"  # Your own secret

def verify_supabase_token(token: str):
    """Verify Supabase JWT token"""
    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated"
        )
        return payload
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid Supabase token: {str(e)}")

def create_drill_server_token(user_data: dict):
    """Create drill server JWT token"""
    payload = {
        "user_id": user_data.get("sub"),
        "email": user_data.get("email"),
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, DRILL_SERVER_JWT_SECRET, algorithm="HS256")

@app.post("/auth/exchange")
async def exchange_token(request: TokenExchange):
    """Exchange Supabase token for drill server token"""
    try:
        # Verify Supabase token
        supabase_user = verify_supabase_token(request.supabase_token)
        
        # Create drill server token
        drill_token = create_drill_server_token(supabase_user)
        
        return {
            "drill_token": drill_token,
            "expires_in": 86400  # 24 hours
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

def verify_drill_token(token: str):
    """Verify drill server JWT token"""
    try:
        payload = jwt.decode(token, DRILL_SERVER_JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid drill server token")

# Update your drill analysis endpoint to use drill server tokens
@app.post("/drill/analyze")
async def analyze_drill(
    file: UploadFile = File(...),
    drill_type: str = Form(...),
    current_user: dict = Depends(verify_drill_token)
):
    # Your existing drill analysis logic
    pass
```

### Alternative: Make Drill Server Accept Supabase Tokens Directly

If you prefer not to use token exchange, update your drill server's JWT verification:

```python
def verify_supabase_token_direct(token: str):
    """Verify Supabase token directly"""
    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,  # Use Supabase's secret
            algorithms=["HS256"],
            audience="authenticated"
        )
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/drill/analyze")
async def analyze_drill(
    file: UploadFile = File(...),
    drill_type: str = Form(...),
    current_user: dict = Depends(verify_supabase_token_direct)
):
    # Your existing drill analysis logic
    pass
```

## Getting Your Supabase JWT Secret

1. Go to https://supabase.com/dashboard/project/nxumfeldylzpqwqlvszz
2. Navigate to Settings -> API
3. Copy the "JWT Secret" value
4. Use this in your drill server configuration

## Testing

Once implemented, test with:

```bash
curl -X POST https://soccertrainingapp.org/auth/exchange \
  -H "Content-Type: application/json" \
  -d '{"supabase_token": "your-supabase-jwt-here"}'
```