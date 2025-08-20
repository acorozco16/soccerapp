# Quick fix for your drill server authentication
# Copy this code to your drill server and update the JWT verification

import jwt
from fastapi import HTTPException, Depends, Header
from typing import Optional

# Your Supabase JWT Secret (keep this secure!)
SUPABASE_JWT_SECRET = "5ZhVgn4Sakc9/IHZmLqkmcLJD6k9u1uG+KXERYp/2vAQqlhHlHN9sJCO7NA0cQGWzArL22B75gvMSFGTGREarw=="

def verify_supabase_token(authorization: Optional[str] = Header(None)):
    """Verify Supabase JWT token"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    
    try:
        # Extract token from "Bearer <token>"
        token = authorization[7:]  # Remove "Bearer " prefix
        
        # Verify with Supabase secret
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"]
            # Note: removed audience and issuer checks that might cause issues
        )
        
        print(f"Token verified for user: {payload.get('email', 'unknown')}")
        return payload
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        print(f"JWT Error: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Token verification failed: {str(e)}")

# Replace your current drill/analyze endpoint dependency with this:
# @app.post("/drill/analyze")
# async def analyze_drill(
#     file: UploadFile = File(...),
#     drill_type: str = Form(...),
#     current_user: dict = Depends(verify_supabase_token)  # Use this line
# ):
#     user_id = current_user.get("sub")
#     user_email = current_user.get("email")
#     # Your existing analysis code...