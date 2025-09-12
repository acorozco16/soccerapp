"""
Authentication API endpoints
Handles user registration, login, logout, and token management
"""

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from auth import (
    create_user_account, 
    authenticate_user, 
    refresh_user_token, 
    sign_out_user,
    get_current_user
)
import logging

logger = logging.getLogger(__name__)

# Create router for auth endpoints
auth_router = APIRouter(prefix="/auth", tags=["authentication"])

# Request/Response Models
class UserRegistration(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenRefresh(BaseModel):
    refresh_token: str

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    created_at: str
    
    @classmethod
    def model_validate(cls, obj):
        # Handle datetime conversion automatically
        if isinstance(obj, dict) and 'created_at' in obj:
            created_at = obj['created_at']
            if hasattr(created_at, 'isoformat'):
                obj = obj.copy()
                obj['created_at'] = created_at.isoformat()
        return super().model_validate(obj)

class AuthResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

class MessageResponse(BaseModel):
    message: str
    success: bool = True

@auth_router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserRegistration):
    """
    Register a new user account
    """
    try:
        result = create_user_account(
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name
        )
        
        if not result.get('session'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Account created but session not established. Please verify your email."
            )
        
        # Return raw dict to avoid Pydantic validation issues
        return {
            "access_token": result['session'].access_token,
            "refresh_token": result['session'].refresh_token,
            "token_type": "bearer",
            "expires_in": result['session'].expires_in,
            "user": {
                "id": result['user']['id'],
                "email": result['user']['email'],
                "full_name": result.get('profile', {}).get('full_name', user_data.email.split('@')[0]),
                "created_at": result['user']['created_at'] if isinstance(result['user']['created_at'], str) else str(result['user']['created_at'])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed. Please try again."
        )

@auth_router.post("/login")
async def login_user(login_data: UserLogin):
    """
    Authenticate user and return access token
    """
    logger.info("Login attempt started")
    try:
        logger.info("Calling authenticate_user")
        result = authenticate_user(
            email=login_data.email,
            password=login_data.password
        )
        logger.info("authenticate_user completed successfully")
        
        # Handle datetime conversion manually
        user_data = result['user']
        created_at_str = ""
        
        # Extract and format created_at safely
        if isinstance(user_data, dict):
            created_at = user_data.get('created_at')
        else:
            # If it's still a Supabase user object
            created_at = getattr(user_data, 'created_at', None)
            user_data = {
                'id': getattr(user_data, 'id', ''),
                'email': getattr(user_data, 'email', ''),
                'created_at': created_at
            }
        
        if created_at:
            if hasattr(created_at, 'isoformat'):
                created_at_str = created_at.isoformat()
            elif isinstance(created_at, str):
                created_at_str = created_at
            else:
                created_at_str = str(created_at)
        
        # Return raw dict to avoid Pydantic validation issues
        return {
            "access_token": result['access_token'],
            "refresh_token": result['session'].refresh_token,
            "token_type": "bearer",
            "expires_in": result['session'].expires_in,
            "user": {
                "id": user_data['id'],
                "email": user_data['email'],
                "full_name": result.get('profile', {}).get('full_name', login_data.email.split('@')[0]),
                "created_at": created_at_str
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Exception details: {str(e)}")
        
        # If it's a validation error, return more details
        if "validation error" in str(e).lower():
            logger.error("This is a Pydantic validation error - check our UserResponse usage")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@auth_router.post("/refresh", response_model=dict)
async def refresh_token(token_data: TokenRefresh):
    """
    Refresh access token using refresh token
    """
    try:
        result = refresh_user_token(token_data.refresh_token)
        
        return {
            "access_token": result['access_token'],
            "token_type": "bearer",
            "expires_in": result['session'].expires_in
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed. Please login again."
        )

@auth_router.post("/logout", response_model=MessageResponse)
async def logout_user(current_user: dict = Depends(get_current_user)):
    """
    Sign out current user
    """
    try:
        success = sign_out_user("")
        
        return MessageResponse(
            message="Successfully logged out" if success else "Logout completed with warnings",
            success=True
        )
        
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        return MessageResponse(
            message="Logout completed with errors",
            success=True  # Still return success as user is effectively logged out
        )

@auth_router.get("/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """
    Get current authenticated user information
    """
    try:
        # Handle datetime conversion for created_at
        created_at = current_user.get('created_at', '')
        if hasattr(created_at, 'isoformat'):
            created_at = created_at.isoformat()
        elif not isinstance(created_at, str):
            created_at = str(created_at) if created_at else ''
            
        return {
            "id": current_user['id'],
            "email": current_user['email'],
            "full_name": current_user.get('user_metadata', {}).get('full_name', current_user['email'].split('@')[0]),
            "created_at": created_at
        }
    except Exception as e:
        logger.error(f"Failed to get user info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user information"
        )

@auth_router.get("/status")
async def auth_status():
    """
    Check authentication system status
    """
    return {
        "status": "active",
        "message": "Authentication system is operational",
        "endpoints": [
            "POST /auth/register - Register new user",
            "POST /auth/login - User login",
            "POST /auth/refresh - Refresh token",
            "POST /auth/logout - User logout",
            "GET /auth/me - Get user info",
            "GET /auth/status - System status"
        ]
    }