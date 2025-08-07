"""
Authentication middleware and endpoints for FastAPI
Integrates with Supabase for user management
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase_client import get_user_from_token, supabase
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Security scheme for JWT tokens
security = HTTPBearer()

class AuthenticationError(HTTPException):
    """Authentication-specific HTTP exception"""
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Dependency to get current authenticated user
    """
    if not credentials:
        raise AuthenticationError("No authentication token provided")
    
    user = get_user_from_token(credentials.credentials)
    if not user:
        raise AuthenticationError("Invalid or expired token")
    
    return user

async def get_optional_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[Dict[str, Any]]:
    """
    Dependency to get current user if token provided (optional)
    """
    if not credentials:
        return None
    
    return get_user_from_token(credentials.credentials)

def create_user_account(email: str, password: str, full_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a new user account in Supabase
    """
    try:
        # Create user in Supabase Auth
        auth_response = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        
        if auth_response.user:
            # Convert user object to dict and format datetime
            user_dict = {
                "id": auth_response.user.id,
                "email": auth_response.user.email,
                "created_at": auth_response.user.created_at.isoformat() if auth_response.user.created_at else ""
            }
            
            # Profile will be created automatically by database trigger
            return {
                "user": user_dict,
                "profile": None,  # Will be created by trigger
                "session": auth_response.session
            }
        else:
            raise Exception("User creation failed")
            
    except Exception as e:
        logger.error(f"User creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Account creation failed: {str(e)}"
        )

def authenticate_user(email: str, password: str) -> Dict[str, Any]:
    """
    Authenticate user with email and password
    """
    try:
        auth_response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        if auth_response.user and auth_response.session:
            # Get user profile
            profile = supabase.table('profiles')\
                .select('*')\
                .eq('id', auth_response.user.id)\
                .execute()
            
            # Convert user object to dict and format datetime
            user_dict = {
                "id": auth_response.user.id,
                "email": auth_response.user.email,
                "created_at": auth_response.user.created_at.isoformat() if auth_response.user.created_at else ""
            }
            
            logger.info(f"Converted user_dict: {user_dict}")
            logger.info(f"created_at type: {type(user_dict['created_at'])}")
            
            return {
                "user": user_dict,
                "profile": profile.data[0] if profile.data else None,
                "session": auth_response.session,
                "access_token": auth_response.session.access_token
            }
        else:
            raise Exception("Invalid credentials")
            
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise AuthenticationError("Invalid email or password")

def refresh_user_token(refresh_token: str) -> Dict[str, Any]:
    """
    Refresh user's access token
    """
    try:
        auth_response = supabase.auth.refresh_session(refresh_token)
        
        if auth_response.session:
            return {
                "session": auth_response.session,
                "access_token": auth_response.session.access_token
            }
        else:
            raise Exception("Token refresh failed")
            
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise AuthenticationError("Token refresh failed")

def sign_out_user(access_token: str) -> bool:
    """
    Sign out user and invalidate token
    """
    try:
        supabase.auth.sign_out()  # Signs out current session
        return True
    except Exception as e:
        logger.error(f"Sign out failed: {e}")
        return False