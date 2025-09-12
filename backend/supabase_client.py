"""
Supabase client configuration
Handles authentication and database connections
"""

import os
from supabase import create_client, Client
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Load environment variables (optional for deployment)
from dotenv import load_dotenv
try:
    load_dotenv()  # This will work locally but fail silently in production
except:
    pass  # Ignore if .env file doesn't exist (like in Railway)

# Supabase configuration (hardcoded for Railway deployment)
SUPABASE_URL = os.getenv("SUPABASE_URL") or "https://nxumfeldylzpqwqlvszz.supabase.co"
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY") or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im54dW1mZWxkeWx6cHF3cWx2c3p6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM5MTY1NDksImV4cCI6MjA2OTQ5MjU0OX0.D2WvA9Ld2YalWbum6qi5CBvXxmj75v1BuDb-NKrJkxo"

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def get_user_from_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify JWT token and return user info
    """
    try:
        # Remove 'Bearer ' prefix if present
        if token.startswith('Bearer '):
            token = token[7:]
        
        user = supabase.auth.get_user(token)
        return user.user if user.user else None
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        return None

def create_drill_attempt(user_id: str, drill_type: str, results: Dict[str, Any], 
                        video_filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Save a drill attempt to the database
    """
    try:
        data = {
            'user_id': user_id,
            'drill_type': drill_type,
            'results': results,
            'video_filename': video_filename
        }
        
        response = supabase.table('drill_attempts').insert(data).execute()
        return response.data[0] if response.data else {}
    except Exception as e:
        logger.error(f"Failed to create drill attempt: {e}")
        raise

def update_user_progress(user_id: str, drill_type: str, new_score: int) -> Dict[str, Any]:
    """
    Update user's progress for a specific drill
    """
    try:
        # First, try to get existing progress
        existing = supabase.table('user_progress')\
            .select('*')\
            .eq('user_id', user_id)\
            .eq('drill_type', drill_type)\
            .execute()
        
        if existing.data:
            # Update existing record
            current = existing.data[0]
            new_personal_best = max(current['personal_best'], new_score)
            new_total_attempts = current['total_attempts'] + 1
            
            response = supabase.table('user_progress')\
                .update({
                    'personal_best': new_personal_best,
                    'total_attempts': new_total_attempts,
                    'last_attempt_at': 'now()',
                    'updated_at': 'now()'
                })\
                .eq('user_id', user_id)\
                .eq('drill_type', drill_type)\
                .execute()
        else:
            # Create new record
            response = supabase.table('user_progress')\
                .insert({
                    'user_id': user_id,
                    'drill_type': drill_type,
                    'personal_best': new_score,
                    'total_attempts': 1,
                    'last_attempt_at': 'now()'
                })\
                .execute()
        
        return response.data[0] if response.data else {}
    except Exception as e:
        logger.error(f"Failed to update user progress: {e}")
        raise

def get_user_progress(user_id: str, drill_type: Optional[str] = None) -> list:
    """
    Get user's progress for all drills or specific drill
    """
    try:
        query = supabase.table('user_progress').select('*').eq('user_id', user_id)
        
        if drill_type:
            query = query.eq('drill_type', drill_type)
        
        response = query.execute()
        return response.data
    except Exception as e:
        logger.error(f"Failed to get user progress: {e}")
        return []

def get_user_attempts(user_id: str, drill_type: Optional[str] = None, limit: int = 10) -> list:
    """
    Get user's recent drill attempts
    """
    try:
        query = supabase.table('drill_attempts')\
            .select('*')\
            .eq('user_id', user_id)\
            .order('created_at', desc=True)\
            .limit(limit)
        
        if drill_type:
            query = query.eq('drill_type', drill_type)
        
        response = query.execute()
        return response.data
    except Exception as e:
        logger.error(f"Failed to get user attempts: {e}")
        return []