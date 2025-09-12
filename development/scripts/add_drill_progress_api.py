#!/usr/bin/env python3
"""
Add drill progress API endpoint to the backend
"""

def add_drill_progress_endpoint():
    """Add drill progress endpoint to main.py"""
    
    endpoint_code = '''
from datetime import datetime, timedelta
from typing import Optional

@app.get("/api/drills/{drill_type}/progress")
async def get_drill_progress(drill_type: str, user_id: Optional[str] = None):
    """Get drill progress data for the last 7 days"""
    try:
        # Use anonymous user if no user_id provided (consistent with current auth model)
        if not user_id:
            user_id = "anonymous"
        
        # Calculate date range for last 7 days
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=6)  # 7 days including today
        
        logger.info(f"Getting drill progress for {drill_type}, user: {user_id}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Query for last 7 days data
        last_7_days_query = f"""
        SELECT 
            DATE(created_at) as drill_date,
            MAX(CAST(results->>'count_detected' AS INTEGER)) as best_count
        FROM drill_attempts 
        WHERE user_id = %s 
            AND drill_type = %s
            AND DATE(created_at) >= %s
            AND DATE(created_at) <= %s
        GROUP BY DATE(created_at)
        ORDER BY drill_date ASC
        """
        
        last_7_days_result = await supabase_client.rpc('execute_sql', {
            'query': last_7_days_query,
            'params': [user_id, drill_type, start_date.isoformat(), end_date.isoformat()]
        })
        
        # Process last 7 days data
        last_7_days = []
        existing_dates = {}
        
        if last_7_days_result.data:
            for row in last_7_days_result.data:
                date_str = row['drill_date']
                existing_dates[date_str] = row['best_count'] or 0
        
        # Fill in all 7 days with 0 for missing dates
        for i in range(7):
            current_date = start_date + timedelta(days=i)
            date_str = current_date.isoformat()
            last_7_days.append({
                "date": date_str,
                "touches": existing_dates.get(date_str, 0)
            })
        
        # Query for personal best
        personal_best_query = f"""
        SELECT 
            MAX(CAST(results->>'count_detected' AS INTEGER)) as best_count,
            created_at
        FROM drill_attempts 
        WHERE user_id = %s 
            AND drill_type = %s
            AND CAST(results->>'count_detected' AS INTEGER) = (
                SELECT MAX(CAST(results->>'count_detected' AS INTEGER))
                FROM drill_attempts 
                WHERE user_id = %s AND drill_type = %s
            )
        ORDER BY created_at DESC 
        LIMIT 1
        """
        
        personal_best_result = await supabase_client.rpc('execute_sql', {
            'query': personal_best_query,
            'params': [user_id, drill_type, user_id, drill_type]
        })
        
        personal_best = None
        if personal_best_result.data and len(personal_best_result.data) > 0:
            best_data = personal_best_result.data[0]
            personal_best = {
                "touches": best_data['best_count'] or 0,
                "date": best_data['created_at']
            }
        
        # Query for recent sessions (last 5)
        recent_sessions_query = f"""
        SELECT 
            created_at,
            CAST(results->>'count_detected' AS INTEGER) as touches,
            CAST(results->>'duration' AS NUMERIC) as duration
        FROM drill_attempts 
        WHERE user_id = %s 
            AND drill_type = %s
        ORDER BY created_at DESC 
        LIMIT 5
        """
        
        recent_sessions_result = await supabase_client.rpc('execute_sql', {
            'query': recent_sessions_query,
            'params': [user_id, drill_type]
        })
        
        recent_sessions = []
        if recent_sessions_result.data:
            for session in recent_sessions_result.data:
                recent_sessions.append({
                    "date": session['created_at'],
                    "touches": session['touches'] or 0,
                    "duration": session['duration'] or 0
                })
        
        # Return structured response
        response_data = {
            "drill_type": drill_type,
            "last_7_days": last_7_days,
            "personal_best": personal_best,
            "recent_sessions": recent_sessions
        }
        
        logger.info(f"Drill progress response: {response_data}")
        return response_data
        
    except Exception as e:
        logger.error(f"Failed to get drill progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get progress data: {str(e)}")
'''
    
    # Read main.py
    try:
        with open('/root/soccerapp/backend/main.py', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ Could not find main.py file")
        return False
    
    # Check if endpoint already exists
    if '/api/drills/{drill_type}/progress' in content:
        print("✅ Drill progress endpoint already exists")
        return True
    
    # Add the endpoint before the main block
    if 'if __name__ == "__main__":' in content:
        content = content.replace(
            'if __name__ == "__main__":',
            endpoint_code + '\n\nif __name__ == "__main__":'
        )
    else:
        content += '\n' + endpoint_code
    
    # Write back
    try:
        with open('/root/soccerapp/backend/main.py', 'w') as f:
            f.write(content)
        print("✅ Added drill progress API endpoint")
        return True
    except Exception as e:
        print(f"❌ Failed to write to main.py: {e}")
        return False

def add_sql_execution_function():
    """Add helper function for SQL execution if it doesn't exist"""
    
    sql_helper_code = '''
# Helper function for direct SQL execution
async def execute_sql_query(query: str, params: list = None):
    """Execute raw SQL query with parameters"""
    try:
        # This is a simplified version - in production you'd want proper SQL execution
        # For now, we'll use Supabase client methods
        from supabase import create_client
        
        # Create a direct connection if needed
        # Note: This might need adjustment based on your Supabase setup
        return None
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        raise
'''
    
    try:
        with open('/root/soccerapp/backend/main.py', 'r') as f:
            content = f.read()
        
        if 'execute_sql_query' not in content:
            # Add helper function after imports
            import_section = content.find('\nfrom')
            if import_section != -1:
                # Find end of imports
                app_definition = content.find('\napp = FastAPI')
                if app_definition != -1:
                    content = content[:app_definition] + '\n' + sql_helper_code + content[app_definition:]
                    
                    with open('/root/soccerapp/backend/main.py', 'w') as f:
                        f.write(content)
                    print("✅ Added SQL helper function")
                    return True
        else:
            print("✅ SQL helper function already exists")
            return True
            
    except Exception as e:
        print(f"❌ Failed to add SQL helper: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Adding drill progress API endpoint...")
    
    success = add_drill_progress_endpoint()
    if success:
        print("✅ Drill progress API endpoint added successfully!")
        print("\nEndpoint: GET /api/drills/{drill_type}/progress")
        print("Returns: last_7_days, personal_best, recent_sessions")
        print("\nRestart the server to activate the new endpoint.")
    else:
        print("❌ Failed to add drill progress endpoint")