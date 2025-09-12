#!/usr/bin/env python3
"""
Manual drill progress endpoint code to add to main.py
Copy and paste this code into the server's main.py file
"""

# Add this endpoint to main.py before the if __name__ == "__main__": line

ENDPOINT_CODE = '''
@app.get("/api/drills/{drill_type}/progress")
async def get_drill_progress(drill_type: str):
    """Get drill progress data for the last 7 days"""
    try:
        from datetime import datetime, timedelta
        
        user_id = "anonymous"  # Using anonymous user consistent with current setup
        
        logger.info(f"Getting drill progress for {drill_type}")
        
        # Get all drill attempts for this drill type (last 30 days to be safe)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        
        response = supabase_client.table('drill_attempts') \\
            .select('*') \\
            .eq('user_id', user_id) \\
            .eq('drill_type', drill_type) \\
            .gte('created_at', thirty_days_ago) \\
            .order('created_at', desc=True) \\
            .execute()
        
        attempts = response.data if response.data else []
        logger.info(f"Found {len(attempts)} attempts for {drill_type}")
        
        # Calculate date range for last 7 days
        today = datetime.now().date()
        last_7_days = []
        
        for i in range(7):
            current_date = today - timedelta(days=6-i)  # Start from 6 days ago to today
            date_str = current_date.isoformat()
            
            # Find best score for this date
            day_attempts = [
                attempt for attempt in attempts 
                if datetime.fromisoformat(attempt['created_at'].replace('Z', '+00:00')).date() == current_date
            ]
            
            best_count = 0
            if day_attempts:
                scores = [
                    attempt['results'].get('count_detected', 0) 
                    for attempt in day_attempts 
                    if attempt['results']
                ]
                best_count = max(scores) if scores else 0
            
            last_7_days.append({
                "date": date_str,
                "touches": best_count
            })
        
        # Find personal best
        personal_best = None
        if attempts:
            all_scores = [
                {
                    'count': attempt['results'].get('count_detected', 0),
                    'date': attempt['created_at']
                }
                for attempt in attempts 
                if attempt['results'] and attempt['results'].get('count_detected')
            ]
            
            if all_scores:
                best_attempt = max(all_scores, key=lambda x: x['count'])
                personal_best = {
                    "touches": best_attempt['count'],
                    "date": best_attempt['date']
                }
        
        # Get recent sessions (last 5)
        recent_sessions = []
        for attempt in attempts[:5]:  # Already ordered by created_at desc
            if attempt['results']:
                recent_sessions.append({
                    "date": attempt['created_at'],
                    "touches": attempt['results'].get('count_detected', 0),
                    "duration": attempt['results'].get('duration', 0)
                })
        
        response_data = {
            "drill_type": drill_type,
            "last_7_days": last_7_days,
            "personal_best": personal_best,
            "recent_sessions": recent_sessions
        }
        
        logger.info(f"Returning progress data: {len(last_7_days)} days, {len(recent_sessions)} recent sessions")
        return response_data
        
    except Exception as e:
        logger.error(f"Failed to get drill progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get progress data: {str(e)}")
'''

print("="*60)
print("MANUAL ENDPOINT ADDITION")
print("="*60)
print()
print("1. SSH to server: ssh root@146.190.81.29")
print("2. Edit main.py: nano /root/soccerapp/backend/main.py")
print("3. Find the line: if __name__ == \"__main__\":")
print("4. Add the endpoint code BEFORE that line")
print("5. Save and restart server")
print()
print("ENDPOINT CODE TO ADD:")
print("="*60)
print(ENDPOINT_CODE)
print("="*60)
print()
print("After adding, restart with: python3 -m uvicorn main:app --host 0.0.0.0 --port 8000")