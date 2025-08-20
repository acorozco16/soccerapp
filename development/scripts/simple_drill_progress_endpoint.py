#!/usr/bin/env python3
"""
Simple drill progress API endpoint using direct Supabase client
"""

def add_simple_drill_progress_endpoint():
    """Add a simple drill progress endpoint that works with existing Supabase setup"""
    
    endpoint_code = '''
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
    
    # Read main.py from server or create a simple version
    print("📝 Creating drill progress endpoint...")
    
    # For now, let's create a standalone script that can be uploaded to the server
    server_script = f'''#!/usr/bin/env python3
"""
Add drill progress endpoint to main.py on the server
"""

def add_endpoint_to_main():
    # Read main.py
    with open('/root/soccerapp/backend/main.py', 'r') as f:
        content = f.read()
    
    # Check if endpoint already exists
    if '/api/drills/{{drill_type}}/progress' in content:
        print("✅ Drill progress endpoint already exists")
        return True
    
    # Add endpoint before main block
    endpoint_code = """{endpoint_code}"""
    
    if 'if __name__ == "__main__":' in content:
        content = content.replace(
            'if __name__ == "__main__":',
            endpoint_code + '\\n\\nif __name__ == "__main__":'
        )
    else:
        content += '\\n' + endpoint_code
    
    # Write back
    with open('/root/soccerapp/backend/main.py', 'w') as f:
        f.write(content)
    
    print("✅ Added drill progress API endpoint")
    return True

if __name__ == "__main__":
    add_endpoint_to_main()
'''
    
    # Save the server script
    with open('/Users/andreworozco/Desktop/server_add_progress_endpoint.py', 'w') as f:
        f.write(server_script)
    
    print("✅ Created server script: server_add_progress_endpoint.py")
    print("📤 Ready to upload to server and execute")
    
    return True

if __name__ == "__main__":
    print("🔧 Creating simple drill progress API endpoint...")
    add_simple_drill_progress_endpoint()
    print("✅ Script created! Upload server_add_progress_endpoint.py to server and run it.")