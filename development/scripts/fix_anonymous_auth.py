#!/usr/bin/env python3
"""
Fix authentication for anonymous users
Allow drill uploads without strict JWT validation for anonymous users
"""

import re

def fix_anonymous_authentication():
    """Fix authentication to allow anonymous users for drill uploads"""
    
    # Read the current main.py file
    with open('/root/soccerapp/backend/main.py', 'r') as f:
        content = f.read()
    
    # Check if we need to modify the drill analyze endpoint
    if 'current_user: dict = Depends(get_current_user)' in content:
        print("🔧 Making drill analyze endpoint anonymous-friendly...")
        
        # Replace strict authentication with optional authentication
        content = content.replace(
            'current_user: dict = Depends(get_current_user)',
            'current_user: dict = Depends(get_optional_user)'
        )
        
        # Also need to import get_optional_user if not already imported
        if 'get_optional_user' not in content:
            content = content.replace(
                'from auth import get_current_user',
                'from auth import get_current_user, get_optional_user'
            )
    
    # Write back the modified content
    with open('/root/soccerapp/backend/main.py', 'w') as f:
        f.write(content)
    
    print("✅ Updated main.py to use optional authentication")
    
    # Now update the drill analyze logic to handle anonymous users
    content = ''
    with open('/root/soccerapp/backend/main.py', 'r') as f:
        content = f.read()
    
    # Find the drill analyze endpoint and modify it
    if 'async def analyze_drill_video(' in content:
        # Replace user ID extraction logic
        old_pattern = r'user_id = current_user\[.user_id.\] if current_user else .anonymous.'
        new_pattern = 'user_id = current_user.get("user_id", "anonymous") if current_user else "anonymous"'
        
        if re.search(old_pattern, content):
            content = re.sub(old_pattern, new_pattern, content)
        else:
            # If the pattern doesn't exist, we need to add user_id handling
            analyze_pattern = r'(async def analyze_drill_video\(.*?\):.*?\n)(.*?)(try:)'
            
            def add_user_handling(match):
                func_def = match.group(1)
                existing_code = match.group(2)
                try_block = match.group(3)
                
                user_handling = '''    # Handle user identification (anonymous by default)
    user_id = current_user.get("user_id", "anonymous") if current_user else "anonymous"
    logger.info(f"Processing drill video for user: {user_id}")
    
    '''
                return func_def + existing_code + user_handling + try_block
            
            content = re.sub(analyze_pattern, add_user_handling, content, flags=re.DOTALL)
    
    # Write the final modified content
    with open('/root/soccerapp/backend/main.py', 'w') as f:
        f.write(content)
    
    print("✅ Updated drill analyze endpoint for anonymous authentication")
    
    # Also extend token expiration in auth exchange
    with open('/root/soccerapp/backend/auth_api.py', 'r') as f:
        auth_content = f.read()
    
    # Extend token expiration from 1 hour to 4 hours for video processing
    auth_content = auth_content.replace(
        '"expires_in": 3600',
        '"expires_in": 14400'  # 4 hours
    )
    
    with open('/root/soccerapp/backend/auth_api.py', 'w') as f:
        f.write(auth_content)
    
    print("✅ Extended token expiration to 4 hours")
    
    return True

if __name__ == "__main__":
    fix_anonymous_authentication()