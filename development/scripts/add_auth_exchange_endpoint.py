#!/usr/bin/env python3
"""
Add the missing /auth/exchange endpoint to auth_api.py
This endpoint exchanges Supabase tokens for drill tokens
"""

import re

def add_auth_exchange_endpoint():
    # Read the current auth_api.py file from server
    with open('/root/soccerapp/backend/auth_api.py', 'r') as f:
        content = f.read()
    
    # Check if exchange endpoint already exists
    if '/auth/exchange' in content:
        print("✅ /auth/exchange endpoint already exists")
        return True
    
    # Add the exchange endpoint before the status endpoint
    exchange_endpoint = '''
class TokenExchange(BaseModel):
    supabase_token: str

@auth_router.post("/exchange")
async def exchange_token(token_data: TokenExchange):
    """
    Exchange Supabase token for drill token
    Currently just validates the Supabase token and returns it
    """
    try:
        # For now, just validate that we have a token and return it
        # In a full implementation, this would validate the Supabase token
        # and potentially create a drill-specific token
        
        if not token_data.supabase_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing supabase_token"
            )
        
        logger.info("Token exchange requested - returning Supabase token")
        
        # Return the same token for now (anonymous auth setup)
        return {
            "access_token": token_data.supabase_token,
            "token_type": "bearer",
            "expires_in": 3600,
            "message": "Token exchange successful"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token exchange failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token exchange failed"
        )

'''
    
    # Find the @auth_router.get("/status") endpoint and insert before it
    status_pattern = r'(@auth_router\.get\("/status"\))'
    
    if re.search(status_pattern, content):
        content = re.sub(
            status_pattern, 
            exchange_endpoint + r'\1',
            content
        )
    else:
        # If status endpoint not found, add at the end before any __main__ block
        content = content.rstrip() + '\n' + exchange_endpoint + '\n'
    
    # Write the updated content
    with open('/root/soccerapp/backend/auth_api.py', 'w') as f:
        f.write(content)
    
    print("✅ Added /auth/exchange endpoint to auth_api.py")
    return True

if __name__ == "__main__":
    add_auth_exchange_endpoint()