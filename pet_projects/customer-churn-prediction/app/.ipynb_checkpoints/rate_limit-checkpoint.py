# app/auth.py
"""
Authentication and Authorization Module

Provides API key-based authentication with role-based access control.
Supports two roles:
- USER: Can make predictions and update labels (data operations)
- ADMIN: Full access including model management and retraining
"""

import secrets
from datetime import datetime
from typing import Dict, Optional
from fastapi import HTTPException, Security, status, Depends
from fastapi.security import APIKeyHeader

# =====================================================
# API Keys Configuration
# =====================================================

# In production, store these in a database or environment variables
# Currently stored in memory for demonstration purposes
API_KEYS: Dict[str, dict] = {
    # USER KEY - for data operations (predictions, labeling)
    "user_7f3e8a2b1c5d9e4f6a8b2c4d6e8f0a1b": {
        "name": "default_user",
        "role": "user",
        "rate_limit": 100,  # Maximum requests per minute
        "created_at": datetime.now().isoformat(),
        "is_active": True
    },
    # ADMIN KEY - for model management (retraining, switching, deleting)
    "admin_9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3": {
        "name": "admin",
        "role": "admin",
        "rate_limit": 1000,  # Higher limit for admin operations
        "created_at": datetime.now().isoformat(),
        "is_active": True
    }
}

# Header name for API key authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)) -> Dict:
    """
    Verify API key and return key information.
    
    This function is used as a dependency for protected endpoints.
    It validates:
    1. API key is present in the request headers
    2. API key exists in our registry
    3. API key is active (not revoked)
    
    Args:
        api_key: The API key extracted from the X-API-Key header
        
    Returns:
        Dictionary containing key metadata (role, rate_limit, etc.)
        
    Raises:
        HTTPException 401: If key is missing, invalid, or deactivated
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Please provide X-API-Key header",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    key_info = API_KEYS[api_key]
    
    if not key_info.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key is deactivated",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    return {"api_key": api_key, **key_info}


def require_user(auth: dict = Depends(verify_api_key)) -> Dict:
    """
    Dependency to enforce user role access.
    
    Use this for endpoints that require ANY valid API key.
    
    Args:
        auth: The authentication dict from verify_api_key
        
    Returns:
        The auth dict if user has a valid key
        
    Raises:
        HTTPException 403: If user does not have required role
    """
    if auth.get("role") not in ["user", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Valid API key required for this endpoint"
        )
    return auth


def require_admin(auth: dict = Depends(verify_api_key)) -> Dict:
    """
    Dependency to enforce admin role access.
    
    Use this dependency for endpoints that should only be accessible
    by users with admin privileges (model management, retraining, etc.)
    
    Args:
        auth: The authentication dict from verify_api_key
        
    Returns:
        The auth dict if user has admin role
        
    Raises:
        HTTPException 403: If user does not have admin role
    """
    if auth.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required for this endpoint"
        )
    return auth


def generate_api_key(role: str = "user", name: str = None) -> str:
    """
    Generate a cryptographically secure random API key.
    
    Uses secrets.token_hex() which is suitable for security-sensitive applications.
    
    Args:
        role: The role to assign ('user' or 'admin')
        name: Optional display name for the key
        
    Returns:
        A 32-character hexadecimal string (16 bytes)
    """
    return secrets.token_hex(16)


def create_api_key(role: str, name: str, rate_limit: int = None) -> Dict:
    """
    Create a new API key (admin only operation).
    
    This function generates a new API key and stores it in the API_KEYS registry.
    
    Args:
        role: The role for the new key ('user' or 'admin')
        name: Display name for identifying the key
        rate_limit: Optional custom rate limit (defaults: 100 for user, 1000 for admin)
        
    Returns:
        Dictionary containing the new API key and its metadata
    """
    api_key = generate_api_key()
    
    if rate_limit is None:
        rate_limit = 1000 if role == "admin" else 100
    
    API_KEYS[api_key] = {
        "name": name,
        "role": role,
        "rate_limit": rate_limit,
        "created_at": datetime.now().isoformat(),
        "is_active": True
    }
    
    return {
        "api_key": api_key,
        "name": name,
        "role": role,
        "rate_limit": rate_limit,
        "created_at": API_KEYS[api_key]["created_at"],
        "message": "API key created successfully"
    }


def revoke_api_key(api_key: str, auth: dict) -> Dict:
    """
    Revoke (deactivate) an API key (admin only operation).
    
    Revoked keys cannot be used for authentication anymore.
    This is useful for key rotation or when a key is compromised.
    
    Args:
        api_key: The API key to revoke
        auth: Authentication dict from verify_api_key (must be admin)
        
    Returns:
        Dictionary with operation status
        
    Note:
        The key is deactivated but remains in the registry for audit purposes.
    """
    if auth.get("role") != "admin":
        return {"status": "error", "message": "Admin privileges required"}
    
    if api_key not in API_KEYS:
        return {"status": "error", "message": "API key not found"}
    
    API_KEYS[api_key]["is_active"] = False
    return {
        "status": "success",
        "message": f"API key {api_key[:8]}... revoked",
        "api_key": api_key
    }


def list_api_keys(auth: dict) -> Dict:
    """
    List all registered API keys (admin only operation).
    
    Returns a preview of each key (only first and last 8 characters)
    along with metadata. Full keys are never exposed for security.
    
    Args:
        auth: Authentication dict from verify_api_key (must be admin)
        
    Returns:
        Dictionary containing list of key previews and total count
    """
    if auth.get("role") != "admin":
        return {"status": "error", "message": "Admin privileges required"}
    
    keys = []
    for key, info in API_KEYS.items():
        keys.append({
            "key_preview": f"{key[:8]}...{key[-8:]}",
            "name": info["name"],
            "role": info["role"],
            "rate_limit": info["rate_limit"],
            "created_at": info["created_at"],
            "is_active": info["is_active"]
        })
    return {"keys": keys, "total": len(keys)}