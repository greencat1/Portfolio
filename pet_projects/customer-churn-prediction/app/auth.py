# app/auth.py
"""
Authentication and Authorization Module

Provides API key-based authentication with role-based access control.
Supports two roles:
- USER: Can make predictions and update labels (data operations)
- ADMIN: Full access including model management and retraining

Security features:
- API key validation
- Role-based access control (RBAC)
- Rate limiting support
- Key revocation (deactivation)
- Secure key generation using secrets.token_hex()
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
# For production: use Redis, PostgreSQL, or AWS Secrets Manager
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
# Clients must include this header in every request
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


# =====================================================
# Core Authentication Functions
# =====================================================

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
        Dictionary containing key metadata (role, rate_limit, name, etc.)
        
    Raises:
        HTTPException 401: If key is missing, invalid, or deactivated
        
    Example:
        >>> auth_info = verify_api_key("user_7f3e8a2b...")
        >>> print(auth_info["role"])
        'user'
    """
    # Check if API key was provided in the request headers
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Please provide X-API-Key header",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    # Check if API key exists in our registry
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    key_info = API_KEYS[api_key]
    
    # Check if API key is still active (not revoked)
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
    
    Use this for endpoints that require ANY valid API key
    (both 'user' and 'admin' roles are allowed).
    
    Args:
        auth: The authentication dict from verify_api_key
        
    Returns:
        The auth dict if user has a valid key
        
    Raises:
        HTTPException 403: If user does not have required role
        
    Example:
        @app.post("/predict")
        def predict(data: dict, auth: dict = Depends(require_user)):
            return {"prediction": 0}
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
        
    Example:
        @app.post("/models/switch")
        def switch_model(request: dict, auth: dict = Depends(require_admin)):
            return {"status": "switched"}
    """
    if auth.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required for this endpoint"
        )
    return auth


# =====================================================
# API Key Management Functions (Admin Only)
# =====================================================

def generate_api_key(role: str = "user", name: str = None) -> str:
    """
    Generate a cryptographically secure random API key.
    
    Uses secrets.token_hex() which is suitable for security-sensitive applications.
    Generates 32-character hexadecimal string (16 bytes of randomness).
    
    Args:
        role: The role to assign ('user' or 'admin') - for logging purposes
        name: Optional display name for the key - for logging purposes
        
    Returns:
        A 32-character hexadecimal string (16 bytes)
        
    Example:
        >>> key = generate_api_key("user", "my_app")
        >>> print(len(key))
        32
    """
    return secrets.token_hex(16)


def create_api_key(role: str, name: str, rate_limit: int = None) -> Dict:
    """
    Create a new API key (admin only operation).
    
    This function generates a new API key and stores it in the API_KEYS registry.
    The full API key is returned only once - store it securely!
    
    Args:
        role: The role for the new key ('user' or 'admin')
        name: Display name for identifying the key
        rate_limit: Optional custom rate limit (defaults: 100 for user, 1000 for admin)
        
    Returns:
        Dictionary containing:
        - status: Operation status
        - api_key: The actual API key (store this securely!)
        - key_preview: Preview for display (first 8 + last 8 chars)
        - name: Display name
        - role: Assigned role
        - rate_limit: Rate limit per minute
        - created_at: Creation timestamp
        - message: Human-readable message
        
    Example:
        >>> result = create_api_key("user", "production_client", 500)
        >>> print(result["api_key"])
        'a1b2c3d4e5f67890abcdef1234567890'
    """
    # Generate secure random API key
    api_key = generate_api_key()
    
    # Set default rate limits based on role
    if rate_limit is None:
        rate_limit = 1000 if role == "admin" else 100
    
    # Store in memory (use database in production)
    API_KEYS[api_key] = {
        "name": name,
        "role": role,
        "rate_limit": rate_limit,
        "created_at": datetime.now().isoformat(),
        "is_active": True
    }
    
    # Return the actual API key (THIS IS THE IMPORTANT PART)
    #  The full key is returned only once!
    return {
        "status": "success",
        "api_key": api_key,           # ← FULL API KEY IS RETURNED HERE
        "key_preview": f"{api_key[:8]}...{api_key[-8:]}",
        "name": name,
        "role": role,
        "rate_limit": rate_limit,
        "created_at": API_KEYS[api_key]["created_at"],
        "message": f"API key created successfully for '{name}' with role '{role}'"
    }


def revoke_api_key(api_key: str, auth: dict) -> Dict:
    """
    Revoke (deactivate) an API key (admin only operation).
    
    Revoked keys cannot be used for authentication anymore.
    This is useful for:
    - Key rotation
    - Security incidents (compromised keys)
    - Offboarding clients
    
    Args:
        api_key: The API key to revoke
        auth: Authentication dict from verify_api_key (must be admin)
        
    Returns:
        Dictionary with operation status
        
    Note:
        The key is deactivated but remains in the registry for audit purposes.
        
    Example:
        >>> result = revoke_api_key("user_7f3e8a2b...", admin_auth)
        >>> print(result["status"])
        'success'
    """
    # Verify admin privileges
    if auth.get("role") != "admin":
        return {"status": "error", "message": "Admin privileges required"}
    
    # Check if API key exists
    if api_key not in API_KEYS:
        return {"status": "error", "message": "API key not found"}
    
    # Deactivate the key
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
        Dictionary containing:
        - status: Operation status
        - keys: List of key previews with metadata
        - total: Total number of keys
        
    Example:
        >>> result = list_api_keys(admin_auth)
        >>> for key in result["keys"]:
        ...     print(key["name"], key["role"])
        'default_user user'
        'admin admin'
    """
    # Verify admin privileges
    if auth.get("role") != "admin":
        return {"status": "error", "message": "Admin privileges required"}
    
    # Build list of key previews (never expose full keys!)
    keys = []
    for key, info in API_KEYS.items():
        keys.append({
            "key_preview": f"{key[:8]}...{key[-8:]}",  # Only show preview
            "name": info["name"],
            "role": info["role"],
            "rate_limit": info["rate_limit"],
            "created_at": info["created_at"],
            "is_active": info["is_active"]
        })
    
    return {
        "status": "success",
        "keys": keys,
        "total": len(keys)
    }


# =====================================================
# Helper Functions
# =====================================================

def get_key_info(api_key: str) -> Optional[Dict]:
    """
    Get information about a specific API key without authentication.
    Useful for debugging and internal checks.
    
    Args:
        api_key: The API key to look up
        
    Returns:
        Dictionary with key metadata or None if not found
    """
    return API_KEYS.get(api_key)


def is_key_active(api_key: str) -> bool:
    """
    Check if an API key is active (exists and not revoked).
    
    Args:
        api_key: The API key to check
        
    Returns:
        True if key exists and is active, False otherwise
    """
    key_info = API_KEYS.get(api_key)
    if key_info is None:
        return False
    return key_info.get("is_active", False)


def get_key_role(api_key: str) -> Optional[str]:
    """
    Get the role of an API key.
    
    Args:
        api_key: The API key to look up
        
    Returns:
        Role string ('user' or 'admin') or None if key not found
    """
    key_info = API_KEYS.get(api_key)
    if key_info is None:
        return None
    return key_info.get("role")