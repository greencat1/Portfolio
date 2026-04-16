# app/auth.py
import hashlib
import secrets
from datetime import datetime
from typing import Dict, Optional
from fastapi import HTTPException, Security, status, Depends
from fastapi.security import APIKeyHeader

from app.core.database import get_db

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Global cache for API keys (key_hash -> info)
KEY_CACHE = {}

def hash_key(api_key: str) -> str:
    """Hash API key using SHA-256"""
    return hashlib.sha256(api_key.encode()).hexdigest()

def refresh_key_cache():
    """Load keys from database into memory cache"""
    global KEY_CACHE
    with get_db() as conn:
        rows = conn.execute('''
            SELECT key_hash, name, role, rate_limit, is_active 
            FROM api_keys WHERE is_active = 1
        ''').fetchall()
    
    KEY_CACHE.clear()
    for row in rows:
        KEY_CACHE[row['key_hash']] = {
            'name': row['name'],
            'role': row['role'],
            'rate_limit': row['rate_limit'],
            'is_active': row['is_active']
        }
    print(f" Key cache refreshed: {len(KEY_CACHE)} active keys loaded")

def verify_api_key(api_key: str = Security(api_key_header)) -> dict:
    """Verify API key - compares hash with database"""
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Please provide X-API-Key header",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    incoming_hash = hash_key(api_key)
    
    # Check cache first
    key_info = KEY_CACHE.get(incoming_hash)
    
    # If not in cache, check database
    if not key_info:
        with get_db() as conn:
            row = conn.execute('''
                SELECT name, role, rate_limit, is_active 
                FROM api_keys 
                WHERE key_hash = ? AND is_active = 1
            ''', (incoming_hash,)).fetchone()
            
            if row:
                key_info = dict(row)
                KEY_CACHE[incoming_hash] = key_info
    
    if not key_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    return {
        "api_key": api_key,
        "name": key_info["name"],
        "role": key_info["role"],
        "rate_limit": key_info["rate_limit"]
    }

def require_user(auth: dict = Depends(verify_api_key)) -> dict:
    """User or admin access"""
    if auth.get("role") not in ["user", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Valid API key required for this endpoint"
        )
    return auth

def require_admin(auth: dict = Depends(verify_api_key)) -> dict:
    """Admin only access"""
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
    """Generate a cryptographically secure random API key"""
    return secrets.token_hex(16)

def create_api_key(role: str, name: str, rate_limit: int = None) -> Dict:
    """Create a new API key (admin only operation)"""
    from app.core.database import get_db
    from app.core.key_input import hash_key
    
    api_key = generate_api_key()
    key_hash = hash_key(api_key)
    
    if rate_limit is None:
        rate_limit = 1000 if role == "admin" else 100
    
    with get_db() as conn:
        conn.execute('''
            INSERT INTO api_keys (key_hash, name, role, rate_limit, is_active)
            VALUES (?, ?, ?, ?, ?)
        ''', (key_hash, name, role, rate_limit, 1))
        conn.commit()
    
    # Refresh cache
    from app.core.key_input import refresh_key_cache
    refresh_key_cache()
    
    return {
        "status": "success",
        "api_key": api_key,
        "key_preview": f"{api_key[:8]}...{api_key[-8:]}",
        "name": name,
        "role": role,
        "rate_limit": rate_limit,
        "created_at": datetime.now().isoformat(),
        "message": f"API key created successfully for '{name}' with role '{role}'"
    }

def revoke_api_key(api_key: str, auth: dict) -> Dict:
    """Revoke (deactivate) an API key (admin only operation)"""
    from app.core.database import get_db
    from app.core.key_input import hash_key
    
    if auth.get("role") != "admin":
        return {"status": "error", "message": "Admin privileges required"}
    
    key_hash = hash_key(api_key)
    
    with get_db() as conn:
        result = conn.execute('''
            UPDATE api_keys SET is_active = 0 WHERE key_hash = ?
        ''', (key_hash,))
        conn.commit()
        
        if result.rowcount == 0:
            return {"status": "error", "message": "API key not found"}
    
    # Refresh cache
    from app.core.key_input import refresh_key_cache
    refresh_key_cache()
    
    return {
        "status": "success",
        "message": f"API key {api_key[:8]}... revoked",
        "api_key": api_key
    }

def list_api_keys(auth: dict) -> Dict:
    """List all registered API keys (admin only operation)"""
    from app.core.database import get_db
    
    if auth.get("role") != "admin":
        return {"status": "error", "message": "Admin privileges required"}
    
    with get_db() as conn:
        rows = conn.execute('''
            SELECT key_hash, name, role, rate_limit, is_active, created_at
            FROM api_keys
        ''').fetchall()
    
    keys = []
    for row in rows:
        key_hash = row['key_hash']
        keys.append({
            "key_preview": f"{key_hash[:8]}...{key_hash[-8:]}",
            "name": row['name'],
            "role": row['role'],
            "rate_limit": row['rate_limit'],
            "created_at": row['created_at'],
            "is_active": bool(row['is_active'])
        })
    
    return {
        "status": "success",
        "keys": keys,
        "total": len(keys)
    }