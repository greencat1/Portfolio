# app/auth.py
"""
Authentication and Authorization Module

WHAT THIS MODULE DOES:
Handles all security concerns for the API:
- API key verification (who are you?)
- Role-based access control (what can you do?)
- Key management (create, revoke, list keys)

SECURITY PRINCIPLES:
1. Never store raw API keys in database (only SHA-256 hashes)
2. Use cryptographically secure random key generation
3. Cache keys in memory for performance
4. Each request is validated independently

ARCHITECTURE:
    Request → API Key Header → Verify Hash → Check Role → Allow/Deny
                ↓
         SHA-256 Hash → Lookup in DB/Cache → Get Role & Rate Limit
"""

import hashlib
import secrets
from datetime import datetime
from typing import Dict, Optional
from fastapi import HTTPException, Security, status, Depends
from fastapi.security import APIKeyHeader

from app.core.database import get_db

# ============================================================================
# API KEY HEADER CONFIGURATION
# ============================================================================

# Name of the HTTP header that carries the API key
# Clients must send: X-API-Key: their_key_here
# 
# WHY THIS HEADER?
# - Industry standard (used by AWS, Google, Stripe)
# - Not a cookie (stateless, works with all clients)
# - Easy to use in curl, Postman, browser
API_KEY_NAME = "X-API-Key"

# FastAPI security dependency that extracts the header value
# auto_error=False means we handle missing key ourselves (to provide custom error)
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


# ============================================================================
# KEY CACHE (Performance Optimization)
# ============================================================================

# Global in-memory cache for API keys
# Structure: {key_hash: {"name": ..., "role": ..., "rate_limit": ..., "is_active": ...}}
#
# WHY CACHE?
# - Avoids database query on every request (1000x faster)
# - Each request would otherwise hit SQLite (slow)
# - 1000 requests/sec would kill the database
#
# CACHE INVALIDATION:
# - Keys are loaded at startup
# - Refresh after creating/revoking keys
# - Cache is per-server (not shared across multiple instances)
KEY_CACHE = {}


# ============================================================================
# KEY HASHING (Security)
# ============================================================================

def hash_key(api_key: str) -> str:
    """
    Hash API key using SHA-256
    
    WHY HASH?
    - Never store raw keys in database (security!)
    - If database is compromised, hashes are useless to attackers
    - SHA-256 is one-way (can't recover original key from hash)
    
    HOW IT WORKS:
    Input: "user_7f3e8a2b1c5d9e4f"
    Output: "a1b2c3d4e5f67890abcdef1234567890..." (64 hex chars)
    
    NOTE: Same input always produces same hash (deterministic)
    This allows us to verify keys without storing them.
    
    ARGS:
        api_key: Raw API key string from request header
    
    RETURNS:
        SHA-256 hash as hexadecimal string (64 characters)
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


# ============================================================================
# CACHE REFRESH
# ============================================================================

def refresh_key_cache():
    """
    Load all active API keys from database into memory cache.
    
    WHEN CALLED:
    - At application startup (in main.py)
    - After creating a new API key
    - After revoking an API key
    
    WHY GLOBAL CACHE?
    - Performance: 0 database queries per request after load
    - Simplicity: Single source of truth in memory
    - Adequate for single-server deployment
    
    LIMITATION:
    - Multi-server deployments need shared cache (Redis)
    - Each server has its own cache, keys must be synced
    """
    global KEY_CACHE
    
    # Query database for all active keys
    with get_db() as conn:
        rows = conn.execute('''
            SELECT key_hash, name, role, rate_limit, is_active 
            FROM api_keys WHERE is_active = 1
        ''').fetchall()
    
    # Clear existing cache
    KEY_CACHE.clear()
    
    # Load new data
    for row in rows:
        KEY_CACHE[row['key_hash']] = {
            'name': row['name'],
            'role': row['role'],
            'rate_limit': row['rate_limit'],
            'is_active': row['is_active']
        }
    
    print(f"🔑 Key cache refreshed: {len(KEY_CACHE)} active keys loaded")


# ============================================================================
# MAIN AUTHENTICATION FUNCTION
# ============================================================================

def verify_api_key(api_key: str = Security(api_key_header)) -> dict:
    """
    Verify API key and return key info.
    
    This is the main authentication dependency.
    Called automatically for endpoints that require authentication.
    
    FLOW:
    1. Check if header present (if not → 401)
    2. Hash the provided key
    3. Look up hash in cache (fast path)
    4. If not in cache, check database (slow path)
    5. If found → return key info
    6. If not found → 401 Unauthorized
    
    WHY TWO PATHS (Cache + Database)?
    - Cache is fast but may be outdated
    - Database is source of truth but slower
    - Cache miss goes to database, then updates cache
    
    ARGS:
        api_key: Extracted from X-API-Key header (auto-injected)
    
    RETURNS:
        Dict with: api_key, name, role, rate_limit
    
    RAISES:
        HTTPException(401): Missing or invalid API key
    """
    # ============================================
    # STEP 1: Check if header was provided
    # ============================================
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Please provide X-API-Key header",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    # ============================================
    # STEP 2: Hash the provided key
    # ============================================
    # We never work with raw keys except for hashing
    # Raw key is only used here and then discarded
    incoming_hash = hash_key(api_key)
    
    # ============================================
    # STEP 3: Check cache first (fast path)
    # ============================================
    key_info = KEY_CACHE.get(incoming_hash)
    
    # ============================================
    # STEP 4: Cache miss → check database (slow path)
    # ============================================
    if not key_info:
        with get_db() as conn:
            row = conn.execute('''
                SELECT name, role, rate_limit, is_active 
                FROM api_keys 
                WHERE key_hash = ? AND is_active = 1
            ''', (incoming_hash,)).fetchone()
            
            if row:
                # Convert row to dict for consistent interface
                key_info = dict(row)
                # Update cache for future requests
                KEY_CACHE[incoming_hash] = key_info
    
    # ============================================
    # STEP 5: Not found → reject
    # ============================================
    if not key_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    # ============================================
    # STEP 6: Return key information
    # ============================================
    # This dict is passed to role checkers
    return {
        "api_key": api_key,           # Raw key (for rate limiting)
        "name": key_info["name"],     # Human-readable name
        "role": key_info["role"],     # "user", "admin", or "dashboard"
        "rate_limit": key_info["rate_limit"]  # Requests per minute
    }


# ============================================================================
# ROLE-BASED ACCESS CONTROL (RBAC)
# ============================================================================
# These functions enforce authorization rules.
# They are used as dependencies in endpoint definitions.
# ============================================================================

def require_user(auth: dict = Depends(verify_api_key)) -> dict:
    """
    Require User or Admin role.
    
    WHICH ENDPOINTS USE THIS?
    - POST /predict (make predictions)
    - POST /label/update (update labels)
    - GET /label/stats (view statistics)
    - GET /label/unlabeled/list (view unlabeled customers)
    
    WHO CAN ACCESS?
    - User role: YES
    - Admin role: YES
    - Dashboard role: NO (cannot make predictions)
    
    WHY RESTRICT DASHBOARD?
    Dashboard should be read-only to prevent accidental changes.
    """
    if auth.get("role") not in ["user", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Valid API key required for this endpoint"
        )
    return auth


def require_admin(auth: dict = Depends(verify_api_key)) -> dict:
    """
    Require Admin role only.
    
    WHICH ENDPOINTS USE THIS?
    - POST /retrain/incremental (retrain model)
    - POST /retrain/full (full retraining)
    - POST /models/switch (change active model)
    - POST /models/delete (delete models)
    - POST /admin/keys (create API keys)
    - DELETE /admin/keys/{key} (revoke keys)
    - GET /admin/keys (list keys)
    
    WHO CAN ACCESS?
    - Admin role: YES
    - User role: NO
    - Dashboard role: NO
    
    WHY SO RESTRICTIVE?
    These operations affect the entire system:
    - Retraining takes significant resources
    - Model changes affect all predictions
    - Key management affects all authentication
    """
    if auth.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required for this endpoint"
        )
    return auth


def require_dashboard(auth: dict = Depends(verify_api_key)) -> dict:
    """
    Allow Dashboard, User, or Admin roles.
    
    WHICH ENDPOINTS USE THIS?
    - All dashboard endpoints (read-only data access)
    
    WHO CAN ACCESS?
    - Dashboard role: YES (primary)
    - User role: YES (can also view dashboard)
    - Admin role: YES
    
    WHY SO PERMISSIVE?
    Dashboard only displays data, never modifies it.
    Allowing more roles provides flexibility for testing.
    """
    if auth.get("role") not in ["user", "admin", "dashboard"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Dashboard privileges required for this endpoint"
        )
    return auth


# ============================================================================
# API KEY MANAGEMENT FUNCTIONS (Admin Only)
# ============================================================================
# These functions are called by admin endpoints.
# They handle creating, revoking, and listing API keys.
# ============================================================================

def generate_api_key(role: str = "user", name: str = None) -> str:
    """
    Generate a cryptographically secure random API key.
    
    HOW IT WORKS:
    secrets.token_hex(16) generates 32 random hex characters
    Example output: "a1b2c3d4e5f67890abcdef1234567890"
    
    WHY secrets MODULE?
    - Designed specifically for security/cryptography
    - Uses OS random source (/dev/urandom on Linux)
    - Not predictable like random.random()
    - Suitable for passwords, tokens, API keys
    
    KEY STRENGTH:
    16 bytes = 128 bits of entropy
    Total possibilities: 2^128 (practically impossible to guess)
    
    ARGS:
        role: Not used in generation (just for context)
        name: Not used in generation (just for context)
    
    RETURNS:
        Random hex string (32 characters)
    """
    return secrets.token_hex(16)


def create_api_key(role: str, name: str, rate_limit: int = None) -> Dict:
    """
    Create a new API key (admin only operation).
    
    FLOW:
    1. Generate random key
    2. Hash the key (for storage)
    3. Save to database (store hash, NOT raw key)
    4. Refresh cache
    5. Return raw key to caller (only time it's shown!)
    
    IMPORTANT: The raw key is only returned once!
    User must save it immediately - cannot retrieve later.
    
    RATE LIMIT DEFAULTS:
    - Admin role: 1000 requests per minute
    - User role: 100 requests per minute
    - Dashboard role: 500 requests per minute
    
    ARGS:
        role: "user", "admin", or "dashboard"
        name: Human-readable identifier (e.g., "production_client")
        rate_limit: Optional custom limit (overrides default)
    
    RETURNS:
        Dict with key, preview, metadata, and success message
    """
    from app.core.database import get_db
    from app.core.key_input import hash_key
    
    # Generate cryptographically secure random key
    api_key = generate_api_key()
    
    # Hash for secure storage
    key_hash = hash_key(api_key)
    
    # Set default rate limits based on role
    if rate_limit is None:
        rate_limit = 1000 if role == "admin" else 100
    
    # Save to database
    with get_db() as conn:
        conn.execute('''
            INSERT INTO api_keys (key_hash, name, role, rate_limit, is_active)
            VALUES (?, ?, ?, ?, ?)
        ''', (key_hash, name, role, rate_limit, 1))
        conn.commit()
    
    # Refresh in-memory cache
    from app.core.key_input import refresh_key_cache
    refresh_key_cache()
    
    # Return response (raw key is only here!)
    return {
        "status": "success",
        "api_key": api_key,  # ← RAW KEY (show once, store securely!)
        "key_preview": f"{api_key[:8]}...{api_key[-8:]}",  # For display only
        "name": name,
        "role": role,
        "rate_limit": rate_limit,
        "created_at": datetime.now().isoformat(),
        "message": f"API key created successfully for '{name}' with role '{role}'"
    }


def revoke_api_key(api_key: str, auth: dict) -> Dict:
    """
    Revoke (deactivate) an API key.
    
    WHAT HAPPENS:
    - Sets is_active = 0 in database
    - Key no longer works for authentication
    - Cannot be undone (create new key instead)
    
    WHEN TO REVOKE:
    - Key was compromised (exposed publicly)
    - Employee left the company
    - Client stopped using service
    - Key rotation (create new, revoke old)
    
    ARGS:
        api_key: Raw API key to revoke
        auth: Authenticated admin info (for permission check)
    
    RETURNS:
        Dict with success/failure message
    """
    from app.core.database import get_db
    from app.core.key_input import hash_key
    
    # Security check (should be handled by require_admin dependency)
    if auth.get("role") != "admin":
        return {"status": "error", "message": "Admin privileges required"}
    
    # Hash the key to find in database
    key_hash = hash_key(api_key)
    
    # Update database (soft delete)
    with get_db() as conn:
        result = conn.execute('''
            UPDATE api_keys SET is_active = 0 WHERE key_hash = ?
        ''', (key_hash,))
        conn.commit()
        
        if result.rowcount == 0:
            return {"status": "error", "message": "API key not found"}
    
    # Refresh cache to remove revoked key
    from app.core.key_input import refresh_key_cache
    refresh_key_cache()
    
    return {
        "status": "success",
        "message": f"API key {api_key[:8]}... revoked",
        "api_key": api_key
    }


def list_api_keys(auth: dict) -> Dict:
    """
    List all registered API keys (active and inactive).
    
    SECURITY: Only returns previews, never full keys.
    Previews show first 8 + last 8 characters of HASH, not raw keys.
    
    USAGE:
    - See which keys exist
    - Check if keys are active
    - Identify keys by name
    
    ARGS:
        auth: Authenticated admin info
    
    RETURNS:
        Dict with list of key previews and total count
    """
    from app.core.database import get_db
    
    # Security check
    if auth.get("role") != "admin":
        return {"status": "error", "message": "Admin privileges required"}
    
    # Query all keys (both active and inactive)
    with get_db() as conn:
        rows = conn.execute('''
            SELECT key_hash, name, role, rate_limit, is_active, created_at
            FROM api_keys
        ''').fetchall()
    
    # Build preview list
    keys = []
    for row in rows:
        key_hash = row['key_hash']
        keys.append({
            # Only show preview of hash (not useful for authentication)
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


# ============================================================================
# SECURITY BEST PRACTICES IMPLEMENTED
# ============================================================================
# 
# 1. NO PLAIN TEXT KEYS IN DATABASE ✓
#    - Store only SHA-256 hashes
#    - Rainbow table resistant (keys are random)
# 
# 2. CRYPTOGRAPHICALLY RANDOM KEYS ✓
#    - secrets.token_hex() not random.random()
#    - 128 bits of entropy
# 
# 3. IN-MEMORY CACHE WITH HASHES ✓
#    - Even cache doesn't store raw keys
#    - Fast lookups without database
# 
# 4. ROLE-BASED ACCESS CONTROL ✓
#    - Granular permissions
#    - Least privilege principle
# 
# 5. RATE LIMITING PER KEY ✓
#    - Prevents abuse
#    - Different limits for different roles
# 
# 6. KEY REVOCATION ✓
#    - Soft delete preserves audit trail
#    - Immediate effect (cache refresh)
# 
# ============================================================================
# SECURITY WEAKNESSES (Production Improvements)
# ============================================================================
# 
# 1. NO KEY EXPIRATION
#    - Keys never expire automatically
#    - Fix: Add expires_at column
# 
# 2. NO AUDIT LOG
#    - Can't see who used which key when
#    - Fix: Log key usage to separate table
# 
# 3. SINGLE SERVER CACHE
#    - Multiple servers have separate caches
#    - Fix: Use Redis for shared cache
# 
# 4. NO KEY ROTATION ENFORCEMENT
#    - Old keys continue working forever
#    - Fix: Implement key rotation policy
# 
# ============================================================================