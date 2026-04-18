# app/rate_limit.py
"""
Rate Limiting Module for API Protection

WHAT IS RATE LIMITING?
Prevents abuse by limiting how many requests a client can make in a time window.
Example: A single API key can only make 100 requests per minute.

WHY DO WE NEED IT?
- Prevent DDoS attacks (too many requests crashing the server)
- Fair usage across all users (no single user hogging resources)
- Reduce server load and costs
- Protect against brute-force attacks

HOW IT WORKS:
- Each API key has its own counter
- Counter tracks requests in a 60-second sliding window
- When limit exceeded → HTTP 429 (Too Many Requests)
- Counter resets automatically after 60 seconds

STORAGE:
Uses in-memory dictionary (request_log). Simple but:
- NOT persistent across server restarts (counters reset)
- Works for single-server deployment
- For multi-server, would need Redis or similar
"""

import time
from collections import defaultdict
from typing import Dict
from fastapi import HTTPException
from app.utils.logger import logger

# ============================================================================
# IN-MEMORY REQUEST STORAGE
# ============================================================================

# Dictionary storing request timestamps for each API key
# Structure: {"api_key_123": [timestamp1, timestamp2, timestamp3, ...]}
# Each timestamp is a float (seconds since epoch) of when a request was made
# 
# defaultdict(list) automatically creates empty list for new keys
# No need to check if key exists before appending
request_log: Dict[str, list] = defaultdict(list)


# ============================================================================
# RATE LIMIT LOOKUP
# ============================================================================

def get_rate_limit_for_key(api_key: str) -> int:
    """
    Retrieve the rate limit value for a specific API key
    
    HOW IT WORKS:
    1. Hashes the API key (for security, keys aren't stored in plain text)
    2. Looks up the key in KEY_CACHE (from auth module)
    3. Returns the rate_limit field (default 100 if not found)
    
    WHY HASH?
    We never store raw API keys in memory. Only hashes.
    This prevents key leakage in logs or debug dumps.
    
    ARGS:
        api_key: Raw API key string from request header
    
    RETURNS:
        int: Maximum requests per minute for this key
             Default is 100 if key not found in cache
    """
    from app.auth import KEY_CACHE, hash_key
    
    # Generate hash of the API key (same hashing as auth module)
    key_hash = hash_key(api_key)
    
    # Look up key info in cache (cached after database load)
    key_info = KEY_CACHE.get(key_hash, {})
    
    # Return rate_limit or default 100 (user default)
    # Admin keys typically have higher limits (1000)
    # Dashboard keys typically have 500
    return key_info.get('rate_limit', 100)


# ============================================================================
# CORE RATE LIMITING LOGIC
# ============================================================================

def check_rate_limit(api_key: str, rate_limit: int = None) -> bool:
    """
    Check if a request is within rate limit.
    Called BEFORE every API endpoint.
    
    RAISES HTTPException(429) if limit exceeded.
    
    ALGORITHM (Sliding Window):
    1. Get current time (now)
    2. Remove timestamps older than 60 seconds (sliding window)
    3. Count remaining timestamps (requests in last 60 seconds)
    4. If count >= rate_limit → REJECT (HTTP 429)
    5. If count < rate_limit → ALLOW, add current timestamp
    
    TIME COMPLEXITY: O(n) where n = requests in last minute
    Typically very small (n <= rate_limit, max 1000)
    
    EXAMPLE:
    Key has limit 100. In last 60 seconds, made 99 requests.
    - Remove old timestamps (older than 60s) → 99 remain
    - 99 < 100 → ALLOW, add new timestamp → now 100
    Next request:
    - 100 >= 100 → REJECT with HTTP 429
    
    ARGS:
        api_key: The API key from request headers
        rate_limit: Optional override (if None, auto-detected from key)
    
    RETURNS:
        bool: True if request allowed
    
    RAISES:
        HTTPException(429): Rate limit exceeded
    """
    # Determine rate limit for this key (if not provided)
    if rate_limit is None:
        rate_limit = get_rate_limit_for_key(api_key)
    
    # Current time in seconds since epoch (Unix timestamp)
    # Used to calculate 60-second windows
    now = time.time()
    
    # Window size in seconds (1 minute)
    window = 60
    
    # ============================================
    # STEP 1: Clean up old requests
    # ============================================
    # Remove timestamps older than 60 seconds
    # This implements the "sliding window"
    # Example: If now = 100.5, window = 60
    # Keep timestamps where 100.5 - timestamp < 60 → timestamp > 40.5
    request_log[api_key] = [
        req_time for req_time in request_log[api_key]
        if now - req_time < window
    ]
    
    # ============================================
    # STEP 2: Check if limit exceeded
    # ============================================
    if len(request_log[api_key]) >= rate_limit:
        # Log warning (useful for monitoring abuse attempts)
        logger.warning(f"Rate limit exceeded for API key {api_key[:8]}...")
        
        # Reject the request with HTTP 429 Too Many Requests
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {rate_limit} requests per minute."
        )
    
    # ============================================
    # STEP 3: Allow request and log it
    # ============================================
    # Add current timestamp to the log
    # This counts toward the next 60 seconds
    request_log[api_key].append(now)
    
    return True


# ============================================================================
# RATE LIMIT STATUS CHECK
# ============================================================================

def get_rate_limit_status(api_key: str) -> Dict:
    """
    Get current rate limit status for an API key.
    Used by GET /admin/rate-limit endpoint.
    
    WHAT IT RETURNS:
    - current_requests: How many requests made in last 60 seconds
    - rate_limit: Maximum allowed per minute
    - remaining: How many more allowed right now
    - reset_in_seconds: Seconds until oldest request expires
    
    WHY THIS IS USEFUL:
    - Clients can check their usage before sending batches
    - Debugging rate limit issues
    - Monitoring API usage
    
    EXAMPLE OUTPUT:
    {
        "current_requests": 25,
        "rate_limit": 100,
        "remaining": 75,
        "reset_in_seconds": 45
    }
    Means: 25 requests made, 75 remaining, resets in 45 seconds
    
    HOW reset_in_seconds CALCULATED:
    - Find oldest request in the window
    - Calculate: window - (now - oldest_timestamp)
    - This tells when that oldest request will expire
    - After expiration, the count decreases by 1
    
    ARGS:
        api_key: The API key to check
    
    RETURNS:
        Dict with current_requests, rate_limit, remaining, reset_in_seconds
    """
    # Current time
    now = time.time()
    window = 60
    
    # Clean up old requests first (same as check_rate_limit)
    request_log[api_key] = [
        req_time for req_time in request_log[api_key]
        if now - req_time < window
    ]
    
    # Get rate limit for this key
    rate_limit = get_rate_limit_for_key(api_key)
    
    # Calculate seconds until oldest request expires
    # If no requests, reset_time = 0 (resets immediately)
    reset_time = 0
    if request_log[api_key]:
        # Oldest request in the window (smallest timestamp)
        oldest_request = request_log[api_key][0]
        
        # Calculate when it will expire: window - (now - oldest)
        # Example: window=60, now=100, oldest=65 → 60 - (100-65) = 60-35 = 25 seconds
        reset_time = max(0, int(window - (now - oldest_request)))
    
    # Return status information
    return {
        "current_requests": len(request_log[api_key]),      # Requests in last 60s
        "rate_limit": rate_limit,                           # Max allowed
        "remaining": max(0, rate_limit - len(request_log[api_key])),  # Remaining
        "reset_in_seconds": reset_time                      # Seconds until reset
    }


