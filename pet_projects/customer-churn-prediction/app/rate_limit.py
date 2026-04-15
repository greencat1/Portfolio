# app/rate_limit.py
"""
Rate Limiting Module

Provides rate limiting functionality to prevent API abuse.
Tracks request counts per API key within a sliding time window.
"""

import time
from collections import defaultdict
from typing import Dict
from fastapi import HTTPException
from app.utils.logger import logger

# In-memory request log (use Redis for production deployments)
# Maps API key -> list of request timestamps
request_log: Dict[str, list] = defaultdict(list)


def check_rate_limit(api_key: str, rate_limit: int = None) -> bool:
    """
    Check if a request is within the rate limit for the given API key.
    
    Implements a sliding window rate limiter with a 60-second window.
    If the limit is exceeded, raises HTTP 429 (Too Many Requests).
    
    Args:
        api_key: The API key being used for the request
        rate_limit: Optional custom rate limit (uses key's default if not provided)
        
    Returns:
        True if request is allowed
        
    Raises:
        HTTPException 429: If rate limit is exceeded
    """
    if rate_limit is None:
        from app.auth import API_KEYS
        key_info = API_KEYS.get(api_key, {})
        rate_limit = key_info.get("rate_limit", 60)
    
    now = time.time()
    window = 60  # 1 minute sliding window
    
    # Clean up old requests outside the current window
    request_log[api_key] = [
        req_time for req_time in request_log[api_key]
        if now - req_time < window
    ]
    
    if len(request_log[api_key]) >= rate_limit:
        logger.warning(f"Rate limit exceeded for API key {api_key[:8]}...")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {rate_limit} requests per minute."
        )
    
    request_log[api_key].append(now)
    return True


def get_rate_limit_status(api_key: str) -> Dict:
    """
    Get current rate limit status for an API key.
    
    Returns information about how many requests have been made
    and how many are still available within the current window.
    
    Args:
        api_key: The API key to check
        
    Returns:
        Dictionary with current request count, limit, remaining, and reset time
    """
    now = time.time()
    window = 60
    
    # Clean up old requests
    from app.auth import API_KEYS
    request_log[api_key] = [
        req_time for req_time in request_log[api_key]
        if now - req_time < window
    ]
    
    key_info = API_KEYS.get(api_key, {})
    rate_limit = key_info.get("rate_limit", 60)
    
    reset_time = 0
    if request_log[api_key]:
        reset_time = window - (now - request_log[api_key][0])
        reset_time = max(0, int(reset_time))
    
    return {
        "current_requests": len(request_log[api_key]),
        "rate_limit": rate_limit,
        "remaining": max(0, rate_limit - len(request_log[api_key])),
        "reset_in_seconds": reset_time
    }