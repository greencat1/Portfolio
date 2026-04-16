# app/rate_limit.py
import time
from collections import defaultdict
from typing import Dict
from fastapi import HTTPException
from app.utils.logger import logger

request_log: Dict[str, list] = defaultdict(list)

def get_rate_limit_for_key(api_key: str) -> int:
    """Get rate limit from cached key info"""
    from app.auth import KEY_CACHE, hash_key
    key_hash = hash_key(api_key)
    key_info = KEY_CACHE.get(key_hash, {})
    return key_info.get('rate_limit', 100)

def check_rate_limit(api_key: str, rate_limit: int = None) -> bool:
    """Check if request is within rate limit"""
    if rate_limit is None:
        rate_limit = get_rate_limit_for_key(api_key)
    
    now = time.time()
    window = 60
    
    # Clean old requests
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
    """Get current rate limit status"""
    now = time.time()
    window = 60
    
    request_log[api_key] = [
        req_time for req_time in request_log[api_key]
        if now - req_time < window
    ]
    
    rate_limit = get_rate_limit_for_key(api_key)
    
    reset_time = 0
    if request_log[api_key]:
        reset_time = max(0, int(window - (now - request_log[api_key][0])))
    
    return {
        "current_requests": len(request_log[api_key]),
        "rate_limit": rate_limit,
        "remaining": max(0, rate_limit - len(request_log[api_key])),
        "reset_in_seconds": reset_time
    }