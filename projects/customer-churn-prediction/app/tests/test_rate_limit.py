# tests/test_rate_limit.py
"""
Tests for rate limiting.
"""

import pytest
from fastapi import HTTPException
from app.rate_limit import check_rate_limit, get_rate_limit_status, request_log
from app.core.database import get_db  
from app.core.key_input import hash_key  

def setup_function():
    """Clear request log before each test."""
    request_log.clear()


def test_requests_under_limit_work():
    """Making less than limit should succeed."""
    for _ in range(50):
        result = check_rate_limit("test_key", rate_limit=100)
        assert result is True


def test_exceeding_limit_fails():
    """Making more than limit should return 429 error."""
    for _ in range(100):
        check_rate_limit("test_key", rate_limit=100)
    
    with pytest.raises(HTTPException) as exc:
        check_rate_limit("test_key", rate_limit=100)
    assert exc.value.status_code == 429


def test_rate_limit_status_shows_remaining():
    
    with get_db() as conn:
        conn.execute('''
            INSERT OR REPLACE INTO api_keys (key_hash, name, role, rate_limit, is_active)
            VALUES (?, ?, ?, ?, ?)
        ''', (hash_key("rate_test_key"), "rate_test", "user", 100, 1))
        conn.commit()
    
    for _ in range(30):
        check_rate_limit("rate_test_key")
    
    status = get_rate_limit_status("rate_test_key")
    assert status["rate_limit"] == 100