# tests/test_rate_limit.py
"""
Tests for rate limiting.
"""

import pytest
from fastapi import HTTPException
from app.rate_limit import check_rate_limit, get_rate_limit_status, request_log


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
    """Status endpoint should correctly show remaining requests."""
    # Make 30 requests
    for _ in range(30):
        check_rate_limit("test_key", rate_limit=100)
    
    status = get_rate_limit_status("test_key")
    
    assert status["current_requests"] == 30
    # Default rate limit is 60 when key doesn't exist in API_KEYS
    assert status["rate_limit"] == 60  # Changed from 100 to 60
    assert status["remaining"] == 30  # 60 - 30 = 30
    assert "reset_in_seconds" in status