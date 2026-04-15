# tests/test_auth.py
"""
Tests for API key authentication.
"""

import pytest
from fastapi import HTTPException
from app.auth import verify_api_key, require_admin, require_user, API_KEYS


def test_valid_api_key_works():
    """Valid key should pass authentication."""
    valid_key = "user_7f3e8a2b1c5d9e4f6a8b2c4d6e8f0a1b"
    result = verify_api_key(valid_key)
    assert result["role"] == "user"


def test_missing_key_fails():
    """No API key should return 401 error."""
    with pytest.raises(HTTPException) as exc:
        verify_api_key(None)
    assert exc.value.status_code == 401


def test_wrong_key_fails():
    """Fake API key should return 401 error."""
    with pytest.raises(HTTPException) as exc:
        verify_api_key("fake_key_123")
    assert exc.value.status_code == 401


def test_user_can_access_user_endpoints():
    """User role should pass require_user check."""
    auth = {"role": "user"}
    result = require_user(auth)
    assert result == auth


def test_user_cannot_access_admin_endpoints():
    """User role should fail require_admin check."""
    auth = {"role": "user"}
    with pytest.raises(HTTPException) as exc:
        require_admin(auth)
    assert exc.value.status_code == 403