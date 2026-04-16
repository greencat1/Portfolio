# tests/test_auth.py
"""
Tests for API key authentication.
"""

import pytest
from fastapi import HTTPException
from app.auth import verify_api_key, require_admin, require_user



# app/tests/test_auth.py
import pytest
from fastapi import HTTPException
from app.auth import verify_api_key, require_admin, require_user
from app.core.database import get_db
from app.core.key_input import hash_key


@pytest.fixture(autouse=True)
def setup_test_keys():
    """Add test keys to database before tests"""
    with get_db() as conn:
        conn.execute('''
            INSERT OR REPLACE INTO api_keys (key_hash, name, role, rate_limit, is_active)
            VALUES (?, ?, ?, ?, ?)
        ''', (hash_key("test_user_key"), "test_user", "user", 100, 1))
        conn.commit()
    yield
    with get_db() as conn:
        conn.execute("DELETE FROM api_keys WHERE name LIKE 'test_%'")
        conn.commit()


def test_valid_api_key_works():
    """Valid key should pass authentication."""
    result = verify_api_key("test_user_key")
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