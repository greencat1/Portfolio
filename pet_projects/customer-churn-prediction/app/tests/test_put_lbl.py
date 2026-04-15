# tests/test_put_lbl.py
"""
Tests for label management.
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch
from app.scripts.put_lbl import update_label, get_label_statistics


@pytest.fixture
def temp_csv():
    """Create temporary CSV with test data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df = pd.DataFrame({
            'customerID': ['CUST-001', 'CUST-002'],
            'prediction': [0, 1],
            'probability': [0.2, 0.8],
            'Churn': [None, None]
        })
        df.to_csv(f.name, index=False)
        path = Path(f.name)
    
    with patch('app.scripts.put_lbl.Path') as mock:
        mock.return_value = path
        yield path
    
    path.unlink()


def test_add_label_to_customer(temp_csv):
    """Should successfully add label to existing customer."""
    result = update_label("CUST-001", "Yes")
    assert result["status"] == "success"
    assert result["new_label"] == "Yes"


def test_update_existing_label(temp_csv):
    """Should update label if customer already has one."""
    update_label("CUST-001", "Yes")
    result = update_label("CUST-001", "No")
    assert result["old_label"] == "Yes"
    assert result["new_label"] == "No"


def test_invalid_label_rejected():
    """Only 'Yes' or 'No' are allowed."""
    result = update_label("CUST-001", "Maybe")
    assert result["status"] == "error"


def test_nonexistent_customer_fails(temp_csv):
    """Updating customer not in database should fail."""
    result = update_label("FAKE-999", "Yes")
    assert result["status"] == "error"
    assert "not found" in result["message"]