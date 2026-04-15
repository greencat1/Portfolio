# tests/test_integration.py
"""
Tests for complete API endpoints.
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

USER_KEY = "user_7f3e8a2b1c5d9e4f6a8b2c4d6e8f0a1b"
ADMIN_KEY = "admin_9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3"


def test_healthcheck():
    """Health check should work without authentication."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_without_key_fails():
    """Prediction endpoint requires API key."""
    response = client.post("/predict", json={"customerID": "test"})
    assert response.status_code == 401


def test_predict_with_valid_key_works():
    """Valid API key should access prediction endpoint."""
    headers = {"X-API-Key": USER_KEY}
    payload = {
        "customerID": "TEST-001",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "358.20"
    }
    
    response = client.post("/predict", json=payload, headers=headers)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_admin_endpoint_rejects_user():
    """Admin endpoints should reject regular users."""
    headers = {"X-API-Key": USER_KEY}
    response = client.get("/models", headers=headers)
    assert response.status_code == 403