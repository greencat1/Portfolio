# app/config.py
"""
Application Configuration

Loads configuration from environment variables.
For local development, use .env file.
For production, set environment variables directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (only for local development)
# This line looks for .env file in the parent directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class Settings:
    """Application settings loaded from environment variables"""
    
    # Model settings
    churn_model_path: str = os.getenv(
        "CHURN_MODEL_PATH",
        "app/models/full_churn_pipeline_cloud.pkl"
    )
    
    # API Keys (for authentication)
    # In production, generate secure random keys
    user_api_key: str = os.getenv(
        "USER_API_KEY",
        "user_7f3e8a2b1c5d9e4f6a8b2c4d6e8f0a1b"
    )
    
    admin_api_key: str = os.getenv(
        "ADMIN_API_KEY",
        "admin_9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3"
    )
    
    # Rate limiting settings
    rate_limit_user: int = int(os.getenv("RATE_LIMIT_USER", "100"))
    rate_limit_admin: int = int(os.getenv("RATE_LIMIT_ADMIN", "1000"))
    
    # API settings
    api_key_name: str = "X-API-Key"
    
    # Application info
    app_name: str = "Churn Prediction API"
    app_version: str = "2.0.0"
    app_description: str = "API for customer churn prediction and labeling"


# Create a single settings instance to import everywhere
settings = Settings()


