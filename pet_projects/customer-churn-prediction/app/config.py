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
env_path = Path(__file__).parent.parent / 'app/.env'
print(env_path)
load_dotenv(dotenv_path=env_path)


class Settings:
    """Application settings loaded from environment variables"""
    
    # Model settings
    churn_model_path: str = os.getenv(
        "CHURN_MODEL_PATH",
        "app/models/full_churn_pipeline_cloud.pkl"
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


