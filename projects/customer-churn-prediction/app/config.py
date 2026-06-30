# app/config.py
"""
Application Configuration Module

WHAT THIS MODULE DOES:
Loads all configuration settings from environment variables.
Centralizes all configurable parameters in one place.

WHY USE ENVIRONMENT VARIABLES?
- Security: API keys, passwords never hardcoded in code
- Flexibility: Different configs for dev/production without code changes
- Docker-friendly: Pass env vars via docker-compose.yml
- 12-factor app best practice

HOW IT WORKS:
1. Looks for .env file in app/ directory (local development)
2. Loads variables from .env into environment
3. Reads variables via os.getenv() with fallback defaults
4. Exports Settings instance for other modules to import

ENVIRONMENT VARIABLE PRECEDENCE (highest to lowest):
1. System environment variables (set in shell/Docker)
2. Variables from .env file
3. Default values in code (if neither exists)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ============================================================================
# LOAD .ENV FILE (LOCAL DEVELOPMENT ONLY)
# ============================================================================

# Construct path to .env file
# __file__ = current file path: /app/app/config.py
# .parent = /app/app
# .parent.parent = /app
# Then add /app/.env → /app/app/.env
# 
# WHY THIS PATH?
# The .env file should be in the same directory as the app code
# This allows docker-compose to mount it easily
env_path = Path(__file__).parent.parent / 'app/.env'
print(env_path)  # Debug output: shows where we're looking for .env

# Load environment variables from .env file
# Overrides= True means .env vars take precedence over system env vars
# This is useful for local development where you want custom settings
load_dotenv(dotenv_path=env_path)


# ============================================================================
# SETTINGS CLASS
# ============================================================================

class Settings:
    """
    Application settings container.
    
    HOW TO USE:
        from app.config import settings
        model_path = settings.churn_model_path
    
    WHY A CLASS INSTEAD OF DICT?
    - Type hints for IDE autocomplete
    - Default values in one place
    - Easy to add computed properties
    - Single source of truth
    
    ENVIRONMENT VARIABLES FORMAT:
    All variables are read via os.getenv(key, default)
    First argument: environment variable name
    Second argument: default value if not set
    """
    
    # ========================================================================
    # MODEL SETTINGS
    # ========================================================================
    
    # Path to the trained churn prediction model file
    # Environment variable: CHURN_MODEL_PATH
    # Default: "app/models/full_churn_pipeline_cloud.pkl"
    #
    # MODEL FILES:
    # - full_churn_pipeline_cloud.pkl: Original trained model
    # - full_churn_pipeline_retrained_cloud.pkl: Fully retrained model
    # - inc_churn_pipeline_retrained_cloud.pkl: Incrementally trained model
    #
    # WHY RELATIVE PATH?
    # Works both locally and in Docker container
    # In container, WORKDIR is /app, so path is correct
    churn_model_path: str = os.getenv(
        "CHURN_MODEL_PATH",
        "app/models/full_churn_pipeline_cloud.pkl"
    )
    
    
    # ========================================================================
    # RATE LIMITING SETTINGS
    # ========================================================================
    
    # Rate limit for regular user API keys (requests per minute)
    # Environment variable: RATE_LIMIT_USER
    # Default: 100 requests per minute
    #
    # WHY 100?
    # - Prevents abuse while allowing normal usage
    # - 100 requests/min = 1.6 requests/second
    # - Enough for most applications
    # - Can be increased for specific keys via admin API
    rate_limit_user: int = int(os.getenv("RATE_LIMIT_USER", "100"))
    
    # Rate limit for admin API keys (requests per minute)
    # Environment variable: RATE_LIMIT_ADMIN
    # Default: 1000 requests per minute
    #
    # WHY HIGHER FOR ADMIN?
    # - Admins may need to run batch operations
    # - Model retraining, key management require more requests
    # - Still has limit to prevent DDoS (even admins)
    rate_limit_admin: int = int(os.getenv("RATE_LIMIT_ADMIN", "1000"))
    
    
    # ========================================================================
    # API SETTINGS
    # ========================================================================
    
    # Header name for API key authentication
    # Environment variable: Not configurable (hardcoded)
    # Clients must send: X-API-Key: your_key_here
    #
    # WHY X-API-Key?
    # - Industry standard for API key authentication
    # - Not a cookie (stateless)
    # - Works with all HTTP clients
    api_key_name: str = "X-API-Key"
    
    
    # ========================================================================
    # APPLICATION METADATA
    # ========================================================================
    
    # Application name (used in Swagger UI title)
    app_name: str = "Churn Prediction API"
    
    # Application version (used in Swagger UI and health checks)
    # Version 2.0.0 indicates major rewrite with new features:
    # - Role-based access control
    # - Rate limiting
    # - Model management endpoints
    # - Dashboard support
    app_version: str = "2.0.0"
    
    # Application description (shown in Swagger UI)
    app_description: str = "API for customer churn prediction and labeling"


# ============================================================================
# SINGLE SETTINGS INSTANCE
# ============================================================================

# Create one settings instance to be imported everywhere
# This ensures all modules use the same configuration
# 
# IMPORT PATTERN:
# from app.config import settings  (not Settings class, but instance)
#
# WHY SINGLETON?
# - Configuration loaded once at startup
# - Consistent across all modules
# - No need to re-read environment variables
settings = Settings()


# ============================================================================
# ENVIRONMENT VARIABLES QUICK REFERENCE
# ============================================================================
# 
# | Variable Name        | Default | Description                    |
# |----------------------|---------|--------------------------------|
# | CHURN_MODEL_PATH     | app/models/full_churn_pipeline_cloud.pkl | Path to model file |
# | RATE_LIMIT_USER      | 100     | Requests/minute for user keys  |
# | RATE_LIMIT_ADMIN     | 1000    | Requests/minute for admin keys |
#
# ============================================================================
# .ENV FILE EXAMPLE
# ============================================================================
# 
# Create file at: app/.env
# 
# CHURN_MODEL_PATH=app/models/full_churn_pipeline_retrained_cloud.pkl
# RATE_LIMIT_USER=200
# RATE_LIMIT_ADMIN=2000
#
# ============================================================================
# DOCKER USAGE
# ============================================================================
# 
# In docker-compose.yml:
# 
# services:
#   api:
#     environment:
#       - CHURN_MODEL_PATH=app/models/full_churn_pipeline_cloud.pkl
#       - RATE_LIMIT_USER=100
#       - RATE_LIMIT_ADMIN=1000
#
# Or using .env file mounted in container:
# 
# volumes:
#   - ./app/.env:/app/app/.env
#
# ============================================================================