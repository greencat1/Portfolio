# app/schemas.py
"""
Pydantic Schemas for Churn Prediction API

Defines request/response models for all API endpoints including:
- Prediction endpoints (single and batch)
- Label management endpoints
- Retraining endpoints
- Model management endpoints
- API key management endpoints
- Evaluation endpoints

WHAT ARE PYDANTIC SCHEMAS?
==========================
Pydantic schemas are like "data contracts" that define:
1. What data the API expects (request validation)
2. What data the API returns (response formatting)
3. Automatic validation (type checking, required fields)
4. Automatic documentation for Swagger/OpenAPI

WHY DO WE NEED THEM?
====================
- FastAPI uses them to validate incoming JSON
- They generate OpenAPI schema for Swagger UI
- They provide type hints and IDE autocomplete
- They prevent invalid data from reaching your code
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.utils.logger import logger


# ============================================================================
# PREDICTION SCHEMAS
# ============================================================================
# These schemas handle customer data for churn predictions
# ============================================================================

class PredictRequest(BaseModel):
    """
    Request model for single customer prediction
    
    WHAT IT DOES:
    Defines the structure of data needed to make ONE churn prediction.
    All 20 fields are required because the ML model needs them all.
    
    WHY 'object' TYPE?
    Fields like 'gender', 'Partner' are strings, but we use 'object'
    to handle both string and None/NaN values safely.
    
    HOW IT'S USED:
    POST /predict endpoint receives JSON → validates against this schema
    → passes to prediction function
    
    MAPPING TO DATASET:
    These 20 features match exactly the Telco Customer Churn dataset columns.
    """
    customerID: object      # Unique identifier for the customer (e.g., "7590-VHVEG")
    gender: object          # 'Male' or 'Female'
    SeniorCitizen: int      # 0 = No, 1 = Yes (senior citizen discount)
    Partner: object         # 'Yes' or 'No' - has a partner
    Dependents: object      # 'Yes' or 'No' - has dependents (children, elderly parents)
    tenure: int             # Number of months the customer has stayed with the company
    PhoneService: object    # 'Yes' or 'No' - has phone service
    MultipleLines: object   # 'Yes', 'No', or 'No phone service' - multiple phone lines
    InternetService: object # 'DSL', 'Fiber optic', or 'No'
    OnlineSecurity: object  # 'Yes', 'No', or 'No internet service'
    OnlineBackup: object    # 'Yes', 'No', or 'No internet service'
    DeviceProtection: object# 'Yes', 'No', or 'No internet service'
    TechSupport: object     # 'Yes', 'No', or 'No internet service'
    StreamingTV: object     # 'Yes', 'No', or 'No internet service'
    StreamingMovies: object # 'Yes', 'No', or 'No internet service'
    Contract: object        # 'Month-to-month', 'One year', or 'Two year'
    PaperlessBilling: object# 'Yes' or 'No' - electronic billing vs paper
    PaymentMethod: object   # 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    MonthlyCharges: float   # Monthly bill amount in USD
    TotalCharges: object    # Total amount charged over entire tenure (string, can be empty for new customers)

    class Config:
        """
        Pydantic configuration for schema
        json_schema_extra provides an example for Swagger documentation
        """
        json_schema_extra = {
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
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
                "TotalCharges": "29.85"
            }
        }


class PredictResponse(BaseModel):
    """
    Response model for single prediction
    
    WHAT IT RETURNS:
    - prediction: 0 = customer will STAY, 1 = customer will CHURN
    - probability: confidence score (0.0 to 1.0), e.g., 0.95 = 95% confident
    
    EXAMPLE:
    {"prediction": 1, "probability": 0.873} means 87.3% chance the customer will leave
    """
    prediction: int      # Binary classification: 0 (No Churn) or 1 (Churn)
    probability: float   # Raw probability from the ML model (0.0 to 1.0)


class BatchPredictRequest(BaseModel):
    """
    Request model for batch predictions
    
    WHAT IT DOES:
    Accepts a LIST of customers instead of just one.
    More efficient for processing multiple customers at once.
    
    WHY USE BATCH?
    - Single API call instead of N separate calls
    - Faster processing (model loaded once)
    - Reduces network overhead
    
    USAGE EXAMPLE:
    POST /predict/batch with 100 customers → returns 100 predictions
    """
    customers: List[PredictRequest]  # Array of individual prediction requests

    class Config:
        json_schema_extra = {
            "example": {
                "customers": [
                    {
                        "customerID": "7590-VHVEG",
                        "gender": "Female",
                        "SeniorCitizen": 0,
                        "Partner": "Yes",
                        "Dependents": "No",
                        "tenure": 1,
                        "PhoneService": "No",
                        "MultipleLines": "No phone service",
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
                        "TotalCharges": "29.85"
                    },
                    {
                        "customerID": "5575-GNVDE",
                        "gender": "Male",
                        "SeniorCitizen": 0,
                        "Partner": "No",
                        "Dependents": "No",
                        "tenure": 34,
                        "PhoneService": "Yes",
                        "MultipleLines": "No",
                        "InternetService": "DSL",
                        "OnlineSecurity": "Yes",
                        "OnlineBackup": "No",
                        "DeviceProtection": "Yes",
                        "TechSupport": "No",
                        "StreamingTV": "No",
                        "StreamingMovies": "No",
                        "Contract": "One year",
                        "PaperlessBilling": "No",
                        "PaymentMethod": "Mailed check",
                        "MonthlyCharges": 56.95,
                        "TotalCharges": "1889.5"
                    }
                ]
            }
        }


class BatchPredictResponse(BaseModel):
    """
    Response model for batch predictions
    
    RETURNS:
    - predictions: array of results for each customer (same order as request)
    - total: number of predictions made
    """
    predictions: List[PredictResponse]  # One PredictResponse per input customer
    total: int                           # Should equal len(predictions)


# ============================================================================
# LABEL MANAGEMENT SCHEMAS
# ============================================================================
# These schemas handle manual labeling of customer churn
# Labels are used for model retraining and evaluation
# ============================================================================

class LabelUpdateRequest(BaseModel):
    """
    Request model for updating a single customer label
    
    WHAT IS A LABEL?
    The "ground truth" - what actually happened to the customer.
    
    WHY DO WE NEED LABELS?
    - Model training: learns from labeled data
    - Model evaluation: compares predictions against actual outcomes
    - Retraining: improves model with new labeled data
    
    LABEL VALUES:
    - "Yes": Customer actually churned (left the company)
    - "No": Customer stayed (still with the company)
    
    NOTE: The model predicts Churn, humans verify and correct these labels.
    """
    customerID: str    # Which customer to label
    Churn: str         # "Yes" or "No" - the actual outcome
    
    class Config:
        json_schema_extra = {
            "example": {
                "customerID": "7590-VHVEG",
                "Churn": "Yes"
            }
        }


class LabelUpdateResponse(BaseModel):
    """
    Response model for label update operation
    
    Shows what changed and when.
    Used to confirm the label was successfully saved.
    """
    status: str                      # "success" or "error"
    message: str                     # Human-readable description
    customerID: str                  # Which customer was updated
    old_label: Optional[str] = None  # Previous label (if any)
    new_label: Optional[str] = None  # New label that was set
    timestamp: Optional[str] = None  # When the update occurred


class BatchLabelUpdateRequest(BaseModel):
    """
    Request model for batch label updates
    
    Update multiple customer labels in a single API call.
    More efficient than individual updates.
    
    USAGE SCENARIO:
    After reviewing a list of customers, mark 50 of them as churned at once.
    """
    updates: List[LabelUpdateRequest]  # Array of label updates
    
    class Config:
        json_schema_extra = {
            "example": {
                "updates": [
                    {"customerID": "7590-VHVEG", "Churn": "Yes"},
                    {"customerID": "5575-GNVDE", "Churn": "No"}
                ]
            }
        }


class BatchLabelUpdateResponse(BaseModel):
    """
    Response model for batch label update operation
    
    Provides summary statistics and individual results.
    Useful for debugging which updates failed.
    """
    status: str                # "completed" or "error"
    total_updates: int         # How many updates were attempted
    successful: int            # How many succeeded
    failed: int                # How many failed
    results: List[dict]        # Detailed results for each update


class LabelInfoResponse(BaseModel):
    """
    Response model for customer label information
    
    Returns BOTH:
    - Model's prediction (what the AI thinks)
    - Actual label (what really happened, if known)
    
    This allows comparison between predictions and reality.
    """
    status: str                         # "success" or "error"
    customerID: str                     # Customer identifier
    Churn: Optional[str] = None         # Actual label ("Yes"/"No" or None if not labeled)
    prediction: Optional[int] = None    # Model's prediction (0/1)
    probability: Optional[float] = None # Model's confidence
    timestamp: Optional[str] = None     # When prediction was made
    label_timestamp: Optional[str] = None # When label was added/updated


class LabelStatisticsResponse(BaseModel):
    """
    Response model for labeling statistics
    
    Shows how much labeled data is available for retraining.
    CRITICAL for understanding when you can retrain the model.
    
    WHY THESE NUMBERS MATTER:
    - Need enough labeled data to retrain (typically 100+ samples)
    - Imbalanced data (too few "Yes") may need special handling
    - Progress shows how much work remains
    """
    status: str                      # "success" or "error"
    total_records: int               # All customers in database
    labeled_records: int             # Customers with Churn label (Yes or No)
    unlabeled_records: int           # Customers without label (calculated)
    churn_yes: int                   # Labeled as "Yes" (actually churned)
    churn_no: int                    # Labeled as "No" (stayed)
    labeling_progress: str           # Human-readable: "150/1000 (15.0%)"


# ============================================================================
# RETRAINING SCHEMAS
# ============================================================================
# These schemas handle model retraining operations
# ============================================================================

class RetrainResponse(BaseModel):
    """
    Response schema for retraining endpoints
    
    Returns comprehensive information about the retraining operation:
    - How many samples were used
    - Performance metrics of the new model
    - Where the model was saved
    
    USED BY:
    POST /retrain/incremental - continue training from existing model
    POST /retrain/full - train completely new model from scratch
    """
    status: str                           # "success" or "error"
    message: str                          # Human-readable status
    samples_used: Optional[int] = None    # Total samples used for training
    old_samples: Optional[int] = None     # Previous training samples count
    new_labeled_samples: Optional[int] = None # Newly labeled samples added
    samples_attempted: Optional[int] = None # Samples tried (for incremental)
    churn_yes: Optional[int] = None       # Number of churn=Yes in training
    churn_no: Optional[int] = None        # Number of churn=No in training
    metrics: Optional[Dict[str, float]] = None # Performance: accuracy, recall, etc.
    model_path: Optional[str] = None      # Where the model was saved
    timestamp: Optional[str] = None       # When retraining happened


class RetrainStatusResponse(BaseModel):
    """
    Response schema for retraining status endpoint
    
    Checks if you have enough labeled data to retrain.
    
    RETRAINING REQUIREMENTS:
    - Minimum 10 labeled samples (both Yes and No)
    - More is better (100+ recommended)
    
    USED BY:
    GET /retrain/status - called before attempting retraining
    """
    status: str
    message: str                         # Tells you if you can retrain
    total_records: Optional[int] = None
    labeled_records: Optional[int] = None
    unlabeled_records: Optional[int] = None
    churn_yes: Optional[int] = None
    churn_no: Optional[int] = None
    can_retrain: Optional[bool] = None   # True if enough data available


# ============================================================================
# MODEL MANAGEMENT SCHEMAS
# ============================================================================
# These schemas handle model versioning and switching
# ============================================================================

class ModelInfo(BaseModel):
    """
    Information about a single model
    
    Each trained model is saved as a .pkl file.
    This schema provides metadata about each model.
    
    MODEL NAMING CONVENTION:
    - full_churn_pipeline_cloud.pkl - original trained model
    - full_churn_pipeline_retrained_cloud.pkl - fully retrained model
    - inc_churn_pipeline_retrained_cloud.pkl - incrementally trained model
    - inc_churn_pipeline_backup_cloud.pkl - backup of previous model
    """
    name: str                           # Filename of the model
    path: str                           # Full filesystem path
    size_mb: float                      # File size in megabytes
    created_at: str                     # ISO timestamp of creation
    is_active: bool                     # Is this model currently used for predictions?
    metrics: Optional[Dict[str, Any]] = None # Performance metrics if available
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "full_churn_pipeline_retrained_cloud.pkl",
                "path": "models/full_churn_pipeline_retrained_cloud.pkl",
                "size_mb": 45.67,
                "created_at": "2026-04-14T14:31:00",
                "is_active": True,
                "metrics": {
                    "samples_used": 7043,
                    "churn_yes": 1869,
                    "churn_no": 5174,
                    "accuracy": 0.8123
                }
            }
        }


class ModelsListResponse(BaseModel):
    """
    Response for listing all models
    
    Returns every model in the models directory.
    Shows which one is currently active.
    """
    status: str
    active_model: str                   # Name of currently active model
    active_model_path: str              # Full path to active model
    models: List[ModelInfo]             # All available models
    total_models: int                   # Count of models


class SwitchModelRequest(BaseModel):
    """
    Request to switch active model
    
    Changing the active model affects ALL subsequent predictions.
    Old predictions are not affected.
    
    WHY SWITCH MODELS?
    - New model performed better on evaluation
    - Rollback to previous model if new one has issues
    - A/B testing different model versions
    """
    model_name: str                     # Filename of model to activate
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "full_churn_pipeline_retrained_cloud.pkl"
            }
        }


class SwitchModelResponse(BaseModel):
    """
    Response after switching model
    
    Confirms the switch and shows what changed.
    """
    status: str
    message: str
    previous_model: str                 # What was active before
    current_model: str                  # What is active now
    model_path: str                     # Path to the new active model


class DeleteModelRequest(BaseModel):
    """
    Request to delete a model
    
    Permanently removes a model file from disk.
    Cannot be undone.
    
    PROTECTED MODELS:
    - The active model cannot be deleted unless force=True
    - Some models may be marked as protected
    """
    model_name: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "full_churn_pipeline_backup_cloud.pkl"
            }
        }


class DeleteModelResponse(BaseModel):
    """Response after deleting model"""
    status: str
    message: str
    deleted_model: str                  # Which model was deleted


class CompareModelsResponse(BaseModel):
    """
    Response for comparing two models
    
    Shows side-by-side comparison of:
    - File sizes
    - Creation dates
    - Performance metrics
    
    USED FOR:
    Deciding which model to make active
    """
    status: str
    model1: Dict[str, Any]              # First model's info
    model2: Dict[str, Any]              # Second model's info
    comparison: Dict[str, Any]          # Differences between them


class ModelMetricsResponse(BaseModel):
    """
    Response for model metrics
    
    Returns performance metrics calculated on test data.
    
    KEY METRICS:
    - accuracy: Overall correctness
    - precision: Of predicted churns, how many actually churned
    - recall: Of actual churns, how many were found
    - f1_score: Harmonic mean of precision and recall
    - roc_auc: Model's ability to separate classes
    """
    status: str
    model_name: str
    metrics: Dict[str, Any]             # Performance metrics


# ============================================================================
# API KEY MANAGEMENT SCHEMAS
# ============================================================================
# These schemas handle API key creation and management
# ============================================================================

class APIKeyCreateRequest(BaseModel):
    """
    Request model for creating a new API key
    
    API keys authenticate clients. Each key has:
    - role: determines access level (user, admin, dashboard)
    - name: human-readable identifier
    - rate_limit: max requests per minute
    
    ROLE PERMISSIONS:
    - user: can make predictions, update labels
    - admin: everything + model management, key management
    - dashboard: read-only access for the Streamlit dashboard
    """
    role: str                           # "user", "admin", or "dashboard"
    name: str                           # Display name (e.g., "production_client")
    rate_limit: Optional[int] = None    # Requests per minute (default: 100 for user, 1000 for admin)
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "name": "production_client",
                "rate_limit": 500
            }
        }


class APIKeyCreateResponse(BaseModel):
    """
    Response model for API key creation
    
    IMPORTANT: The full API key is returned only ONCE!
    Store it securely - you cannot retrieve it again.
    
    The key is a random string that must be included in X-API-Key headers.
    """
    status: str
    api_key: str                        # The actual key (store this!)
    name: str
    role: str
    rate_limit: int
    created_at: str
    message: str


class APIKeyInfo(BaseModel):
    """
    Information about an API key (without exposing the full key)
    
    For security, only shows preview (first 8 and last 8 chars).
    Example: "user_7f3e...f0a1b" shows "user_7f3e" and "...f0a1b"
    """
    key_preview: str                    # Partial key for identification
    name: str
    role: str
    rate_limit: int
    created_at: str
    is_active: bool                     # Can this key still be used?


class APIKeyListResponse(BaseModel):
    """Response model for listing API keys"""
    status: str
    keys: List[APIKeyInfo]              # All keys (without exposing full values)
    total: int


class APIKeyRevokeRequest(BaseModel):
    """Request model for revoking an API key"""
    api_key: str                        # The key to deactivate
    
    class Config:
        json_schema_extra = {
            "example": {
                "api_key": "user_7f3e8a2b1c5d9e4f6a8b2c4d6e8f0a1b"
            }
        }


class APIKeyRevokeResponse(BaseModel):
    """Response model for API key revocation"""
    status: str
    message: str
    api_key: str                        # Preview of revoked key


class RateLimitStatusResponse(BaseModel):
    """
    Response model for rate limit status
    
    Shows how many requests you've made and have remaining.
    
    RATE LIMITING:
    - Each API key has its own rate limit
    - Limits reset every minute
    - Exceeding limit returns HTTP 429
    """
    current_requests: int               # Requests made in current window
    rate_limit: int                     # Maximum allowed per minute
    remaining: int                      # Requests still available
    reset_in_seconds: int               # Seconds until counter resets


# ============================================================================
# EVALUATION SCHEMAS
# ============================================================================
# These schemas handle model evaluation on new labeled data
# ============================================================================

class EvaluationMetrics(BaseModel):
    """
    Metrics from model evaluation
    
    Calculated by comparing model predictions against actual labels.
    
    HOW TO READ:
    - recall: 0.85 means model finds 85% of churning customers
    - precision: 0.70 means 70% of predicted churns are correct
    - accuracy: 0.82 means 82% of all predictions are correct
    - f1_score: 0.77 balances recall and precision
    - roc_auc: 0.88 means excellent separation between classes
    """
    recall: float
    precision: float
    accuracy: float
    f1_score: float
    roc_auc: float
    test_samples: int                   # Number of samples used for evaluation


class EvaluationResponse(BaseModel):
    """Response from model evaluation"""
    status: str
    model_name: str
    metrics: EvaluationMetrics
    timestamp: str


class ComparisonItem(BaseModel):
    """Single model comparison item for side-by-side view"""
    model_name: str
    recall: float
    precision: float
    accuracy: float
    f1_score: float
    roc_auc: float


class ComparisonResponse(BaseModel):
    """Response from models comparison"""
    status: str
    comparison: List[ComparisonItem]    # Metrics for each model
    best_model: str                     # Which model performed best
    best_recall: float                  # Best recall score achieved
    timestamp: str


# ============================================================================
# AUTHENTICATION SCHEMAS
# ============================================================================
# These schemas define HTTP error responses for authentication and authorization failures
# ============================================================================

class AuthErrorResponse(BaseModel):
    """
    Response model for authentication errors (HTTP 401 Unauthorized)
    
    WHEN THIS OCCURS:
    - No API key provided in X-API-Key header
    - Invalid/expired API key provided
    - API key exists but is inactive (revoked)
    
    HOW TO FIX:
    - Include a valid X-API-Key header in your request
    - Contact administrator to get a valid key
    """
    detail: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Missing API Key. Please provide X-API-Key header"
            }
        }


class ForbiddenResponse(BaseModel):
    """
    Response model for forbidden access (HTTP 403 Forbidden)
    
    WHEN THIS OCCURS:
    - API key is valid but role doesn't have permission
    - Example: User key trying to access admin-only endpoint
    - Example: Dashboard key trying to make predictions
    
    ROLE REQUIREMENTS:
    - /retrain/* → admin only
    - /models/switch → admin only
    - /admin/keys → admin only
    - /predict → user, admin (NOT dashboard)
    - /label/stats → user, admin, dashboard
    """
    detail: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Admin privileges required for this endpoint"
            }
        }


class RateLimitErrorResponse(BaseModel):
    """
    Response model for rate limit exceeded (HTTP 429 Too Many Requests)
    
    WHEN THIS OCCURS:
    - You've made more requests per minute than your key allows
    - Default limits: user=100/min, admin=1000/min, dashboard=500/min
    
    WHAT HAPPENS:
    - Additional requests are rejected until the next minute
    - Use GET /admin/rate-limit to check your current usage
    
    HOW TO HANDLE:
    - Implement exponential backoff in your client
    - Request higher rate limit from administrator
    - Spread requests across multiple minutes
    """
    detail: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Rate limit exceeded. Maximum 100 requests per minute."
            }
        }


# ============================================================================
# API KEY MANAGEMENT SCHEMAS
# ============================================================================
# These schemas handle creating, listing, and revoking API keys
# All these endpoints require ADMIN role
# ============================================================================

class APIKeyCreateRequest(BaseModel):
    """
    Request model for creating a new API key (POST /admin/keys)
    
    API keys are used for authentication. Each key has:
    - A ROLE that determines what endpoints can be accessed
    - A NAME for human identification (e.g., "production_server", "analytics_team")
    - A RATE_LIMIT to prevent abuse
    
    ROLE OPTIONS:
    - "user": Can make predictions, update labels, view stats
    - "admin": Everything + model management, key management
    - "dashboard": Read-only access (for Streamlit dashboard)
    
    RATE LIMIT:
    - Defaults: user=100/min, admin=1000/min, dashboard=500/min
    - Can be customized per key
    - Counts ALL requests to any endpoint
    
    SECURITY NOTE:
    - Keys should be stored securely (environment variables, secret manager)
    - Never commit keys to version control
    - Rotate keys periodically
    """
    role: str                      # "user", "admin", or "dashboard"
    name: str                      # Human-readable identifier (e.g., "production_client")
    rate_limit: Optional[int] = None  # Requests per minute (None = use default for role)
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "name": "production_client",
                "rate_limit": 500
            }
        }


class APIKeyCreateResponse(BaseModel):
    """
    Response model for API key creation
    
    ⚠️ CRITICAL SECURITY NOTE ⚠️
    The full API key is returned ONLY in this response!
    Store it immediately - you cannot retrieve it again.
    
    The key is a cryptographically random string.
    Example: "a1b2c3d4e5f67890abcdef1234567890"
    
    HOW TO USE:
    Include in requests: headers = {"X-API-Key": "your_key_here"}
    
    WHAT YOU GET:
    - Full key (store this!)
    - Preview for display (first 8 + last 8 chars)
    - Metadata about the key
    """
    status: str
    api_key: str                    # ← THE ACTUAL KEY (store this securely!)
    key_preview: str                # Display version: "a1b2c3d4...34567890"
    name: str
    role: str
    rate_limit: int
    created_at: str
    message: str


class APIKeyInfo(BaseModel):
    """
    Information about an API key (without exposing the full key)
    
    Used when listing keys - shows only a preview for security.
    
    WHY PREVIEW ONLY?
    If someone gets access to your key list, they still can't use the keys.
    The full key is needed for authentication.
    
    PREVIEW FORMAT:
    - Takes first 8 characters + "..." + last 8 characters
    - Example: "user_7f3e...f0a1b" (can't be used for authentication)
    """
    key_preview: str               # Partial key for identification only
    name: str                      # Human-readable name
    role: str                      # "user", "admin", or "dashboard"
    rate_limit: int                # Max requests per minute
    created_at: str                # ISO timestamp of creation
    is_active: bool                # False if revoked


class APIKeyListResponse(BaseModel):
    """
    Response model for listing API keys (GET /admin/keys)
    
    Returns all keys in the system with previews.
    Full keys are NOT shown for security.
    
    USE CASE:
    - See which keys exist
    - Check if a key is still active
    - Identify keys by name
    """
    status: str
    keys: List[APIKeyInfo]         # List of key previews (no full keys!)
    total: int                     # Total number of keys
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "keys": [
                    {
                        "key_preview": "user_7f3e...f0a1b",
                        "name": "default_user",
                        "role": "user",
                        "rate_limit": 100,
                        "created_at": "2026-04-15T10:00:00",
                        "is_active": True
                    },
                    {
                        "key_preview": "admin_9a8b...5c4d3",
                        "name": "admin",
                        "role": "admin",
                        "rate_limit": 1000,
                        "created_at": "2026-04-15T10:00:00",
                        "is_active": True
                    }
                ],
                "total": 2
            }
        }


class APIKeyRevokeRequest(BaseModel):
    """
    Request model for revoking an API key (DELETE /admin/keys/{api_key})
    
    Revoking makes a key permanently invalid.
    
    WHEN TO REVOKE:
    - Key was compromised (exposed in logs, committed to git)
    - Employee left the company
    - Client stopped using the service
    - Key rotation (create new, revoke old)
    
    WHAT HAPPENS:
    - Key is marked as inactive in database
    - All future requests with this key get HTTP 401
    - Cannot be undone (create a new key instead)
    """
    api_key: str                   # The full API key to revoke
    
    class Config:
        json_schema_extra = {
            "example": {
                "api_key": "user_7f3e8a2b1c5d9e4f6a8b2c4d6e8f0a1b"
            }
        }


class APIKeyRevokeResponse(BaseModel):
    """
    Response model for API key revocation
    
    Confirms that the key has been revoked.
    Returns a preview (not the full key) for safety.
    """
    status: str
    message: str
    api_key: str                   # Preview of revoked key (e.g., "user_7f3e...")


class RateLimitStatusResponse(BaseModel):
    """
    Response model for rate limit status (GET /admin/rate-limit)
    
    Shows your current rate limit usage for the current minute window.
    
    HOW RATE LIMITING WORKS:
    - Each API key has a limit (e.g., 100 requests per minute)
    - Counter resets every minute
    - All endpoints count toward the limit
    
    WHAT THE NUMBERS MEAN:
    - current_requests: How many you've made so far this minute
    - rate_limit: Maximum allowed per minute
    - remaining: How many more you can make (rate_limit - current_requests)
    - reset_in_seconds: Seconds until the counter resets
    
    WHEN TO CHECK:
    - Before sending a large batch of requests
    - After receiving a 429 error
    - For monitoring and debugging
    
    EXAMPLE:
    current_requests=25, rate_limit=100, remaining=75, reset_in_seconds=45
    → You've used 25 of 100 requests, can make 75 more, resets in 45 seconds
    """
    current_requests: int          # Requests made in current window
    rate_limit: int                # Maximum allowed per minute
    remaining: int                 # Requests still available (rate_limit - current_requests)
    reset_in_seconds: int          # Seconds until counter resets to zero
    
    class Config:
        json_schema_extra = {
            "example": {
                "current_requests": 25,
                "rate_limit": 100,
                "remaining": 75,
                "reset_in_seconds": 45
            }
        }