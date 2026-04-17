# app/main.py
"""
Churn Prediction API - Main Application Entry Point

This FastAPI application provides endpoints for:
- Customer churn prediction (single and batch)
- Label management (update and retrieve customer churn labels)
- Model management (switch, delete, compare models)
- Retraining (incremental and full retraining)
- Security with API key authentication and role-based access control

Security Levels:
- Public: Health check endpoint (no authentication)
- User: Any valid API key (predictions, labeling)
- Admin: Admin API key required (model management, retraining, key management)
"""

from fastapi import FastAPI, HTTPException, Depends
from typing import List
import pandas as pd
from pathlib import Path

# Import authentication and rate limiting modules
from app.auth import (
    verify_api_key,      # Validates any API key
    require_user,        # Requires user or admin role (for data operations)
    require_admin,       # Requires admin role (for model management)
    create_api_key,      # Create new API keys (admin only)
    revoke_api_key,      # Revoke existing API keys (admin only)
    list_api_keys        # List all API keys (admin only)
)
from app.rate_limit import check_rate_limit, get_rate_limit_status

# Import schemas (request/response models)
from app.schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    LabelUpdateRequest, LabelUpdateResponse,
    BatchLabelUpdateRequest, BatchLabelUpdateResponse,
    LabelInfoResponse, LabelStatisticsResponse,
    RetrainResponse, RetrainStatusResponse,
    ModelInfo, ModelsListResponse,
    SwitchModelRequest, SwitchModelResponse,
    DeleteModelRequest, DeleteModelResponse,
    CompareModelsResponse, ModelMetricsResponse,
    APIKeyCreateRequest,
    APIKeyCreateResponse,
    APIKeyInfo,
    APIKeyListResponse,
    APIKeyRevokeRequest,
    APIKeyRevokeResponse,
    RateLimitStatusResponse
)

# Import business logic scripts
from app.scripts.predict import make_prediction
from app.scripts.predict_batch import predict_batch
from app.scripts.put_lbl import (
    update_label,
    batch_update_labels,
    get_label,
    get_label_statistics,
    get_unlabeled_data,
    get_labeled_data
)
from app.scripts.model_manager import (
    get_all_models,
    get_active_model_info,
    switch_active_model,
    delete_model,
    compare_models,
    get_model_metrics
)
from app.scripts.retrain import (
    retrain_on_new_data,
    full_retrain_on_combined_data,
    get_retraining_status
)
from app.scripts.evaluate_model import (
    test_model_on_new_data,
    compare_models_on_new_data
)

# Import logger
from app.utils.logger import logger

# Import DB
import app.core.database
from app.core.database import init_db
from app.core.key_input import input_and_save_keys, refresh_key_cache


# Initialize FastAPI application
app = FastAPI(
    title="Churn Prediction API",
    description="API for customer churn prediction and labeling with role-based access control",
    version="2.0.0"
)

# Create keys and write them to the database
@app.on_event("startup")
def startup_event():
    """Initialize database, prompt for keys, refresh cache on startup"""
    print("\n" + "="*60)
    print(" STARTING CHURN PREDICTION API")
    print("="*60)
    
    # 1. Initialize database tables
    from app.core.database import init_db, get_db
    init_db()
    
    # 2. Prompt for API keys (will block until entered)
    from app.core.key_input import input_and_save_keys
    try:
        input_and_save_keys()
    except ValueError as e:
        print(f"❌ Startup failed: {e}")
        raise
    
    # 3. Refresh auth cache
    from app.auth import refresh_key_cache
    refresh_key_cache()
    
    print("="*60)
    print(" API is ready! Press Ctrl+C to stop")
    print("="*60 + "\n")

# ============================================================================
# PUBLIC ENDPOINTS (No Authentication Required)
# ============================================================================

@app.get("/")
def healthcheck():
    """
    Health check endpoint.
    No authentication required. Used for monitoring and service discovery.
    
    Returns:
        dict: Status message indicating API is running
    """
    logger.info("Health check requested")
    return {
        "status": "ok",
        "message": "Churn Prediction API is running",
        "version": "2.0.0"
    }


# ============================================================================
# DATA & PREDICTION ENDPOINTS (User + Admin Access)
# These endpoints require any valid API key (user or admin role)
# ============================================================================

@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest, auth: dict = Depends(require_user)):
    """
    Make a single churn prediction for a customer.
    
    Requires a valid API key (user or admin role).
    Rate limits apply based on the API key configuration.
    
    Args:
        data: Customer information (all features required for prediction)
        auth: Authentication data from API key (injected by dependency)
    
    Returns:
        PredictResponse: Prediction result (0 = No Churn, 1 = Churn) and probability
    """
    # Apply rate limiting based on API key
    check_rate_limit(auth["api_key"])
    
    # Make prediction using the loaded model
    try:
        result = make_prediction(data)
        logger.info(f"Prediction successful for customer {data.customerID}")
        return result
    except Exception as e:
        logger.error(f"Prediction failed for customer {data.customerID}: {str(e)}", exc_info=True)
        raise


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch_endpoint(
    request: BatchPredictRequest,
    auth: dict = Depends(require_user)
):
    """
    Make batch predictions for multiple customers.
    
    More efficient than making individual predictions.
    Requires a valid API key (user or admin role).
    
    Args:
        request: List of customer data objects
        auth: Authentication data from API key (injected by dependency)
    
    Returns:
        BatchPredictResponse: List of predictions with probabilities
    """
    check_rate_limit(auth["api_key"])
    
    try:
        result = predict_batch(request.customers)
        logger.info(f"Batch prediction completed: {result['total']} customers")
        return result
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
        raise


@app.get("/label/stats", response_model=LabelStatisticsResponse)
def get_label_statistics_endpoint(auth: dict = Depends(require_user)):
    """
    Get statistics about labeled vs unlabeled data.
    
    Returns counts of total, labeled, unlabeled records,
    and churn distribution.
    Requires a valid API key (user or admin role).
    
    Args:
        auth: Authentication data from API key (injected by dependency)
    
    Returns:
        LabelStatisticsResponse: Statistics about labeling progress
    """
    check_rate_limit(auth["api_key"])
    
    try:
        result = get_label_statistics()
        logger.info(f"Label statistics retrieved: {result.get('labeling_progress', 'N/A')}")
        return result
    except Exception as e:
        logger.error(f"Failed to get label statistics: {str(e)}", exc_info=True)
        raise


@app.get("/label/unlabeled/list")
def get_unlabeled_customers(auth: dict = Depends(require_user)):
    """
    Get list of all customers without Churn labels.
    
    Useful for identifying which customers need manual labeling.
    Requires a valid API key (user or admin role).
    
    Args:
        auth: Authentication data from API key (injected by dependency)
    
    Returns:
        dict: List of unlabeled customers with their prediction data
    """
    check_rate_limit(auth["api_key"])
    
    try:
        unlabeled_df = get_unlabeled_data()
        if unlabeled_df.empty:
            logger.info("No unlabeled customers found")
            return {"status": "success", "count": 0, "customers": []}
        
        customers = []
        for _, row in unlabeled_df.iterrows():
            customer_data = {
                                "customerID": row['customerID'],
                                "prediction": int(row['prediction']) if pd.notna(row.get('prediction')) else None,
                                "probability": float(row['probability']) if pd.notna(row.get('probability')) else None,
                                "timestamp": row.get('created_at'),
                                
                                "MonthlyCharges": float(row['MonthlyCharges']) if pd.notna(row.get('MonthlyCharges')) else 0,
                                "tenure": int(row['tenure']) if pd.notna(row.get('tenure')) else 0,
                                "Contract": row.get('Contract'),
                                "OnlineSecurity": row.get('OnlineSecurity')
                            }
            customers.append(customer_data)
            
            
        logger.info(f"Retrieved {len(customers)} unlabeled customers")
        return {
            "status": "success",
            "count": len(customers),
            "customers": customers
        }
    except Exception as e:
        logger.error(f"Failed to get unlabeled customers: {str(e)}", exc_info=True)
        raise


@app.get("/label/labeled/list")
def get_labeled_customers(auth: dict = Depends(require_user)):
    """
    Get list of all customers with Churn labels.
    
    Returns already labeled customers with their true labels.
    Requires a valid API key (user or admin role).
    
    Args:
        auth: Authentication data from API key (injected by dependency)
    
    Returns:
        dict: List of labeled customers with their true labels
    """
    check_rate_limit(auth["api_key"])
    
    try:
        labeled_df = get_labeled_data()
        if labeled_df.empty:
            logger.info("No labeled customers found")
            return {"status": "success", "count": 0, "customers": []}
        
        customers = []
        for _, row in labeled_df.iterrows():
            customer_data = {
                                "customerID": row['customerID'],
                                "prediction": int(row['prediction']) if pd.notna(row.get('prediction')) else None,
                                "probability": float(row['probability']) if pd.notna(row.get('probability')) else None,
                                "timestamp": row.get('created_at'),
                                "Churn" : row['churn_label'],
                                "MonthlyCharges": float(row['MonthlyCharges']) if pd.notna(row.get('MonthlyCharges')) else 0,
                                "tenure": int(row['tenure']) if pd.notna(row.get('tenure')) else 0,
                                "Contract": row.get('Contract'),
                                "OnlineSecurity": row.get('OnlineSecurity')
                            }
            customers.append(customer_data)
        
        logger.info(f"Retrieved {len(customers)} labeled customers")
        return {
            "status": "success",
            "count": len(customers),
            "customers": customers
        }
    except Exception as e:
        logger.error(f"Failed to get labeled customers: {str(e)}", exc_info=True)
        raise


@app.get("/label/{customer_id}", response_model=LabelInfoResponse)
def get_label_endpoint(customer_id: str, auth: dict = Depends(require_user)):
    """
    Get current Churn label and prediction info for a specific customer.
    
    Returns both the true label (if available) and the model's prediction.
    Requires a valid API key (user or admin role).
    
    Args:
        customer_id: Unique customer identifier
        auth: Authentication data from API key (injected by dependency)
    
    Returns:
        LabelInfoResponse: Label and prediction information for the customer
    """
    check_rate_limit(auth["api_key"])
    
    # Prevent path traversal attacks on special IDs
    if customer_id in ['stats', 'unlabeled', 'labeled']:
        logger.warning(f"Attempt to access reserved ID: {customer_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Customer ID '{customer_id}' not found"
        )
    
    try:
        result = get_label(customer_id)
        if result["status"] == "error":
            logger.warning(f"Label not found for customer {customer_id}")
            raise HTTPException(status_code=404, detail=result["message"])
        
        logger.info(f"Label retrieved for customer {customer_id}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get label for customer {customer_id}: {str(e)}", exc_info=True)
        raise


@app.post("/label/update", response_model=LabelUpdateResponse)
def update_label_endpoint(
    request: LabelUpdateRequest,
    auth: dict = Depends(require_user)
):
    """
    Update Churn label for a single customer.
    
    Labels are used for model retraining and evaluation.
    Requires a valid API key (user or admin role).
    
    Args:
        request: Contains customerID and Churn label ('Yes' or 'No')
        auth: Authentication data from API key (injected by dependency)
    
    Returns:
        LabelUpdateResponse: Status of the update operation
    """
    check_rate_limit(auth["api_key"])
    
    try:
        result = update_label(request.customerID, request.Churn)
        if result["status"] == "error":
            logger.warning(f"Label update failed for {request.customerID}: {result.get('message')}")
            raise HTTPException(status_code=404, detail=result["message"])
        
        logger.info(f"Label updated for customer {request.customerID}: {request.Churn}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update label for {request.customerID}: {str(e)}", exc_info=True)
        raise


@app.post("/label/batch", response_model=BatchLabelUpdateResponse)
def batch_update_labels_endpoint(
    request: BatchLabelUpdateRequest,
    auth: dict = Depends(require_user)
):
    """
    Update Churn labels for multiple customers at once.
    
    More efficient than individual updates for bulk operations.
    Requires a valid API key (user or admin role).
    
    Args:
        request: List of customer label updates
        auth: Authentication data from API key (injected by dependency)
    
    Returns:
        BatchLabelUpdateResponse: Summary of successful and failed updates
    """
    check_rate_limit(auth["api_key"])
    
    try:
        result = batch_update_labels(request.updates)
        logger.info(f"Batch label update completed: {result['successful']} successful, {result['failed']} failed")
        
        if result["status"] == "error":
            raise HTTPException(status_code=404, detail=result["message"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to batch update labels: {str(e)}", exc_info=True)
        raise


@app.get("/admin/rate-limit")
def get_rate_limit_info(auth: dict = Depends(require_user)):
    """
    Get current rate limit status for your API key.
    
    Shows how many requests you've made and how many remain.
    Requires a valid API key (user or admin role).
    
    Args:
        auth: Authentication data from API key (injected by dependency)
    
    Returns:
        dict: Rate limit information (current, limit, remaining, reset time)
    """
    result = get_rate_limit_status(auth["api_key"])
    logger.info(f"Rate limit info requested for key {auth['api_key'][:8]}...")
    return result


# ============================================================================
# MODEL MANAGEMENT ENDPOINTS (Admin Only)
# These endpoints require admin API key
# ============================================================================

@app.get("/retrain/status", response_model=RetrainStatusResponse)
def retrain_status(auth: dict = Depends(require_admin)):
    """
    Check if there's enough labeled data for retraining.
    
    Returns statistics about available labeled data.
    Admin only endpoint.
    
    Args:
        auth: Authentication data from API key (admin role required)
    
    Returns:
        RetrainStatusResponse: Status and statistics for retraining readiness
    """
    try:
        result = get_retraining_status()
        logger.info(f"Retraining status checked: {result.get('status')}")
        return result
    except Exception as e:
        logger.error(f"Failed to get retraining status: {str(e)}", exc_info=True)
        raise


@app.post("/retrain/incremental", response_model=RetrainResponse)
def retrain_incremental(auth: dict = Depends(require_admin)):
    """
    Incremental retraining on new labeled data only.
    
    Faster method - continues training from current model.
    Requires at least 10 labeled samples.
    Admin only endpoint.
    
    Process:
    1. Loads existing model
    2. Applies transformations to new labeled data
    3. Continues training (incremental learning)
    4. Saves updated model
    
    Args:
        auth: Authentication data from API key (admin role required)
    
    Returns:
        RetrainResponse: Status and metrics of retraining operation
    """
    logger.info("Starting incremental retraining")
    result = retrain_on_new_data()
    
    if result["status"] == "error":
        logger.error(f"Incremental retraining failed: {result.get('message')}")
        raise HTTPException(status_code=400, detail=result["message"])
    
    logger.info(f"Incremental retraining completed: {result.get('samples_used', 0)} samples used")
    return result


@app.post("/retrain/full", response_model=RetrainResponse)
def retrain_full(auth: dict = Depends(require_admin)):
    """
    Full retraining from scratch on combined data (old + new labeled).
    
    Slower but potentially more accurate.
    Uses all available data including original training data.
    Admin only endpoint.
    
    Process:
    1. Loads original training data
    2. Loads newly labeled data
    3. Combines both datasets
    4. Trains brand new model from scratch
    
    Args:
        auth: Authentication data from API key (admin role required)
    
    Returns:
        RetrainResponse: Status and metrics of retraining operation
    """
    logger.info("Starting full retraining")
    result = full_retrain_on_combined_data()
    
    if result["status"] == "error":
        logger.error(f"Full retraining failed: {result.get('message')}")
        raise HTTPException(status_code=400, detail=result["message"])
    
    logger.info(f"Full retraining completed: {result.get('samples_used', 0)} total samples")
    return result


@app.get("/models", response_model=ModelsListResponse)
def list_models(auth: dict = Depends(require_admin)):
    """
    List all available models.
    
    Returns all models in the models directory (excluding protected models).
    Admin only endpoint.
    
    Args:
        auth: Authentication data from API key (admin role required)
    
    Returns:
        ModelsListResponse: List of models with metadata and active model info
    """
    try:
        models = get_all_models()
        active_info = get_active_model_info()
        logger.info(f"Listed {len(models)} models")
        
        return {
            "status": "success",
            "active_model": active_info['name'],
            "active_model_path": active_info['path'],
            "models": models,
            "total_models": len(models)
        }
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}", exc_info=True)
        raise


@app.get("/models/active")
def get_active_model(auth: dict = Depends(require_admin)):
    """
    Get currently active model information.
    
    Returns details about the model currently used for predictions.
    Admin only endpoint.
    
    Args:
        auth: Authentication data from API key (admin role required)
    
    Returns:
        dict: Active model information (name, path, size, creation date)
    """
    try:
        result = get_active_model_info()
        logger.info(f"Active model retrieved: {result.get('name')}")
        return {
            "status": "success",
            **result
        }
    except Exception as e:
        logger.error(f"Failed to get active model: {str(e)}", exc_info=True)
        raise


@app.post("/models/switch", response_model=SwitchModelResponse)
def switch_model(request: SwitchModelRequest, auth: dict = Depends(require_admin)):
    """
    Switch active model by filename.
    
    Changes which model is used for predictions.
    Admin only endpoint.
    
    Available models:
    - full_churn_pipeline_cloud.pkl (original trained model)
    - full_churn_pipeline_retrained_cloud.pkl (full retrained model)
    - inc_churn_pipeline_retrained_cloud.pkl (incrementally retrained model)
    - inc_churn_pipeline_backup_cloud.pkl (backup of previous model)
    
    Args:
        request: Contains model_name of the target model
        auth: Authentication data from API key (admin role required)
    
    Returns:
        SwitchModelResponse: Status with previous and current model info
    """
    logger.info(f"Switching model to: {request.model_name}")
    result = switch_active_model(request.model_name)
    
    if result['status'] == 'error':
        logger.error(f"Model switch failed: {result.get('message')}")
        raise HTTPException(status_code=400, detail=result['message'])
    
    logger.info(f"Model switched from {result['previous_model']} to {result['current_model']}")
    return result


@app.post("/models/delete", response_model=DeleteModelResponse)
def delete_model_endpoint(
    request: DeleteModelRequest,
    force: bool = False,
    auth: dict = Depends(require_admin)
):
    """
    Delete a model file.
    
    Cannot delete the active model unless force=true is specified.
    Protected models cannot be deleted.
    Admin only endpoint.
    
    Args:
        request: Contains model_name of the model to delete
        force: If True, allows deletion of active model (use with caution)
        auth: Authentication data from API key (admin role required)
    
    Returns:
        DeleteModelResponse: Status of deletion operation
    """
    logger.warning(f"Deleting model: {request.model_name}, force={force}")
    result = delete_model(request.model_name, force=force)
    
    if result['status'] == 'error':
        logger.error(f"Model deletion failed: {result.get('message')}")
        raise HTTPException(status_code=400, detail=result['message'])
    
    logger.info(f"Model deleted: {result['deleted_model']}")
    return result


@app.get("/models/compare/{model1}/{model2}", response_model=CompareModelsResponse)
def compare_models_endpoint(
    model1: str = "full_churn_pipeline_cloud.pkl",
    model2: str = "full_churn_pipeline_retrained_cloud.pkl",
    auth: dict = Depends(require_admin)
):
    """
    Compare two models by filename.
    
    Shows metrics, file sizes, and creation dates for both models.
    Admin only endpoint.
    
    Default values:
    - model1: full_churn_pipeline_cloud.pkl
    - model2: full_churn_pipeline_retrained_cloud.pkl
    
    Args:
        model1: First model filename to compare
        model2: Second model filename to compare
        auth: Authentication data from API key (admin role required)
    
    Returns:
        CompareModelsResponse: Detailed comparison of both models
    """
    logger.info(f"Comparing models: {model1} vs {model2}")
    result = compare_models(model1, model2)
    
    if result['status'] == 'error':
        logger.error(f"Model comparison failed: {result.get('message')}")
        raise HTTPException(status_code=404, detail=result['message'])
    
    return result


@app.get("/models/{model_name}/metrics", response_model=ModelMetricsResponse)
def get_model_metrics_endpoint(
    model_name: str = "full_churn_pipeline_retrained_cloud.pkl",
    auth: dict = Depends(require_admin)
):
    """
    Get metrics for a specific model.
    
    Returns training metrics (accuracy, precision, recall, F1 score, ROC-AUC).
    Admin only endpoint.
    
    Args:
        model_name: Name of the model file
        auth: Authentication data from API key (admin role required)
    
    Returns:
        ModelMetricsResponse: Model metrics and metadata
    """
    metrics = get_model_metrics(model_name)
    
    if not metrics:
        logger.warning(f"No metrics found for model {model_name}")
        raise HTTPException(
            status_code=404,
            detail=f"No metrics found for model {model_name}"
        )
    
    logger.info(f"Metrics retrieved for model {model_name}")
    return {
        "status": "success",
        "model_name": model_name,
        "metrics": metrics
    }


@app.post("/test/{model_name}")
def test_model(model_name: str, auth: dict = Depends(require_admin)):
    """
    Test a model on new labeled data.
    
    Evaluates model performance on recently labeled data.
    Admin only endpoint.
    
    Args:
        model_name: Name of the model to test
        auth: Authentication data from API key (admin role required)
    
    Returns:
        dict: Evaluation metrics (recall, precision, accuracy, F1, ROC-AUC)
    """
    logger.info(f"Testing model: {model_name}")
    result = test_model_on_new_data(model_name)
    
    if result['status'] == 'error':
        logger.error(f"Model test failed: {result.get('message')}")
        raise HTTPException(status_code=404, detail=result['message'])
    
    logger.info(f"Model {model_name} test completed: recall={result['metrics'].get('recall', 'N/A')}")
    return result


# ============================================================================
# ADMIN KEY MANAGEMENT ENDPOINTS (Admin Only)
# ============================================================================

@app.post("/admin/keys", response_model=APIKeyCreateResponse)
def create_new_api_key(
    role: str,
    name: str,
    rate_limit: int = None,
    auth: dict = Depends(require_admin)
):
    """
    Create a new API key.
    
    - **role**: 'user' or 'admin' - determines access level
    - **name**: Display name for identifying the key
    - **rate_limit**: Optional custom rate limit (default: 100 for user, 1000 for admin)
    
    Returns the new API key. Store it securely!
    """
    result = create_api_key(role, name, rate_limit)
    logger.info(f"New API key created: role={role}, name={name}, rate_limit={rate_limit}")
    return result


@app.delete("/admin/keys/{api_key}")
def revoke_existing_api_key(api_key: str, auth: dict = Depends(require_admin)):
    """
    Revoke (deactivate) an API key.
    
    Revoked keys cannot be used for authentication anymore.
    Useful for key rotation or when a key is compromised.
    Admin only endpoint.
    
    Args:
        api_key: The API key to revoke
        auth: Authentication data from API key (admin role required)
    
    Returns:
        dict: Status of the revocation operation
    """
    logger.warning(f"Revoking API key: {api_key[:8]}...")
    return revoke_api_key(api_key, auth)


@app.get("/admin/keys")
def list_all_api_keys(auth: dict = Depends(require_admin)):
    """
    List all registered API keys.
    
    Returns preview of each key (only first and last 8 characters for security).
    Admin only endpoint.
    
    Args:
        auth: Authentication data from API key (admin role required)
    
    Returns:
        dict: List of key previews with metadata and total count
    """
    result = list_api_keys(auth)
    logger.info(f"Listed {result.get('total', 0)} API keys")
    return result