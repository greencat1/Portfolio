from fastapi import FastAPI, HTTPException
from app.schemas import (
    PredictRequest, PredictResponse, 
    BatchPredictRequest, BatchPredictResponse,
    LabelUpdateRequest, LabelUpdateResponse,
    BatchLabelUpdateRequest, BatchLabelUpdateResponse,
    LabelInfoResponse, LabelStatisticsResponse, RetrainResponse, RetrainStatusResponse,
    ModelInfo, ModelsListResponse,
    SwitchModelRequest, SwitchModelResponse,
    DeleteModelRequest, DeleteModelResponse,
    CompareModelsResponse, ModelMetricsResponse
)
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

from app.scripts.retrain import retrain_on_new_data, full_retrain_on_combined_data, get_retraining_status
from typing import List
import pandas as pd
from pathlib import Path

from app.scripts.evaluate_model import (
    test_model_on_new_data,
    compare_models_on_new_data
)


app = FastAPI(title="Churn Prediction API", description="API for customer churn prediction and labeling")

@app.get("/")
def healthcheck():
    return {"status": "ok", "message": "Churn Prediction API is running"}

# ====================== PREDICTION ENDPOINTS ======================
@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    """Make a single prediction"""
    result = make_prediction(data)
    return result

@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch_endpoint(request: BatchPredictRequest):
    """Make batch predictions for multiple customers"""
    return predict_batch(request.customers)

# ====================== LABELING ENDPOINTS ======================
# ВАЖНО: специфические пути ДО параметризованных!

@app.get("/label/stats", response_model=LabelStatisticsResponse)
def get_label_statistics_endpoint():
    """Get statistics about labeled vs unlabeled data"""
    return get_label_statistics()

@app.get("/label/unlabeled/list")
def get_unlabeled_customers():
    """Get list of all customers without Churn labels"""
    unlabeled_df = get_unlabeled_data()
    if unlabeled_df.empty:
        return {"status": "success", "count": 0, "customers": []}
    
    customers = []
    for _, row in unlabeled_df.iterrows():
        customers.append({
            "customerID": row['customerID'],
            "prediction": int(row['prediction']) if pd.notna(row.get('prediction')) else None,
            "probability": float(row['probability']) if pd.notna(row.get('probability')) else None,
            "timestamp": row.get('timestamp')
        })
    
    return {
        "status": "success",
        "count": len(customers),
        "customers": customers
    }

@app.get("/label/labeled/list")
def get_labeled_customers():
    """Get list of all customers with Churn labels"""
    labeled_df = get_labeled_data()
    if labeled_df.empty:
        return {"status": "success", "count": 0, "customers": []}
    
    customers = []
    for _, row in labeled_df.iterrows():
        customers.append({
            "customerID": row['customerID'],
            "Churn": row['Churn'],
            "prediction": int(row['prediction']) if pd.notna(row.get('prediction')) else None,
            "probability": float(row['probability']) if pd.notna(row.get('probability')) else None,
            "label_timestamp": row.get('label_timestamp')
        })
    
    return {
        "status": "success",
        "count": len(customers),
        "customers": customers
    }

# Параметризованный эндпоинт ДОЛЖЕН быть ПОСЛЕ специфических
@app.get("/label/{customer_id}", response_model=LabelInfoResponse)
def get_label_endpoint(customer_id: str):
    """Get current Churn label and prediction info for a specific customer"""
    # Проверяем, что customer_id не является специальным значением
    if customer_id in ['stats', 'unlabeled', 'labeled']:
        raise HTTPException(status_code=404, detail=f"Customer ID '{customer_id}' not found")
    
    result = get_label(customer_id)
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    return result

@app.post("/label/update", response_model=LabelUpdateResponse)
def update_label_endpoint(request: LabelUpdateRequest):
    """
    Update Churn label for a single customer.
    Label must be 'Yes' (churned) or 'No' (not churned)
    """
    result = update_label(request.customerID, request.Churn)
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    return result

@app.post("/label/batch", response_model=BatchLabelUpdateResponse)
def batch_update_labels_endpoint(request: BatchLabelUpdateRequest):
    """
    Update Churn labels for multiple customers at once.
    Each label must be 'Yes' or 'No'
    """
    result = batch_update_labels(request.updates)
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    return result

# ====================== RETRAINING ENDPOINTS ======================

@app.get("/retrain/status", response_model=RetrainStatusResponse)
def retrain_status():
    """
    Check if there's enough labeled data for retraining.
    Returns statistics about available labeled data.
    """
    return get_retraining_status()

@app.post("/retrain/incremental", response_model=RetrainResponse)
def retrain_incremental():
    """
    Incremental retraining on new labeled data only.
    Faster - continues training from current model.
    Requires at least 10 labeled samples.
    
    This method:
    - Loads existing model
    - Applies transformations to new labeled data
    - Continues training (incremental learning)
    - Saves updated model to models/full_churn_pipeline_retrained.pkl
    """
    result = retrain_on_new_data()
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.post("/retrain/full", response_model=RetrainResponse)
def retrain_full():
    """
    Full retraining from scratch on combined data (old + new labeled).
    Slower but potentially more accurate.
    Uses all available data including original training data.
    
    This method:
    - Loads original training data (data/raw/telco_churn.csv)
    - Loads newly labeled data (data/new_data/new_data.csv)
    - Combines both datasets
    - Trains brand new model from scratch
    - Saves retrained model to models/full_churn_pipeline_retrained.pkl
    """
    result = full_retrain_on_combined_data()
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

# ====================== MODEL MANAGEMENT ENDPOINTS ======================

@app.get("/models", response_model=ModelsListResponse)
def list_models():
    """
    List all available models (excluding full_churn_pipeline.pkl)
    """
    models = get_all_models()
    active_info = get_active_model_info()
    
    return {
        "status": "success",
        "active_model": active_info['name'],
        "active_model_path": active_info['path'],
        "models": models,
        "total_models": len(models)
    }


@app.get("/models/active")
def get_active_model():
    """
    Get currently active model information
    """
    return {
        "status": "success",
        **get_active_model_info()
    }


@app.post("/models/switch", response_model=SwitchModelResponse)
def switch_model(request: SwitchModelRequest):
    """
    Switch active model by filename
    
    Example model names:
    - full_churn_pipeline_cloud.pkl
    - full_churn_pipeline_retrained_cloud.pkl
    - inc_churn_pipeline_retrained_cloud.pkl
    - inc_churn_pipeline_backup_cloud.pkl
    """
    result = switch_active_model(request.model_name)
    
    if result['status'] == 'error':
        raise HTTPException(status_code=400, detail=result['message'])
    
    return result


@app.post("/models/delete", response_model=DeleteModelResponse)
def delete_model_endpoint(request: DeleteModelRequest, force: bool = False):
    """
    Delete a model (cannot delete active model unless force=true)
    
    Example model names:
    - full_churn_pipeline_backup_cloud.pkl
    - inc_churn_pipeline_backup_cloud.pkl
    """
    result = delete_model(request.model_name, force=force)
    
    if result['status'] == 'error':
        raise HTTPException(status_code=400, detail=result['message'])
    
    return result


@app.get("/models/compare/{model1}/{model2}", response_model=CompareModelsResponse)
def compare_models_endpoint(
    model1: str = "full_churn_pipeline_cloud.pkl",
    model2: str = "full_churn_pipeline_retrained_cloud.pkl"
):
    """
    Compare two models by filename
    
    Default values:
    - model1: full_churn_pipeline_cloud.pkl
    - model2: full_churn_pipeline_retrained_cloud.pkl
    
    Available models:
    - full_churn_pipeline_cloud.pkl
    - full_churn_pipeline_retrained_cloud.pkl
    - inc_churn_pipeline_retrained_cloud.pkl
    - inc_churn_pipeline_backup_cloud.pkl
    """
    result = compare_models(model1, model2)
    
    if result['status'] == 'error':
        raise HTTPException(status_code=404, detail=result['message'])
    
    return result


@app.get("/models/{model_name}/metrics", response_model=ModelMetricsResponse)
def get_model_metrics_endpoint(
    model_name: str = "full_churn_pipeline_retrained_cloud.pkl"
):
    """
    Get metrics for a specific model
    
    Default: full_churn_pipeline_retrained_cloud.pkl
    
    Available models:
    - full_churn_pipeline_cloud.pkl
    - full_churn_pipeline_retrained_cloud.pkl
    - inc_churn_pipeline_retrained_cloud.pkl
    - inc_churn_pipeline_backup_cloud.pkl
    """
    metrics = get_model_metrics(model_name)
    
    if not metrics:
        raise HTTPException(status_code=404, detail=f"No metrics found for model {model_name}")
    
    return {
        "status": "success",
        "model_name": model_name,
        "metrics": metrics
    }

@app.post("/test/{model_name}")
def test_model(model_name: str):
    result = test_model_on_new_data(model_name)
    if result['status'] == 'error':
        raise HTTPException(status_code=404, detail=result['message'])
    return result
