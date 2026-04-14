from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class PredictRequest(BaseModel):
    customerID : object 
    gender : object 
    SeniorCitizen : int
    Partner : object 
    Dependents : object 
    tenure : int
    PhoneService : object
    MultipleLines : object
    InternetService : object
    OnlineSecurity : object
    OnlineBackup : object
    DeviceProtection : object
    TechSupport : object
    StreamingTV : object
    StreamingMovies : object
    Contract : object
    PaperlessBilling : object
    PaymentMethod : object
    MonthlyCharges : float
    TotalCharges : object

    class Config:
        
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
    prediction: int
    probability: float


class BatchPredictRequest(BaseModel):
    customers: List[PredictRequest]

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
    predictions: List[PredictResponse]
    total: int


# Добавьте эти классы в конец файла schemas.py

# ====================== LABEL SCHEMAS ======================
class LabelUpdateRequest(BaseModel):
    customerID: str
    Churn: str  # "Yes" или "No"
    
    class Config:
        json_schema_extra = {
            "example": {
                "customerID": "7590-VHVEG",
                "Churn": "Yes"
            }
        }

class LabelUpdateResponse(BaseModel):
    status: str
    message: str
    customerID: str
    old_label: Optional[str] = None
    new_label: Optional[str] = None
    timestamp: Optional[str] = None

class BatchLabelUpdateRequest(BaseModel):
    updates: List[LabelUpdateRequest]
    
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
    status: str
    total_updates: int
    successful: int
    failed: int
    results: List[dict]

class LabelInfoResponse(BaseModel):
    status: str
    customerID: str
    Churn: Optional[str] = None
    prediction: Optional[int] = None
    probability: Optional[float] = None
    timestamp: Optional[str] = None
    label_timestamp: Optional[str] = None

class LabelStatisticsResponse(BaseModel):
    status: str
    total_records: int
    labeled_records: int
    unlabeled_records: int
    churn_yes: int
    churn_no: int
    labeling_progress: str


# ====================== RETRAINING SCHEMAS ======================


class RetrainResponse(BaseModel):
    """Response schema for retraining endpoints"""
    status: str
    message: str
    samples_used: Optional[int] = None
    old_samples: Optional[int] = None
    new_labeled_samples: Optional[int] = None
    samples_attempted: Optional[int] = None
    churn_yes: Optional[int] = None
    churn_no: Optional[int] = None
    metrics: Optional[Dict[str, float]] = None
    model_path: Optional[str] = None
    timestamp: Optional[str] = None


class RetrainStatusResponse(BaseModel):
    """Response schema for retraining status endpoint"""
    status: str
    message: str
    total_records: Optional[int] = None
    labeled_records: Optional[int] = None
    unlabeled_records: Optional[int] = None
    churn_yes: Optional[int] = None
    churn_no: Optional[int] = None
    can_retrain: Optional[bool] = None


# ====================== MODEL MANAGEMENT SCHEMAS ======================

class ModelInfo(BaseModel):
    """Information about a single model"""
    name: str
    path: str
    size_mb: float
    created_at: str
    is_active: bool
    metrics: Optional[Dict[str, Any]] = None
    
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
    """Response for listing all models"""
    status: str
    active_model: str
    active_model_path: str
    models: List[ModelInfo]
    total_models: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "active_model": "full_churn_pipeline_retrained_cloud.pkl",
                "active_model_path": "models/full_churn_pipeline_retrained_cloud.pkl",
                "models": [],
                "total_models": 0
            }
        }


class SwitchModelRequest(BaseModel):
    """Request to switch active model"""
    model_name: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "full_churn_pipeline_retrained_cloud.pkl"
            }
        }


class SwitchModelResponse(BaseModel):
    """Response after switching model"""
    status: str
    message: str
    previous_model: str
    current_model: str
    model_path: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Active model switched to full_churn_pipeline_retrained_cloud.pkl",
                "previous_model": "full_churn_pipeline_cloud.pkl",
                "current_model": "full_churn_pipeline_retrained_cloud.pkl",
                "model_path": "models/full_churn_pipeline_retrained_cloud.pkl"
            }
        }


class DeleteModelRequest(BaseModel):
    """Request to delete a model"""
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
    deleted_model: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Model deleted successfully",
                "deleted_model": "full_churn_pipeline_backup_cloud.pkl"
            }
        }


class CompareModelsResponse(BaseModel):
    """Response for comparing two models"""
    status: str
    model1: Dict[str, Any]
    model2: Dict[str, Any]
    comparison: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "model1": {},
                "model2": {},
                "comparison": {}
            }
        }


class ModelMetricsResponse(BaseModel):
    """Response for model metrics"""
    status: str
    model_name: str
    metrics: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "model_name": "full_churn_pipeline_retrained_cloud.pkl",
                "metrics": {
                    "samples_used": 7043,
                    "accuracy": 0.8123
                }
            }
        }


# ====================== EVALUATION SCHEMAS ======================

class EvaluationMetrics(BaseModel):
    """Metrics from model evaluation"""
    recall: float
    precision: float
    accuracy: float
    f1_score: float
    roc_auc: float
    test_samples: int


class EvaluationResponse(BaseModel):
    """Response from model evaluation"""
    status: str
    model_name: str
    metrics: EvaluationMetrics
    timestamp: str


class ComparisonItem(BaseModel):
    """Single model comparison item"""
    model_name: str
    recall: float
    precision: float
    accuracy: float
    f1_score: float
    roc_auc: float


class ComparisonResponse(BaseModel):
    """Response from models comparison"""
    status: str
    comparison: List[ComparisonItem]
    best_model: str
    best_recall: float
    timestamp: str