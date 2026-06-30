# app/scripts/evaluate_model.py
"""
Model Evaluation Module

WHAT THIS MODULE DOES:
Evaluates trained churn prediction models on new labeled data.
Helps track model performance over time and compare different versions.

WHY EVALUATION MATTERS:
- Models degrade over time (concept drift)
- New data reveals if model still works
- Compare retrained models against original
- Make data-driven decisions about model deployment

KEY METRICS EXPLAINED:
- Recall: Of actual churners, how many did we catch? (Most important for churn!)
- Precision: Of predicted churners, how many actually churned?
- Accuracy: Overall correctness (less useful for imbalanced data)
- F1 Score: Balance between recall and precision
- ROC-AUC: Model's ability to separate churners from non-churners
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import cloudpickle
from typing import Dict, Any, Optional, List
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score
)

from app.config import settings
from app.scripts.model_manager import get_active_model_info
from app.utils.logger import logger
from app.core.database import get_db


# ============================================================================
# PIPELINE INSPECTION (Debugging Tool)
# ============================================================================

def inspect_pipeline_features(model_path):
    """
    Inspect what features the pipeline expects and produces.
    
    USE CASE:
    Debugging feature mismatch errors when testing models.
    Shows what columns the pipeline expects at each step.
    
    WHAT IT REVEALS:
    - Raw features the pipeline expects
    - Each transformer step in the pipeline
    - What features each step outputs
    
    EXAMPLE OUTPUT:
    ============================================================
    PIPELINE INSPECTION
    ============================================================
    
    Pipeline expects 19 raw features:
      0: gender
      1: SeniorCitizen
      2: Partner
      ...
    
    Pipeline steps:
      - totalcharges_cleaner: TotalChargesCleaner
      - feature_engineer: FeatureEngineer
        Outputs: ['tenure_months', 'avg_monthly', ...]
      - categorical_encoder: CategoricalEncoder
      - model: RandomForestClassifier
    
    ARGS:
        model_path: Path to .pkl model file
    
    RETURNS:
        Loaded pipeline object (for further inspection)
    """
    with open(model_path, 'rb') as f:
        pipeline = cloudpickle.load(f)
    
    print("=" * 60)
    print("PIPELINE INSPECTION")
    print("=" * 60)
    
    # Check if pipeline has feature_names_in_ (scikit-learn 1.0+)
    if hasattr(pipeline, 'feature_names_in_'):
        print(f"\nPipeline expects {len(pipeline.feature_names_in_)} raw features:")
        for i, f in enumerate(pipeline.feature_names_in_):
            print(f"  {i}: {f}")
    
    # Check each step in the pipeline
    print("\nPipeline steps:")
    for name, step in pipeline.named_steps.items():
        print(f"  - {name}: {step.__class__.__name__}")
        
        # Some transformers track their input features
        if hasattr(step, 'feature_names_in_'):
            print(f"    Expects: {step.feature_names_in_[:3]}...")
        
        # Some transformers can tell us their output features
        if hasattr(step, 'get_feature_names_out'):
            try:
                out = step.get_feature_names_out()
                print(f"    Outputs: {out[:3]}...")
            except:
                pass
    
    return pipeline


# ============================================================================
# LOAD LABELED DATA
# ============================================================================

def load_new_labeled_data_raw():
    """
    Load raw new labeled data from SQLite WITHOUT any preprocessing.
    
    WHAT THIS DOES:
    1. Fetches all rows with churn_label from database
    2. Renames churn_label → Churn for consistency
    3. Selects the 19 feature columns expected by the model
    4. Converts Churn (Yes/No) to numeric (1/0)
    
    WHY RAW?
    The model pipeline includes its own preprocessing steps.
    We should NOT apply additional transformations here.
    The pipeline's transformers handle everything.
    
    RETURNS:
        X_test: DataFrame with 19 feature columns (raw, unprocessed)
        y_test: Series with binary labels (1 = Churn, 0 = No Churn)
    
    RAISES:
        ValueError: If no labeled data found in database
    """
    with get_db() as conn:
        # Load only rows that have churn_label (labeled data)
        # These are ground truth labels provided by users
        df_labeled = pd.read_sql_query('''
            SELECT * FROM new_data 
            WHERE churn_label IS NOT NULL
        ''', conn)
    
    if len(df_labeled) == 0:
        raise ValueError("No labeled data found in database")
    
    # Rename for compatibility with older code
    # Database uses 'churn_label', older code expects 'Churn'
    df_labeled = df_labeled.rename(columns={'churn_label': 'Churn'})
    
    # List of all 19 features the model expects
    # Matches the Telco Customer Churn dataset columns
    feature_columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]
    
    # Extract features (X) and labels (y)
    X_test = df_labeled[feature_columns].copy()
    
    # Convert labels: "Yes" → 1, "No" → 0
    y_test = df_labeled['Churn'].map(lambda x: 1 if x == 'Yes' else 0)
    
    return X_test, y_test


# ============================================================================
# TEST SINGLE MODEL
# ============================================================================

def test_model_on_new_data(model_name: str = None, save_results: bool = True) -> Dict[str, Any]:
    """
    Test a model on new labeled data from SQLite database.
    
    THIS IS THE MAIN EVALUATION FUNCTION.
    
    EVALUATION PROCESS:
    1. Load labeled data from database (ground truth)
    2. Load the model (.pkl file)
    3. Apply the same preprocessing pipeline used during training
    4. Make predictions
    5. Calculate performance metrics
    6. (Optional) Save metrics to JSON file
    
    WHY THIS IS CRITICAL:
    - Validates model still works on new data
    - Detects performance degradation (concept drift)
    - Provides metrics for model comparison
    - Informs retraining decisions
    
    ARGS:
        model_name: Name of model file (None = use active model)
        save_results: Whether to save metrics to app/metrics/
    
    RETURNS:
        Dict with status, model_name, metrics, timestamp
    
    EXAMPLE RETURN:
    {
        'status': 'success',
        'model_name': 'full_churn_pipeline_cloud.pkl',
        'metrics': {
            'recall': 0.884,
            'precision': 0.565,
            'accuracy': 0.728,
            'f1_score': 0.689,
            'roc_auc': 0.824,
            'test_samples': 202,
            'churn_yes': 69,
            'churn_no': 133
        },
        'timestamp': '2026-04-18T10:30:00'
    }
    """
    print("=" * 50)
    print("TESTING MODEL ON NEW DATA")
    print("=" * 50)
    
    # ============================================
    # STEP 1: Determine which model to test
    # ============================================
    if model_name is None:
        # Use currently active model
        active_info = get_active_model_info()
        model_name = active_info['name']
        model_path = active_info['path']
        print(f"Using active model: {model_name}")
    else:
        # Use specified model
        model_path = Path('app/models') / model_name
        if not model_path.exists():
            return {
                'status': 'error',
                'message': f"Model {model_name} not found"
            }
        print(f"Using model: {model_name}")
    
    # ============================================
    # STEP 2: Load labeled data from database
    # ============================================
    with get_db() as conn:
        df_labeled = pd.read_sql_query('''
            SELECT * FROM new_data 
            WHERE churn_label IS NOT NULL
        ''', conn)
    
    if len(df_labeled) == 0:
        return {
            'status': 'error',
            'message': "No labeled data found in database"
        }
    
    # Extract true labels (ground truth)
    # Map: "Yes" → 1, "No" → 0
    y_true = df_labeled['churn_label'].map({'Yes': 1, 'No': 0}).values
    
    # ============================================
    # STEP 3: Load and prepare the model
    # ============================================
    with open(model_path, 'rb') as f:
        pipeline = cloudpickle.load(f)
    
    # ============================================
    # STEP 4: Prepare input data (apply pipeline)
    # ============================================
    # Drop columns that are NOT features:
    # - id, churn_label, label_timestamp, created_at (metadata)
    # - prediction, probability (previous model outputs)
    columns_to_drop = ['id', 'churn_label', 'label_timestamp', 'created_at', 'prediction', 'probability']
    X_raw = df_labeled.drop(columns=[col for col in columns_to_drop if col in df_labeled.columns], errors='ignore')
    
    # Drop customerID if present (not a feature for prediction)
    if 'customerID' in X_raw.columns:
        X_raw = X_raw.drop(columns=['customerID'])
    
    # ============================================
    # STEP 5: Apply preprocessing step by step
    # ============================================
    # This replicates what happens during training
    # Each transformer is applied in sequence
    
    X = X_raw.copy()
    
    # List of transformer steps in the pipeline (in order)
    transformers = ['totalcharges_cleaner', 'feature_engineer', 'categorical_encoder', 'drop_redundant', 'numerical_scaler']
    
    for name in transformers:
        if name in pipeline.named_steps:
            X = pipeline.named_steps[name].transform(X)
    
    # ============================================
    # STEP 6: Get the final model and expected columns
    # ============================================
    model = pipeline.named_steps['model']
    
    # The model expects specific column names (from training)
    # These are the columns after all transformations
    expected_cols = model.feature_names_
    
    # ============================================
    # STEP 7: Align columns (add missing, remove extras)
    # ============================================
    # This handles case where new data has different columns
    if hasattr(X, 'columns'):
        # Add missing columns with default value 0
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0
        
        # Remove extra columns not expected by model
        X = X[expected_cols]
    
    # ============================================
    # STEP 8: Make predictions
    # ============================================
    y_pred = model.predict(X)           # Binary: 0 or 1
    y_proba = model.predict_proba(X)[:, 1]  # Probability of churn (0.0 to 1.0)
    
    # ============================================
    # STEP 9: Calculate performance metrics
    # ============================================
    metrics = {
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_proba)),
        'test_samples': len(y_true),
        'churn_yes': int(sum(y_true == 1)),
        'churn_no': int(sum(y_true == 0))
    }
    
    # ============================================
    # STEP 10: Display results
    # ============================================
    print("\n" + "-" * 40)
    print("TEST RESULTS")
    print("-" * 40)
    print(f"Samples: {metrics['test_samples']} (Churn: {metrics['churn_yes']}, No: {metrics['churn_no']})")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    
    # ============================================
    # STEP 11: Save metrics to file (optional)
    # ============================================
    if save_results:
        import json
        metrics_dir = Path('app/metrics')
        metrics_dir.mkdir(exist_ok=True)
        metrics_path = metrics_dir / f"{Path(model_name).stem}_test_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {metrics_path}")
    
    return {
        'status': 'success',
        'model_name': model_name,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }


# ============================================================================
# COMPARE MULTIPLE MODELS
# ============================================================================

def compare_models_on_new_data(model_names: List[str]) -> Dict[str, Any]:
    """
    Compare multiple models on new labeled data from SQLite.
    
    USE CASE:
    - Compare original model vs retrained model
    - Decide which model to make active
    - Track performance over time
    
    HOW IT WORKS:
    1. Test each model using the same labeled data
    2. Collect metrics for all models
    3. Sort by recall (most important for churn prediction)
    4. Return comparison and best model
    
    WHY RECALL IS THE SORT KEY:
    For churn prediction, finding churners (recall) is more important
    than avoiding false alarms (precision). Missing a churner = lost revenue.
    
    ARGS:
        model_names: List of model filenames to compare
    
    RETURNS:
        Dict with comparison results, best model, and best recall score
    
    EXAMPLE RETURN:
    {
        'status': 'success',
        'comparison': [
            {'model_name': 'inc_churn_pipeline_retrained_cloud.pkl', 'recall': 0.89, ...},
            {'model_name': 'full_churn_pipeline_cloud.pkl', 'recall': 0.88, ...}
        ],
        'best_model': 'inc_churn_pipeline_retrained_cloud.pkl',
        'best_recall': 0.89,
        'timestamp': '2026-04-18T10:30:00'
    }
    """
    print("=" * 50)
    print("COMPARING MODELS ON NEW DATA")
    print("=" * 50)
    
    results = []
    
    # Test each model
    for model_name in model_names:
        print(f"\nTesting {model_name}...")
        result = test_model_on_new_data(model_name, save_results=False)
        
        if result['status'] == 'success':
            results.append({
                'model_name': model_name,
                'recall': result['metrics']['recall'],
                'precision': result['metrics']['precision'],
                'accuracy': result['metrics']['accuracy'],
                'f1_score': result['metrics']['f1_score'],
                'roc_auc': result['metrics']['roc_auc']
            })
        else:
            print(f"  Error: {result.get('message', 'Unknown error')}")
    
    if not results:
        return {
            'status': 'error',
            'message': 'No models were successfully tested'
        }
    
    # Sort by recall (highest first)
    # Recall is most important for churn prediction
    results.sort(key=lambda x: x['recall'], reverse=True)
    
    # Display comparison table
    print("\n" + "-" * 60)
    print("COMPARISON SUMMARY (Sorted by RECALL)")
    print("-" * 60)
    print(f"{'Model':<45} {'Recall':<10} {'Prec':<8} {'Acc':<8} {'F1':<8} {'ROC AUC':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['model_name']:<45} {r['recall']:.4f}   {r['precision']:.4f}   {r['accuracy']:.4f}   {r['f1_score']:.4f}   {r['roc_auc']:.4f}")
    
    print(f"\nBest model by RECALL: {results[0]['model_name']}")
    print(f"Recall: {results[0]['recall']:.4f}")
    
    return {
        'status': 'success',
        'comparison': results,
        'best_model': results[0]['model_name'],
        'best_recall': results[0]['recall'],
        'timestamp': datetime.now().isoformat()
    }


# ============================================================================
# HOW TO INTERPRET METRICS
# ============================================================================
# 
# RECALL (Most Important for Churn)
# --------------------------------
# Question: Of all customers who actually churned, how many did we catch?
# Formula: TP / (TP + FN)
# Example: 0.88 means we identified 88% of churners
# Goal: As high as possible (0.80+ is good)
# 
# PRECISION
# ---------
# Question: Of customers we predicted would churn, how many actually did?
# Formula: TP / (TP + FP)
# Example: 0.56 means 44% of our alerts were false alarms
# Goal: Balance with recall (trade-off)
# 
# ACCURACY
# --------
# Question: Overall, how many predictions were correct?
# Formula: (TP + TN) / (TP + TN + FP + FN)
# Warning: Can be misleading with imbalanced data
# Example: If 90% stay, predicting "stay" always gives 90% accuracy
# 
# F1 SCORE
# --------
# Question: Harmonic mean of precision and recall
# Formula: 2 * (precision * recall) / (precision + recall)
# Use: Single metric to compare models
# 
# ROC-AUC
# -------
# Question: How well does model separate churners from non-churners?
# Range: 0.5 (random) to 1.0 (perfect)
# Example: 0.82 means excellent separation
# 
# ============================================================================
# BUSINESS IMPACT INTERPRETATION
# ============================================================================
# 
# With recall = 0.88 and precision = 0.56:
# 
# For every 100 actual churners:
# - Model catches 88 of them (recall)
# - Misses 12 churners (lost revenue)
# 
# For every 100 customers flagged as "will churn":
# - 56 actually churn (precision)
# - 44 are false alarms (wasted retention budget)
# 
# This is GOOD for churn prediction because:
# - Catching churners is more important than false alarms
# - Better to offer discounts to 44 loyal customers
#   than to miss 12 churners
# 
# ============================================================================