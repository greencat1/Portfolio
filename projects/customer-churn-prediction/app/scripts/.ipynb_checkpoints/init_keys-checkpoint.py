# app/scripts/evaluate_model.py
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


def inspect_pipeline_features(model_path):
    """
    Inspect what features the pipeline expects and produces
    """
    with open(model_path, 'rb') as f:
        pipeline = cloudpickle.load(f)
    
    print("=" * 60)
    print("PIPELINE INSPECTION")
    print("=" * 60)
    
    # Check if pipeline has feature_names_in_
    if hasattr(pipeline, 'feature_names_in_'):
        print(f"\nPipeline expects {len(pipeline.feature_names_in_)} raw features:")
        for i, f in enumerate(pipeline.feature_names_in_):
            print(f"  {i}: {f}")
    
    # Check each step
    print("\nPipeline steps:")
    for name, step in pipeline.named_steps.items():
        print(f"  - {name}: {step.__class__.__name__}")
        if hasattr(step, 'feature_names_in_'):
            print(f"    Expects: {step.feature_names_in_[:3]}...")
        if hasattr(step, 'get_feature_names_out'):
            try:
                out = step.get_feature_names_out()
                print(f"    Outputs: {out[:3]}...")
            except:
                pass
    
    return pipeline


def load_new_labeled_data_raw():
    """
    Load raw new labeled data WITHOUT any preprocessing
    """
    new_data_path = Path('app/data/new_data/new_data.csv')
    
    if not new_data_path.exists():
        raise FileNotFoundError(f"New data file not found at {new_data_path}")
    
    df = pd.read_csv('app/data/new_data/new_data.csv')
    df_labeled = df[df['Churn'].notna()].copy()
        
    feature_columns = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'
        ]
        
    X_test = df_labeled[feature_columns].copy()
    y_test = df_labeled['Churn'].map(lambda x: 1 if x =='Yes' else 0)
    return X_test, y_test

def test_model_on_new_data(model_name: str = None, save_results: bool = True) -> Dict[str, Any]:
    """
    Test a model on new labeled data from new_data.csv
    """
    print("=" * 50)
    print("TESTING MODEL ON NEW DATA")
    print("=" * 50)
    
    if model_name is None:
        active_info = get_active_model_info()
        model_name = active_info['name']
        model_path = active_info['path']
        print(f"Using active model: {model_name}")
    else:
        model_path = Path('app/models') / model_name
        if not model_path.exists():
            return {
                'status': 'error',
                'message': f"Model {model_name} not found"
            }
        print(f"Using model: {model_name}")
    
    # Load data
    new_data_path = Path('app/data/new_data/new_data.csv')
    if not new_data_path.exists():
        return {
            'status': 'error',
            'message': f"New data file not found at {new_data_path}"
        }
    
    df = pd.read_csv(new_data_path)
    df_labeled = df[df['Churn'].notna()].copy()
    
    if len(df_labeled) == 0:
        return {
            'status': 'error',
            'message': "No labeled data found"
        }
    
    # Get true labels
    y_true = df_labeled['Churn'].map({'Yes': 1, 'No': 0}).values
    
    # Load model
    with open(model_path, 'rb') as f:
        pipeline = cloudpickle.load(f)
    
    # Prepare data (remove Churn column, keep everything else)
    X_raw = df_labeled.drop('Churn', axis=1)
    
    # Apply transformations
    X = X_raw.copy()
    transformers = ['totalcharges_cleaner', 'feature_engineer', 'categorical_encoder', 'drop_redundant', 'numerical_scaler']
    
    for name in transformers:
        if name in pipeline.named_steps:
            X = pipeline.named_steps[name].transform(X)
    
    # Get model and expected columns
    model = pipeline.named_steps['model']
    expected_cols = model.feature_names_
    
    # Align columns
    if hasattr(X, 'columns'):
        # Add missing columns with 0
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0
        
        # Remove extra columns
        X = X[expected_cols]
    
    # Predict
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
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
    
    print("\n" + "-" * 40)
    print("TEST RESULTS")
    print("-" * 40)
    print(f"Samples: {metrics['test_samples']} (Churn: {metrics['churn_yes']}, No: {metrics['churn_no']})")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    
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


def compare_models_on_new_data(model_names: List[str]) -> Dict[str, Any]:
    """
    Compare multiple models on new labeled data
    """
    print("=" * 50)
    print("COMPARING MODELS ON NEW DATA")
    print("=" * 50)
    
    results = []
    
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
    
    results.sort(key=lambda x: x['recall'], reverse=True)
    
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


