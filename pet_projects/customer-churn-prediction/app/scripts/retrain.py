# app/scripts/retrain.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import cloudpickle
from sklearn.pipeline import Pipeline
from app.scripts.model import load_model
from app.config import settings
from app.scripts.transformers import (
    TotalChargesCleaner,
    FeatureEngineer,
    CategoricalEncoder,
    DropRedundant,
    NumericalScaler
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import shutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def retrain_on_new_data():
    """
    Retrains the model on newly labeled data from data/new_data/new_data.csv
    Saves the updated model to models/full_churn_pipeline_retrained.pkl
    """
    
    # File paths
    new_data_path = Path('app/data/new_data/new_data.csv')
    backup_path = Path('app/models/inc_churn_pipeline_backup_cloud.pkl')
    model_path = Path(settings.churn_model_path)
    retrained_model_path = Path('app/models/inc_churn_pipeline_retrained_cloud.pkl')
    
    # 1. Check if new data file exists
    if not new_data_path.exists():
        return {
            "status": "error",
            "message": f"New data file not found at {new_data_path}"
        }
    
    # 2. Load new data with labels
    df_new = pd.read_csv(new_data_path)
    
    # 3. Filter only labeled data (Churn is not empty)
    df_labeled = df_new[df_new['Churn'].notna()].copy()
    
    if len(df_labeled) == 0:
        return {
            "status": "error",
            "message": "No labeled data found for retraining. Please add Churn labels (Yes/No) first."
        }
    
    # 4. Convert Churn from Yes/No to 0/1
    df_labeled['Churn_numeric'] = df_labeled['Churn'].map({'Yes': 1, 'No': 0})
    
    # 5. Prepare features (X) and target variable (y)
    feature_columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]
    
    X_new = df_labeled[feature_columns].copy()
    y_new = df_labeled['Churn_numeric'].values
    
    # 6. Load existing model
    try:
        old_model = load_model()
        print(f"Loaded existing model from {settings.churn_model_path}")
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load model: {str(e)}"
        }
    
    # 7. Create backup of old model
    import shutil
    if model_path.exists():
        shutil.copy(model_path, backup_path)
        print(f"Backup saved to {backup_path}")
    
    # 8. Extract CatBoost base model from pipeline
    # The pipeline consists of transformers and a model at the end
    # Need to find the CatBoost model inside the pipeline
    
    catboost_model = None
    model_index = None
    
    # Search for CatBoost model in the pipeline
    for i, (name, transformer) in enumerate(old_model.named_steps.items()):
        if 'catboost' in name.lower() or 'model' in name.lower():
            catboost_model = transformer
            model_index = i
            break
    
    # If not found by name, take the last step (usually the model)
    if catboost_model is None:
        # Take the last step of the pipeline as the model
        last_step_name = list(old_model.named_steps.keys())[-1]
        catboost_model = old_model.named_steps[last_step_name]
        model_index = len(old_model.named_steps) - 1
        print(f"Using last step '{last_step_name}' as model")
    
    if catboost_model is None:
        return {
            "status": "error",
            "message": "Could not find CatBoost model in pipeline"
        }
    
    # 9. Apply all transformations to new data (excluding the final model)
    # Create a temporary pipeline without the model
    temp_pipeline = Pipeline(steps=list(old_model.named_steps.items())[:model_index])
    
    try:
        # Transform new data
        X_transformed = temp_pipeline.transform(X_new)
        print(f"Transformed {len(X_transformed)} samples for retraining")
        
        # 10. Retrain the model
        # For CatBoost, use fit with init_model to continue training
        print("Starting incremental training...")
        
        # Get current model parameters
        current_params = catboost_model.get_params()
        
        # Important: to continue training, need to pass init_model
        # and set warm_start=True or use fit with init_model
        catboost_model.fit(
            X_transformed, 
            y_new,
            init_model=catboost_model,  # Continue training from current model
            verbose=50
        )
        
        print(f"Model retrained successfully on {len(X_new)} samples")
        
        # 11. Update pipeline with the new model
        new_steps = list(old_model.named_steps.items())
        new_steps[model_index] = (new_steps[model_index][0], catboost_model)
        new_pipeline = Pipeline(steps=new_steps)
        
        # 12. Save the updated model
        with open(retrained_model_path, 'wb') as f:
            cloudpickle.dump(new_pipeline, f)
        print(f"Retrained model saved to {retrained_model_path}")
        
        # 13. Update config paths (optional)
        # Can replace old model with new one
        # shutil.copy(retrained_model_path, model_path)
        
        # 14. Get retraining statistics
        
        # Make predictions on the same data for evaluation
        y_pred = catboost_model.predict(X_transformed)
        y_proba = catboost_model.predict_proba(X_transformed)[:, 1]
        
        metrics = {
            "accuracy": float(accuracy_score(y_new, y_pred)),
            "precision": float(precision_score(y_new, y_pred, zero_division=0)),
            "recall": float(recall_score(y_new, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_new, y_pred, zero_division=0))
        }
        
        return {
            "status": "success",
            "message": f"Model retrained on {len(X_new)} labeled samples",
            "samples_used": len(X_new),
            "churn_yes": int((y_new == 1).sum()),
            "churn_no": int((y_new == 0).sum()),
            "metrics": metrics,
            "model_path": str(retrained_model_path),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Retraining failed: {str(e)}",
            "samples_attempted": len(X_new)
        }




def full_retrain_on_combined_data():
    """
    Fully retrains the model from scratch on all data (old + new)
    Saves the retrained model to models/full_churn_pipeline_retrained.pkl
    """
    
    
    # File paths
    old_data_path = Path('app/data/raw/telco_churn.csv')
    new_data_path = Path('app/data/new_data/new_data.csv')
    backup_path = Path('app/models/full_churn_pipeline_backup_cloud.pkl')
    model_path = Path(settings.churn_model_path)
    retrained_model_path = Path('app/models/full_churn_pipeline_retrained_cloud.pkl')
    
    # 1. Load and combine data
    print("Loading data...")
    
    # Load old data
    if old_data_path.exists():
        df_old = pd.read_csv(old_data_path)
        print(f"Loaded {len(df_old)} samples from old data")
    else:
        df_old = None
        print(f"Warning: Old data not found at {old_data_path}")
    
    # Load new labeled data
    if not new_data_path.exists():
        return {
            "status": "error",
            "message": f"New data file not found at {new_data_path}"
        }
    
    df_new = pd.read_csv(new_data_path)
    
    # Filter labeled new data (Churn is not empty)
    df_new_labeled = df_new[df_new['Churn'].notna()].copy()
    
    if len(df_new_labeled) == 0:
        return {
            "status": "error",
            "message": "No labeled data found for retraining. Please add Churn labels (Yes/No) first."
        }
    
    
    
    # Combine data
    if df_old is not None:
        # Ensure both dataframes have the same columns
        # Keep only common columns
        common_cols = list(set(df_old.columns) & set(df_new_labeled.columns))
        
        # If 'Churn' is in common_cols, make sure it's the target
        if 'Churn' in common_cols:
            common_cols.remove('Churn')
        
        # Add 'Churn' back as target
        common_cols = ['Churn'] + common_cols
        
        df_old = df_old[common_cols]
        df_new_labeled = df_new_labeled[common_cols]
        
        df_combined = pd.concat([df_old, df_new_labeled], ignore_index=True)
        print(f"Combined {len(df_old)} old + {len(df_new_labeled)} new = {len(df_combined)} total samples")
    else:
        df_combined = df_new_labeled
        print(f"Using only {len(df_combined)} new samples")
    
    # 2. Prepare features and target
    # Define feature columns (exclude customerID and Churn)
    exclude_cols = ['customerID', 'Churn', 'prediction', 'probability', 'timestamp', 'label_timestamp']
    feature_columns = [col for col in df_combined.columns if col not in exclude_cols]
    
    
    
    X = df_combined[feature_columns].copy()
    # Convert Churn from Yes/No to 0/1
    df_combined['Churn'] = df_combined['Churn'].map({'Yes': 1, 'No': 0})
    y = df_combined['Churn'].values
    
    print(f"Features: {len(feature_columns)} columns")
    print(f"Target distribution: Yes={sum(y==1)}, No={sum(y==0)}")

    
    
    # 4. Create the full pipeline
    print("Building pipeline...")
    
    pipeline = Pipeline([
        ('totalcharges_cleaner', TotalChargesCleaner()),
        ('feature_engineer', FeatureEngineer()),
        ('categorical_encoder', CategoricalEncoder()),
        ('drop_redundant', DropRedundant()),
        ('numerical_scaler', NumericalScaler()),
        ('model', CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=50,
        min_data_in_leaf=1,
        random_strength=5,
        class_weights=[1, 5],          # Best weight from tuning
        random_state=42,
        verbose=0
    ))
    ])
    
    # 5. Train the model
    print("Training model...")
    try:
        pipeline.fit(X, y)
        print("Model training completed successfully!")
    except Exception as e:
        return {
            "status": "error",
            "message": f"Model training failed: {str(e)}"
        }
    
    # 6. Evaluate on training data
    
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]
    
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, zero_division=0)),
        
    }
    
    print(f"Training metrics: {metrics}")
    
    # 7. Create backup of old model
    if model_path.exists():
        shutil.copy(model_path, backup_path)
        print(f"Backup saved to {backup_path}")
    
    # 8. Save the retrained model
    with open(retrained_model_path, 'wb') as f:
        cloudpickle.dump(pipeline, f)
    print(f"Retrained model saved to {retrained_model_path}")
    
    
    
    return {
        "status": "success",
        "message": f"Model fully retrained on {len(X)} total samples",
        "samples_used": len(X),
        "old_samples": len(df_old) if df_old is not None else 0,
        "new_labeled_samples": len(df_new_labeled),
        "churn_yes": int(sum(y == 1)),
        "churn_no": int(sum(y == 0)),
        "metrics": metrics,
        "model_path": str(retrained_model_path),
        "timestamp": datetime.now().isoformat()
    }


# For API usage
def get_retraining_status():
    """
    Gets information about available data for retraining
    
    Returns:
        Dictionary with retraining status and data statistics
    """
    new_data_path = Path('app/data/new_data/new_data.csv')
    
    if not new_data_path.exists():
        return {
            "status": "error",
            "message": "No new data file found"
        }
    
    df = pd.read_csv(new_data_path)
    labeled = df[df['Churn'].notna()]
    
    if len(labeled) == 0:
        return {
            "status": "warning",
            "message": "No labeled data available for retraining",
            "total_records": len(df),
            "labeled_records": 0
        }
    
    churn_yes = len(labeled[labeled['Churn'] == 'Yes'])
    churn_no = len(labeled[labeled['Churn'] == 'No'])
    
    return {
        "status": "ready",
        "total_records": len(df),
        "labeled_records": len(labeled),
        "churn_yes": churn_yes,
        "churn_no": churn_no,
        "can_retrain": len(labeled) >= 10,  # Minimum 10 labeled records required
        "message": f"Ready to retrain on {len(labeled)} labeled samples" if len(labeled) >= 10 else f"Need at least 10 labeled samples, currently have {len(labeled)}"
    }