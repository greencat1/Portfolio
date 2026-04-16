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
from catboost import CatBoostClassifier
import shutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from app.utils.logger import logger
from app.core.database import get_db


def retrain_on_new_data():
    """
    Incrementally retrains the model on newly labeled data.
    Reads data from SQLite database instead of CSV.
    """
    
    # File paths
    backup_path = Path('app/models/inc_churn_pipeline_backup_cloud.pkl')
    model_path = Path(settings.churn_model_path)
    retrained_model_path = Path('app/models/inc_churn_pipeline_retrained_cloud.pkl')
    
    # 1. Load labeled data from SQLite
    with get_db() as conn:
        df_labeled = pd.read_sql_query('''
            SELECT * FROM new_data 
            WHERE churn_label IS NOT NULL
        ''', conn)
    
    if len(df_labeled) == 0:
        return {
            "status": "error",
            "message": "No labeled data found for retraining. Please add Churn labels (Yes/No) first."
        }
    
    # 2. Convert Churn from Yes/No to 0/1
    df_labeled['Churn_numeric'] = df_labeled['churn_label'].map({'Yes': 1, 'No': 0})
    
    # 3. Prepare features (X) and target variable (y)
    feature_columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]
    
    X_new = df_labeled[feature_columns].copy()
    y_new = df_labeled['Churn_numeric'].values
    
    logger.info(f"Loaded {len(X_new)} labeled samples from database")
    
    # 4. Load existing model
    try:
        old_model = load_model()
        logger.info(f"Loaded existing model from {settings.churn_model_path}")
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to load model: {str(e)}"
        }
    
    # 5. Create backup of old model
    if model_path.exists():
        shutil.copy(model_path, backup_path)
        logger.info(f"Backup saved to {backup_path}")
    
    # 6. Extract CatBoost base model from pipeline
    catboost_model = None
    model_index = None
    
    for i, (name, transformer) in enumerate(old_model.named_steps.items()):
        if 'catboost' in name.lower() or 'model' in name.lower():
            catboost_model = transformer
            model_index = i
            break
    
    if catboost_model is None:
        last_step_name = list(old_model.named_steps.keys())[-1]
        catboost_model = old_model.named_steps[last_step_name]
        model_index = len(old_model.named_steps) - 1
        logger.info(f"Using last step '{last_step_name}' as model")
    
    if catboost_model is None:
        return {
            "status": "error",
            "message": "Could not find CatBoost model in pipeline"
        }
    
    # 7. Apply transformations to new data
    temp_pipeline = Pipeline(steps=list(old_model.named_steps.items())[:model_index])
    
    try:
        X_transformed = temp_pipeline.transform(X_new)
        
        if not hasattr(X_transformed, 'columns'):
            if hasattr(catboost_model, 'feature_names_'):
                expected_features = catboost_model.feature_names_
                X_transformed = pd.DataFrame(
                    X_transformed,
                    columns=expected_features[:X_transformed.shape[1]]
                )
        
        logger.info(f"Transformed {len(X_transformed)} samples for retraining")
        
        # 8. Continue training
        model_params = catboost_model.get_params()
        
        continued_model = CatBoostClassifier(
            iterations=model_params.get('iterations', 500),
            learning_rate=model_params.get('learning_rate', 0.03),
            depth=model_params.get('depth', 4),
            l2_leaf_reg=model_params.get('l2_leaf_reg', 50),
            min_data_in_leaf=model_params.get('min_data_in_leaf', 1),
            random_strength=model_params.get('random_strength', 5),
            class_weights=model_params.get('class_weights', [1, 5]),
            random_state=42,
            verbose=50
        )
        
        continued_model.fit(
            X_transformed,
            y_new,
            init_model=catboost_model,
            verbose=50
        )
        
        # 9. Update and save pipeline
        new_steps = list(old_model.named_steps.items())
        new_steps[model_index] = (new_steps[model_index][0], continued_model)
        new_pipeline = Pipeline(steps=new_steps)
        
        with open(retrained_model_path, 'wb') as f:
            cloudpickle.dump(new_pipeline, f)
        
        # 10. Calculate metrics
        y_pred = continued_model.predict(X_transformed)
        
        metrics = {
            "accuracy": float(accuracy_score(y_new, y_pred)),
            "precision": float(precision_score(y_new, y_pred, zero_division=0)),
            "recall": float(recall_score(y_new, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_new, y_pred, zero_division=0))
        }
        
        return {
            "status": "success",
            "message": f"Model incrementally retrained on {len(X_new)} labeled samples",
            "samples_used": len(X_new),
            "churn_yes": int((y_new == 1).sum()),
            "churn_no": int((y_new == 0).sum()),
            "metrics": metrics,
            "model_path": str(retrained_model_path),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Retraining failed: {str(e)}",
            "samples_attempted": len(X_new)
        }


def full_retrain_on_combined_data():
    """
    Fully retrains the model from scratch on all data (old + new)
    Reads old data from raw_data table, new labels from new_data table
    """
    
    # File paths
    backup_path = Path('app/models/full_churn_pipeline_backup_cloud.pkl')
    model_path = Path(settings.churn_model_path)
    retrained_model_path = Path('app/models/full_churn_pipeline_retrained_cloud.pkl')
    
    # 1. Load old data from raw_data table
    with get_db() as conn:
        df_old = pd.read_sql_query('SELECT * FROM raw_data', conn)
        logger.info(f"Loaded {len(df_old)} samples from raw_data table")
    
    # 2. Load new labeled data from new_data table
    with get_db() as conn:
        df_new_labeled = pd.read_sql_query('''
            SELECT * FROM new_data 
            WHERE churn_label IS NOT NULL
        ''', conn)
        logger.info(f"Loaded {len(df_new_labeled)} labeled samples from new_data table")
    
    if len(df_new_labeled) == 0:
        return {
            "status": "error",
            "message": "No labeled data found for retraining. Please add Churn labels (Yes/No) first."
        }
    
    # 3. Rename columns to match (churn_label -> Churn)
    df_new_labeled = df_new_labeled.rename(columns={'churn_label': 'Churn'})
    
    # 4. Combine data
    if len(df_old) > 0:
        # Keep only common columns
        common_cols = list(set(df_old.columns) & set(df_new_labeled.columns))
        
        if 'Churn' in common_cols:
            common_cols.remove('Churn')
        
        common_cols = ['Churn'] + common_cols
        
        df_old = df_old[common_cols]
        df_new_labeled = df_new_labeled[common_cols]
        
        df_combined = pd.concat([df_old, df_new_labeled], ignore_index=True)
        logger.info(f"Combined {len(df_old)} old + {len(df_new_labeled)} new = {len(df_combined)} total samples")
    else:
        df_combined = df_new_labeled
        logger.info(f"Using only {len(df_combined)} new samples")
    
    # 5. Prepare features and target
    exclude_cols = ['id', 'customer_id', 'Churn', 'prediction', 'probability', 'created_at', 'label_timestamp', 'churn_label']
    feature_columns = [col for col in df_combined.columns if col not in exclude_cols]
    
    X = df_combined[feature_columns].copy()
    df_combined['Churn'] = df_combined['Churn'].map({'Yes': 1, 'No': 0})
    y = df_combined['Churn'].values
    
    logger.info(f"Features: {len(feature_columns)} columns")
    logger.info(f"Target distribution: Yes={sum(y==1)}, No={sum(y==0)}")
    
    # 6. Create pipeline
    logger.info("Building pipeline...")
    
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
            class_weights=[1, 5],
            random_state=42,
            verbose=0
        ))
    ])
    
    # 7. Train the model
    logger.info("Training model...")
    try:
        pipeline.fit(X, y)
        logger.info("Model training completed successfully!")
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Model training failed: {str(e)}"
        }
    
    # 8. Evaluate
    y_pred = pipeline.predict(X)
    
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, zero_division=0)),
    }
    
    logger.info(f"Training metrics: {metrics}")
    
    # 9. Backup old model
    if model_path.exists():
        shutil.copy(model_path, backup_path)
        logger.info(f"Backup saved to {backup_path}")
    
    # 10. Save retrained model
    with open(retrained_model_path, 'wb') as f:
        cloudpickle.dump(pipeline, f)
    logger.info(f"Retrained model saved to {retrained_model_path}")
    
    return {
        "status": "success",
        "message": f"Model fully retrained on {len(X)} total samples",
        "samples_used": len(X),
        "old_samples": len(df_old) if len(df_old) > 0 else 0,
        "new_labeled_samples": len(df_new_labeled),
        "churn_yes": int(sum(y == 1)),
        "churn_no": int(sum(y == 0)),
        "metrics": metrics,
        "model_path": str(retrained_model_path),
        "timestamp": datetime.now().isoformat()
    }


def get_retraining_status():
    """
    Gets information about available data for retraining from SQLite
    """
    with get_db() as conn:
        # Get total records from new_data
        total = conn.execute("SELECT COUNT(*) FROM new_data").fetchone()[0]
        
        # Get labeled records
        labeled = conn.execute("SELECT COUNT(*) FROM new_data WHERE churn_label IS NOT NULL").fetchone()[0]
        
        if labeled > 0:
            churn_yes = conn.execute("SELECT COUNT(*) FROM new_data WHERE churn_label = 'Yes'").fetchone()[0]
            churn_no = conn.execute("SELECT COUNT(*) FROM new_data WHERE churn_label = 'No'").fetchone()[0]
        else:
            churn_yes = 0
            churn_no = 0
    
    if labeled == 0:
        return {
            "status": "warning",
            "message": "No labeled data available for retraining",
            "total_records": total,
            "labeled_records": 0,
            "churn_yes": 0,
            "churn_no": 0,
            "can_retrain": False
        }
    
    return {
        "status": "ready",
        "total_records": total,
        "labeled_records": labeled,
        "churn_yes": churn_yes,
        "churn_no": churn_no,
        "can_retrain": labeled >= 10,
        "message": f"Ready to retrain on {labeled} labeled samples" if labeled >= 10 else f"Need at least 10 labeled samples, currently have {labeled}"
    }