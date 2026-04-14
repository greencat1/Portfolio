# app/scripts/model_manager.py
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

from app.config import settings
from app.model import _model, load_model

# Models directory
MODELS_DIR = Path('models')
EXCLUDE_MODELS = ['full_churn_pipeline.pkl']  # Models to exclude from list


def ensure_models_dir():
    """Ensure models directory exists"""
    MODELS_DIR.mkdir(exist_ok=True)


def get_all_models() -> List[Dict[str, Any]]:
    """
    Get list of all models in models directory (except excluded ones)
    """
    ensure_models_dir()
    
    models = []
    current_active_path = settings.churn_model_path
    
    # Look for .pkl files
    for model_path in MODELS_DIR.glob('*.pkl'):
        model_name = model_path.name
        
        # Skip excluded models
        if model_name in EXCLUDE_MODELS:
            continue
        
        # Get file stats
        stat = model_path.stat()
        size_mb = stat.st_size / (1024 * 1024)
        created_at = datetime.fromtimestamp(stat.st_ctime).isoformat()
        
        # Check if this is the active model
        is_active = str(model_path) == current_active_path or model_path.name == Path(current_active_path).name
        
        # Try to load metrics if available
        metrics = None
        metrics_path = MODELS_DIR / f'{model_path.stem}_metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        
        models.append({
            'name': model_name,
            'path': str(model_path),
            'size_mb': round(size_mb, 2),
            'created_at': created_at,
            'is_active': is_active,
            'metrics': metrics
        })
    
    # Sort by creation date (newest first)
    models.sort(key=lambda x: x['created_at'], reverse=True)
    
    return models


def get_active_model_info() -> Dict[str, Any]:
    """
    Get currently active model information
    """
    active_path = settings.churn_model_path
    active_name = Path(active_path).name
    
    # Get file stats
    model_path = Path(active_path)
    if not model_path.exists():
        return {
            'name': active_name,
            'path': active_path,
            'exists': False,
            'error': 'Model file not found'
        }
    
    stat = model_path.stat()
    size_mb = stat.st_size / (1024 * 1024)
    created_at = datetime.fromtimestamp(stat.st_ctime).isoformat()
    
    return {
        'name': active_name,
        'path': active_path,
        'size_mb': round(size_mb, 2),
        'created_at': created_at,
        'exists': True
    }


def switch_active_model(model_name: str) -> Dict[str, Any]:
    """
    Switch active model by filename
    """
    model_path = MODELS_DIR / model_name
    
    if not model_path.exists():
        return {
            'status': 'error',
            'message': f'Model {model_name} not found in {MODELS_DIR}'
        }
    
    # Get previous active model
    previous_model = Path(settings.churn_model_path).name
    
    # Update settings path
    settings.churn_model_path = str(model_path)
    
    # Clear cached model to force reload on next prediction
    global _model
    _model = None
    
    return {
        'status': 'success',
        'message': f'Active model switched to {model_name}',
        'previous_model': previous_model,
        'current_model': model_name,
        'model_path': str(model_path)
    }


def delete_model(model_name: str, force: bool = False) -> Dict[str, Any]:
    """
    Delete a model file
    """
    model_path = MODELS_DIR / model_name
    current_active = Path(settings.churn_model_path).name
    
    # Prevent deleting active model
    if model_name == current_active and not force:
        return {
            'status': 'error',
            'message': f'Cannot delete active model "{model_name}". Switch to another model first or use force=True'
        }
    
    # Prevent deleting excluded models
    if model_name in EXCLUDE_MODELS:
        return {
            'status': 'error',
            'message': f'Cannot delete protected model "{model_name}"'
        }
    
    if not model_path.exists():
        return {
            'status': 'error',
            'message': f'Model {model_name} not found'
        }
    
    # Delete model file
    model_path.unlink()
    
    # Delete metrics file if exists
    metrics_path = MODELS_DIR / f'{Path(model_name).stem}_metrics.json'
    if metrics_path.exists():
        metrics_path.unlink()
    
    return {
        'status': 'success',
        'message': f'Model {model_name} deleted successfully',
        'deleted_model': model_name
    }


def save_model_metrics(model_name: str, metrics: Dict[str, Any]):
    """
    Save metrics for a model
    """
    metrics_path = MODELS_DIR / f'{Path(model_name).stem}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)


def get_model_metrics(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get metrics for a specific model from models/metrics directory
    
    Args:
        model_name: Name of the model file (e.g., 'full_churn_pipeline_cloud.pkl')
    
    Returns:
        Dictionary with metrics or None if not found
    """
    import json
    from pathlib import Path
    
    model_stem = Path(model_name).stem
    
    # Search in models/metrics directory
    metrics_dir = MODELS_DIR / 'metrics'
    
    if not metrics_dir.exists():
        print(f"Metrics directory not found: {metrics_dir}")
        return None
    
    # Try different naming patterns
    patterns = [
        f'{model_stem}_metrics.json',
        f'{model_stem}_test_metrics.json',
        f'{model_stem}_training_metrics.json',
        f'*{model_stem}*metrics*.json',
        f'*{model_stem}*.json',
    ]
    
    for pattern in patterns:
        for metrics_file in metrics_dir.glob(pattern):
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    print(f"Found metrics for {model_name} at {metrics_file}")
                    return metrics
            except Exception as e:
                print(f"Error reading {metrics_file}: {e}")
                continue
    
    print(f"No metrics found for model {model_name} in {metrics_dir}")
    return None


def compare_models(model_name1: str, model_name2: str) -> Dict[str, Any]:
    """
    Compare two models
    """
    model1_path = MODELS_DIR / model_name1
    model2_path = MODELS_DIR / model_name2
    
    if not model1_path.exists():
        return {'status': 'error', 'message': f'Model {model_name1} not found'}
    if not model2_path.exists():
        return {'status': 'error', 'message': f'Model {model_name2} not found'}
    
    stat1 = model1_path.stat()
    stat2 = model2_path.stat()
    
    metrics1 = get_model_metrics(model_name1)
    metrics2 = get_model_metrics(model_name2)
    
    # Calculate comparison metrics
    comparison = {
        'size_diff_mb': round(stat1.st_size / (1024 * 1024) - stat2.st_size / (1024 * 1024), 2),
        'newer_model': model_name1 if stat1.st_ctime > stat2.st_ctime else model_name2,
        'created_at_diff': datetime.fromtimestamp(stat1.st_ctime).isoformat() if stat1.st_ctime > stat2.st_ctime else datetime.fromtimestamp(stat2.st_ctime).isoformat()
    }
    
    # Add metric comparisons if both have metrics
    if metrics1 and metrics2 and 'metrics' in metrics1 and 'metrics' in metrics2:
        m1 = metrics1.get('metrics', {})
        m2 = metrics2.get('metrics', {})
        
        if 'accuracy' in m1 and 'accuracy' in m2:
            comparison['accuracy_diff'] = round(m2['accuracy'] - m1['accuracy'], 4)
        
        if 'f1_score' in m1 and 'f1_score' in m2:
            comparison['f1_diff'] = round(m2['f1_score'] - m1['f1_score'], 4)
        
        if 'roc_auc' in m1 and 'roc_auc' in m2:
            comparison['roc_auc_diff'] = round(m2['roc_auc'] - m1['roc_auc'], 4)
    
    return {
        'status': 'success',
        'model1': {
            'name': model_name1,
            'size_mb': round(stat1.st_size / (1024 * 1024), 2),
            'created_at': datetime.fromtimestamp(stat1.st_ctime).isoformat(),
            'is_active': model_name1 == Path(settings.churn_model_path).name,
            'metrics': metrics1
        },
        'model2': {
            'name': model_name2,
            'size_mb': round(stat2.st_size / (1024 * 1024), 2),
            'created_at': datetime.fromtimestamp(stat2.st_ctime).isoformat(),
            'is_active': model_name2 == Path(settings.churn_model_path).name,
            'metrics': metrics2
        },
        'comparison': comparison  # Add this field
    }