"""
Model Loading Module

WHAT THIS MODULE DOES:
Provides a singleton pattern for loading and accessing the ML model.
Ensures the model is loaded only once and reused across all requests.

WHY SINGLETON PATTERN?
- ML models are large (often 100MB+) and slow to load (1-3 seconds)
- Loading once and reusing saves memory and startup time
- Without singleton, each request would reload the model (very slow!)

HOW IT WORKS:
1. First call to load_model() loads the model from disk
2. Subsequent calls return the cached model from memory
3. Model stays in memory until server restarts

ALTERNATIVE APPROACHES:
- Load at startup (FastAPI startup event) - also valid
- This module's approach: lazy loading (load on first use)

MEMORY IMPACT:
- Model stays in RAM permanently after first load
- For a 100MB model, that's 100MB of RAM used
- Worth it for the performance gain
"""

import joblib
import os
from app.config import settings
from app.scripts.transformers import *  # Imports custom transformers needed for model
from app.utils.logger import logger

import cloudpickle


# ============================================================================
# GLOBAL MODEL CACHE (Singleton Pattern)
# ============================================================================

# Private module variable to hold the loaded model
# Starts as None (not loaded)
# After first load, stores the model object
# This variable is shared across all imports of this module
_model = None


# ============================================================================
# MODEL LOADING FUNCTION
# ============================================================================

def load_model():
    """
    Load the ML model from disk (singleton pattern).
    
    LOADING STRATEGY (Lazy Loading):
    - Model loads ONLY on first call
    - Subsequent calls return cached model
    - This prevents loading on server start if model never used
    
    WHEN MODEL IS LOADED:
    - First prediction request
    - Or any endpoint that needs predictions
    
    FILE FORMAT:
    Uses cloudpickle (not standard pickle) because:
    - cloudpickle can serialize lambda functions
    - cloudpickle can serialize functions defined in __main__
    - Better compatibility with scikit-learn pipelines
    
    MODEL PATH:
    Retrieved from settings.churn_model_path
    Default: "app/models/full_churn_pipeline_cloud.pkl"
    
    WHAT'S INSIDE THE MODEL FILE:
    Complete scikit-learn pipeline including:
    - TotalChargesCleaner (handles empty strings)
    - FeatureEngineer (creates derived features)
    - CategoricalEncoder (OneHotEncoder for categoricals)
    - DropRedundantFeatures (removes unnecessary columns)
    - NumericalScaler (StandardScaler for numeric columns)
    - RandomForestClassifier (the actual ML model)
    
    RETURNS:
        Loaded sklearn Pipeline object
    
    RAISES:
        FileNotFoundError: If model file doesn't exist at specified path
    
    EXAMPLE:
        from app.scripts.predict import load_model
        model = load_model()  # First call - loads from disk
        predictions = model.predict(X_test)
        
        model2 = load_model()  # Second call - returns cached model
        assert model is model2  # Same object in memory!
    """
    global _model  # We're modifying the global variable
    
    # ============================================
    # STEP 1: Check if already loaded (cache hit)
    # ============================================
    if _model is None:
        # ========================================
        # STEP 2: Verify model file exists
        # ========================================
        if not os.path.exists(settings.churn_model_path):
            raise FileNotFoundError(f"Model not found at {settings.churn_model_path}")
        
        # ========================================
        # STEP 3: Log loading attempt
        # ========================================
        logger.info(f"Loading model from {settings.churn_model_path}")
        
        # ========================================
        # STEP 4: Load model from disk
        # ========================================
        # Use cloudpickle (not joblib) because:
        # - Handles complex objects better
        # - Works with custom transformers
        # - Supports lambda functions
        with open(settings.churn_model_path, 'rb') as f:
            _model = cloudpickle.load(f)
        
        # ========================================
        # STEP 5: Log success
        # ========================================
        logger.info("Model loaded successfully!")
    
    # Return cached model (either just loaded or from previous call)
    return _model


