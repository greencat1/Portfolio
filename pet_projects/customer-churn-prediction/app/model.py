import joblib
import os
from app.config import settings
from app.transformers import *

import cloudpickle


_model = None

def load_model():
    global _model
    if _model is None:
        if not os.path.exists(settings.churn_model_path):
            raise FileNotFoundError(f"Model not found at {settings.churn_model_path}")
        print(f"Loading model from {settings.churn_model_path}")
        with open(settings.churn_model_path, 'rb') as f:
            _model = cloudpickle.load(f)
        print("Model loaded successfully!")
    return _model