# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    churn_model_path: str = "app/models/full_churn_pipeline_cloud.pkl"  
    
    model_config = {
        'protected_namespaces': ('settings_',)
    }

settings = Settings()