from ultralytics import YOLO
import torch
import torchvision.models as models
from .config import *

# ============================================================
# Model Loader for Two-Stage Road Sign Detector
# ============================================================

def load_models():
    """
    Load YOLO model for first-stage detection (5 categories).
    
    Returns:
        yolo: Loaded YOLO model instance
    """
    # Load YOLO model from weights file path defined in config
    yolo = YOLO(YOLO_WEIGHT_PATH)
    
    # Note: ResNet models (second stage) will be added here later
    # Each ResNet model will handle fine-grained classification
    # within its specific category (e.g., 11 sign types per category)
    
    return yolo