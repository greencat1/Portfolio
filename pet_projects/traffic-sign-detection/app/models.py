from ultralytics import YOLO
import torch
import torchvision.models as models
from .config import *
import asyncio
from functools import partial
import torch
from ultralytics import YOLO
from typing import List, Tuple, Dict
import torch.nn as nn
from pathlib import Path
import torchvision.models as models
import numpy as np
# Confidence threshold for YOLO detections
CONFIDENCE_THRESHOLD = 0.5
import cv2

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


def load_trained_resnets(resnet_dir: Path, category_signs: Dict[int, List[str]], device: torch.device='cpu') -> Dict[int, nn.Module]:
    """
    Load trained ResNet models for inference.
    
    Args:
        resnet_dir: Directory containing saved model weights
        category_signs: Dictionary mapping category_id to ordered list of sign names
        device: Device to load models on
    
    Returns:
        Dictionary of loaded ResNet models
    """
    resnets = {}
    
    for category_id in range(5):
        model_path = resnet_dir + f"/best_resnet_cat_{category_id}.pt"
        
        
        
        # Number of classes = number of signs in category + 1 (background)
        num_classes = len(category_signs.get(category_id, [])) + 1
        # Create model and load weights
        model = create_resnet_model(num_classes, freeze_backbone=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        resnets[category_id] = model
        print(f" Loaded ResNet for category {category_id} ({num_classes-1} sign classes)")
    
    return resnets

def load_all_models(best_model_path="../models/detect/cascade_model/weights/best.pt",
                   resnet_dir='../models/resnet_weights/prod',
                   device='cpu'):
    
    category_to_signs = {
        1: sorted([41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]),  # warn (YOLO cat 1)
        0: sorted([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),  # forb (YOLO cat 0)
        3: sorted([21, 22, 23, 24, 25, 26]),  # info (YOLO cat 3)
        2: sorted([27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]),  # mand (YOLO cat 2)
        4: sorted([38, 39, 40])  # prio (YOLO cat 4)
    }
    
    yolo = YOLO(str(best_model_path)).to(device)
    resnets = load_trained_resnets(resnet_dir=resnet_dir, category_signs=category_to_signs, device=device)
    
    return yolo, resnets

def create_resnet_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    Create a ResNet18 model with custom FC layer for given number of classes.
    
    Args:
        num_classes: Number of classes (signs + 1 background)
        freeze_backbone: If True, freeze all layers except the final FC layer
    
    Returns:
        ResNet18 model ready for training
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze backbone if requested
    if freeze_backbone:
        # Freeze all parameters initially
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the final FC layer only
        # But FC layer doesn't exist yet, so we'll freeze after replacing
        # Actually: freeze all first, then replace FC (which will be trainable by default)
        # Let's do it cleanly:
        
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    # FC layer is trainable by default (requires_grad = True)
    # If backbone was frozen, only FC will be trained
    
    return model

