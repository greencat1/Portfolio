import torch

# ============================================================
# CONFIGURATION FILE FOR TWO-STAGE ROAD SIGN DETECTOR
# ============================================================

# Path to trained YOLO model weights (first stage)
# YOLO detects 5 categories of signs
YOLO_WEIGHT_PATH = "../app/weights/best.pt"

# Paths to 5 separate ResNet models (second stage)
# Each ResNet refines classification WITHIN one category (e.g., 11 specific sign types)
# Key: category_id (0-4), Value: path to .pt weights file
RESNET_WEIGHTS = {
    0: "weights/resnet_cat0.pt",   # Category 0: warning signs
    1: "weights/resnet_cat1.pt",   # Category 1: regulatory signs  
    2: "weights/resnet_cat2.pt",   # Category 2: informational signs
    3: "weights/resnet_cat3.pt",   # Category 3: temporary signs
    4: "weights/resnet_cat4.pt",   # Category 4: guidance signs
}

# Human-readable names for the 5 high-level categories
CATEGORY_NAMES = {
    0: "warning",       # Warning signs (e.g., curve, crosswalk)
    1: "regulatory",    # Regulatory signs (e.g., stop, yield, speed limit)
    2: "informational", # Informational signs (e.g., parking, direction)
    3: "temporary",     # Temporary signs (e.g., construction, detour)
    4: "guidance",      # Guidance signs (e.g., highway exit, distance)
}

# Device configuration: Use CUDA GPU if available, otherwise fall back to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Confidence threshold for YOLO detections
# Only detections with confidence >= 0.5 will be kept
CONFIDENCE_THRESHOLD = 0.5