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


LOCAL_TO_GLOBAL_MAPPING = {
    0: {  # YOLO detected forb (0) -> GT category 1
        0: (0, 'forb_ahead'),
        1: (1, 'forb_left'),
        2: (2, 'forb_overtake'),
        3: (3, 'forb_right'),
        4: (4, 'forb_speed_over_10'),
        5: (5, 'forb_speed_over_100'),
        6: (6, 'forb_speed_over_130'),
        7: (7, 'forb_speed_over_20'),
        8: (8, 'forb_speed_over_30'),
        9: (9, 'forb_speed_over_40'),
        10: (10, 'forb_speed_over_5'),
        11: (11, 'forb_speed_over_50'),
        12: (12, 'forb_speed_over_60'),
        13: (13, 'forb_speed_over_70'),
        14: (14, 'forb_speed_over_80'),
        15: (15, 'forb_speed_over_90'),
        16: (16, 'forb_stopping'),
        17: (17, 'forb_trucks'),
        18: (18, 'forb_u_turn'),
        19: (19, 'forb_weight_over_3.5t'),
        20: (20, 'forb_weight_over_7.5t'),
    },
    1: {  # YOLO detected warn (1) -> GT category 0
        0: (41, 'warn_children'),
        1: (42, 'warn_construction'),
        2: (43, 'warn_crosswalk'),
        3: (44, 'warn_cyclists'),
        4: (45, 'warn_domestic_animals'),
        5: (46, 'warn_other_dangers'),
        6: (47, 'warn_poor_road_surface'),
        7: (48, 'warn_roundabout'),
        8: (49, 'warn_slippery_road'),
        9: (50, 'warn_speed_bumper'),
        10: (51, 'warn_traffic_light'),
        11: (52, 'warn_tram'),
        12: (53, 'warn_two_way_traffic'),
        13: (54, 'warn_wild_animals'),
    },
    2: {  # YOLO detected mand (2) -> GT category 3
        0: (27, 'mand_bike_lane'),
        1: (28, 'mand_left'),
        2: (29, 'mand_left_right'),
        3: (30, 'mand_pass_left'),
        4: (31, 'mand_pass_left_right'),
        5: (32, 'mand_pass_right'),
        6: (33, 'mand_right'),
        7: (34, 'mand_roundabout'),
        8: (35, 'mand_straigh_left'),
        9: (36, 'mand_straight'),
        10: (37, 'mand_straight_right'),
    },
    3: {  # YOLO detected info (3) -> GT category 2
        0: (21, 'info_bus_station'),
        1: (22, 'info_crosswalk'),
        2: (23, 'info_highway'),
        3: (24, 'info_one_way_traffic'),
        4: (25, 'info_parking'),
        5: (26, 'info_taxi_parking'),
    },
    4: {  # YOLO detected other (4) -> GT category 4
        0: (38, 'prio_give_way'),
        1: (39, 'prio_priority_road'),
        2: (40, 'prio_stop'),
    },
}

def predict_yolo_sync(yolo, img):
    """
    Synchronous YOLO inference - returns list of detections as dictionaries.
    
    Args:
        yolo: Loaded YOLO model instance (None for testing with placeholder)
        img: Image as numpy array (BGR format from OpenCV)
    
    Returns:
        List of dictionaries with keys: bbox, class_id, confidence, class_name
    """
    
    # Placeholder for testing without real model
    if yolo is None:
        h, w = img.shape[:2]
        return [{
            "bbox": [w//4, h//4, 3*w//4, 3*h//4],
            "class_id": 0,
            "confidence": 0.9,
            "class_name": "test_sign"
        }]
    
    # Actual YOLO inference
    results = yolo(img, conf=CONFIDENCE_THRESHOLD)
    
    # Handle empty results
    if len(results) == 0 or results[0].boxes is None:
        return []
    
    # Convert YOLO results to list of dictionaries
    detections = []
    for box in results[0].boxes:
        detections.append({
            "bbox": box.xyxy[0].tolist(),           # [x1, y1, x2, y2]
            "class_id": int(box.cls[0]),            # Category ID (0-4)
            "confidence": float(box.conf[0]),       # Detection confidence
            "class_name": results[0].names[int(box.cls[0])]  # Human-readable name
        })
    
    return detections





def extract_crop(img, bbox):
    """
    Extract cropped region from image using bounding box coordinates.
    
    Args:
        img: Source image as numpy array
        bbox: Bounding box coordinates [x1, y1, x2, y2]
    
    Returns:
        Cropped image region as numpy array, None if invalid
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = img.shape[:2]
    
    # Clamp coordinates to image boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # Check for valid crop
    if x2 <= x1 or y2 <= y1:
        return None
    
    return img[y1:y2, x1:x2]


def format_detections(boxes, refined_classes):
    """
    Format detection results into a clean dictionary list.
    
    Args:
        boxes: YOLO boxes object with detection info
        refined_classes: List of refined class IDs from ResNet models
    
    Returns:
        List of formatted detection dictionaries
    """
    detections = []
    for idx, box in enumerate(boxes):
        detections.append({
            "bbox": box.xyxy[0].tolist(),
            "category_id": int(box.cls[0]),
            "class_id": refined_classes[idx] if idx < len(refined_classes) else int(box.cls[0]),
            "confidence": float(box.conf[0]),
            "class_name": f"class_{refined_classes[idx]}" if idx < len(refined_classes) else f"cat_{int(box.cls[0])}"
        })
    return detections

def predict_signs(
    yolo: YOLO,
    resnets: Dict[int, nn.Module],
    image: np.ndarray,
    device: torch.device = 'cpu',
    threshold: float = 0.5) -> List[Dict]:
    """
    Two-stage detection on CPU/GPU.
    """
    
    img_h, img_w = image.shape[:2]
    
    # Normalization constants
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Stage 1: YOLO detection
    results = yolo(image, conf=threshold, verbose=False)[0]
    
    if results.boxes is None:
        return []
    
    detections = []
    
    for box in results.boxes:
        # YOLO Ultralytics returns PIXEL coordinates
        x1_pixel, y1_pixel, x2_pixel, y2_pixel = map(int, box.xyxy[0].tolist())
        
        # Normalize to 0-1 range for comparison with ground truth
        x1 = x1_pixel / img_w
        y1 = y1_pixel / img_h
        x2 = x2_pixel / img_w
        y2 = y2_pixel / img_h
        
        # YOLO category (now matches GT category after renaming)
        category = int(box.cls[0])
        yolo_conf = float(box.conf[0])
        
        # Extract crop using PIXEL coordinates
        crop = image[y1_pixel:y2_pixel, x1_pixel:x2_pixel]
        if crop.size == 0:
            continue
        
        # Get ResNet for this category
        resnet = resnets.get(category)
        if resnet is None:
            continue
        
        # Prepare crop for ResNet
        crop = cv2.resize(crop, (224, 224))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        crop_tensor = (crop_tensor - IMAGENET_MEAN) / IMAGENET_STD
        crop_tensor = crop_tensor.unsqueeze(0).to(device)
        
        # Stage 2: ResNet inference
        resnet = resnet.to(device)
        resnet.eval()
        
        with torch.no_grad():
            outputs = resnet(crop_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, local_class = torch.max(probs, dim=1)
        
        local_class = local_class.item()
        confidence = confidence.item()
        
        # Check if local_class exists in mapping
        if category not in LOCAL_TO_GLOBAL_MAPPING:
            continue
        if local_class not in LOCAL_TO_GLOBAL_MAPPING[category]:
            continue
        
        class_number, class_name = LOCAL_TO_GLOBAL_MAPPING[category][local_class]
        
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "class_number": class_number,
            "class_name": class_name,
            "confidence": confidence,
            "yolo_confidence": yolo_conf
        })
    
    return detections

