import asyncio
from functools import partial

# Confidence threshold for YOLO detections
CONFIDENCE_THRESHOLD = 0.5


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

