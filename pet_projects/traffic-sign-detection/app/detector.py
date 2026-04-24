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


def predict_resnet_sync(model, crop):
    """
    Synchronous ResNet inference for fine-grained classification.
    
    Args:
        model: Loaded ResNet model for specific category
        crop: Cropped image region containing the sign
    
    Returns:
        int: Refined class ID within the category (e.g., 0-10 for 11 specific signs)
    """
    # TODO: Add preprocessing (resize to 224x224, normalize, convert to tensor)
    # TODO: Run model inference
    # TODO: Return predicted class ID
    return 0  # Placeholder - implement with actual model logic


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


async def detect_two_stage(yolo, resnets, img):
    """
    Two-stage detection pipeline:
    1. YOLO detects signs and predicts 5 categories (Stage 1)
    2. Category-specific ResNet refines to precise sign class (Stage 2)
    
    Args:
        yolo: Loaded YOLO model (first stage)
        resnets: Dictionary of 5 ResNet models (second stage), keyed by category ID
        img: Input image as numpy array
    
    Returns:
        List of detections with final refined class IDs
    """
    loop = asyncio.get_event_loop()
    
    # Stage 1: YOLO inference (CPU/GPU bound, run in thread pool)
    yolo_detections = await loop.run_in_executor(
        None, predict_yolo_sync, yolo, img
    )
    
    # Extract boxes from YOLO results
    if not yolo_detections:
        return []
    
    # For simplicity in this two-stage function, we need the original box objects
    # Note: This implementation expects yolo_results with .boxes attribute.
    # For production, refactor to work with dictionary format.
    
    # Alternative approach: integrate with actual YOLO results object
    # The following is placeholder logic - adapt to your actual data structures
    return yolo_detections  # Return YOLO results (stage 2 not yet implemented)