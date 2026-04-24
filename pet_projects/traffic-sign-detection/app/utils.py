import io
import base64
import cv2
import numpy as np
import os
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw


# ============================================================
# UTILITY FUNCTIONS FOR IMAGE PROCESSING
# ============================================================

async def read_image(file: UploadFile) -> np.ndarray:
    """
    Read uploaded image file and convert to numpy array (OpenCV format).
    
    Args:
        file: UploadFile object from FastAPI request
    
    Returns:
        Image as numpy array in BGR format (OpenCV-compatible)
    """
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def base64_to_numpy(base64_str: str) -> np.ndarray:
    """
    Convert base64 encoded image string to numpy array (OpenCV format).
    
    Args:
        base64_str: Base64 string containing JPEG image data
    
    Returns:
        Image as numpy array in BGR format (OpenCV-compatible)
    """
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def draw_boxes_on_image(img: np.ndarray, detections: list) -> bytes:
    """
    Draw bounding boxes and labels on image, return as JPEG bytes.
    
    Args:
        img: Source image as numpy array (BGR format)
        detections: List of detection dictionaries with 'bbox' and 'class_name' keys
    
    Returns:
        JPEG image bytes with drawn bounding boxes
    """
    # Convert BGR (OpenCV) to RGB (PIL)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    # Draw each detection
    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            # Draw label text
            class_name = det.get("class_name", str(det.get("class_id", "")))
            draw.text((x1, y1 - 10), class_name, fill="red")
    
    # Convert back to JPEG bytes
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()


