import io
import base64
import cv2
import numpy as np
import os
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw

# Import custom modules
from app.models import load_models
from app.detector import detect_two_stage, predict_yolo_sync
from app.utils import read_image, base64_to_numpy, draw_boxes_on_image 

# Initialize FastAPI application
app = FastAPI(title="Two-Stage Road Sign Detector")

# Load models at startup
yolo = load_models()

# Path to static HTML file
INDEX_PATH = os.path.join(os.path.dirname(__file__), "static", "index.html")


# ============================================================
# Root endpoint - serves the web interface
# ============================================================
@app.get("/")
async def get_index():
    """Return the main HTML page with camera and WebSocket interface"""
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ============================================================
# WebSocket endpoint for real-time video detection
# ============================================================
@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket handler for real-time road sign detection.
    
    Receives base64-encoded JPEG frames from the client,
    runs YOLO detection, and returns bounding boxes as JSON.
    """
    await websocket.accept()
    print("✅ WebSocket client connected")
    
    try:
        while True:
            # Receive base64 image string from client
            data = await websocket.receive_text()
            
            # Convert base64 string back to numpy array (OpenCV format)
            img = base64_to_numpy(data)
            
            # Validate image
            if img is None:
                await websocket.send_json({"error": "Invalid image"})
                continue
            
            # Run single-stage YOLO detection
            detections = predict_yolo_sync(yolo, img)
            
            # Send detection results back to client as JSON
            await websocket.send_json(detections)
            
    except WebSocketDisconnect:
        print("❌ WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")