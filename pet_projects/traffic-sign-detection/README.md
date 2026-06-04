# 🚦 Two-Stage Traffic Sign Detection System (YOLO + ResNet)

Real-time traffic sign recognition system using a **two-stage hybrid architecture**:
**YOLO (detection) → ResNet (classification)**

🌐 **Live Demo:** https://cv.greencat1.tech/
▶️ **Video Guide:** https://www.youtube.com/watch?v=XS1jED6fzpY  

---

## ✨ Key Features

- **Two-stage cascade architecture**
  - YOLO detects 5 coarse categories
  - 5 specialized ResNet-18 models classify 55 signs
- **High accuracy**
  - **94.07% mAP50**
  - **91.71% Recall**
- **Real-time detection via WebSocket**
- **Browser-based interface (camera streaming)**
- **Fully containerized (Docker)**

---

## 📊 Model Performance

### Test Set Comparison

| Model                         | mAP50  | mAP50-95 | Precision | Recall |
|------------------------------|--------|----------|-----------|--------|
| Baseline YOLO (55 classes)   | 81.36% | 71.07%   | 84.48%    | 73.28% |
| YOLO + Preprocessing         | 84.30% | 73.67%   | 82.50%    | 79.63% |
| **YOLO + 5×ResNet (Hybrid)** | **94.07%** | **85.60%** | **90.33%** | **91.71%** |

### Improvements over baseline:
- **+12.71% mAP50**
- **+18.43% Recall**

---

## 🏗️ Architecture
Browser (Camera)
│
▼
WebSocket (Real-time)
│
▼
FastAPI Server
│
▼
YOLO → Crop → 5×ResNet → JSON → Frontend


### Stage 1 — YOLO (Detection)

Detects **5 high-level categories**:
- `forb` — prohibitory
- `warn` — warning
- `mand` — mandatory
- `info` — informational
- `other` — priority

### Stage 2 — ResNet (Classification)

- 5 independent **ResNet-18 models**
- Each specializes in one category
- Final classification across **55 traffic signs**

---

## 📈 Dataset & Challenges

| Metric | Value |
|--------|-------|
| Images | 2,838 |
| Annotations | 4,895 |
| Classes | 55 |
| **Imbalance ratio** | **163:1** |

### Key Challenge

Severe class imbalance:
- Rare classes: **2–6 samples**
- Frequent classes: **300+ samples**

---

## 🛠️ Data Processing & Augmentation

To address imbalance:

### Oversampling Strategy
- Rare classes → upsampled to **150–200 samples**
- Medium classes → balanced to **100–280 samples**

### Augmentations (Albumentations)

| Technique | Purpose |
|----------|--------|
| Rotation / Flip | Orientation invariance |
| Shift / Scale | Spatial robustness |
| Brightness / Contrast | Lighting conditions |
| Noise / Blur | Real-world imperfections |
| CLAHE | Contrast enhancement |

---

## 🎯 Training Details

| Component | Value |
|----------|------|
| Backbone | ResNet-18 (ImageNet pretrained) |
| Fine-tuning | Full |
| Loss | CrossEntropy + Label Smoothing (0.1) |
| Optimizer | AdamW |
| Scheduler | ReduceLROnPlateau |


---

## 📊 Per-Category Performance

| Category | Recall |
|----------|--------|
| Warning | 98.5% |
| Prohibitory | 100% |
| Informational | 98.4% |
| Priority | 99.6% |
| Mandatory | 79% |

> Mandatory signs remain the weakest due to visual similarity (arrow directions).

---

## 🚀 Live Demo

1. Open: https://greencat1.tech  
2. Allow camera access  
3. Point at traffic signs  
4. See real-time detection with bounding boxes and confidence scores  

---

## 🐳 Quick Start (Docker)

```bash
git clone https://github.com/greencat1/traffic-sign-detection.git
cd traffic-sign-detection

docker build -t traffic-sign-detector .
docker run -p 8000:8000 traffic-sign-detector
```
--- 

🛠️ Local Setup

pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

---

📁 Project Structure

    traffic-sign-detection/
    ├── app/
    │   ├── main.py              # FastAPI + WebSocket
    │   ├── detector.py          # Two-stage pipeline
    │   ├── models.py            # Model loading
    │   ├── utils.py             # Utilities
    │   ├── config.py            # config
    │   ├── weights/
    │   │   ├── yolo/
    │   │   └── resnet/
    │   └── static/index.html    # Frontend
    ├── data/
    ├── models/
    ├── notebooks/
    ├── Dockerfile
    ├── requirements.txt
    └── README.md

--- 

📚 Development Stages (Notebooks)

1. EDA & imbalance analysis
2. YOLO baseline (55 classes)
3. YOLO with preprocessing
4. YOLO (5-category detector)
5. ResNet training (stage 2)
6. Final hybrid evaluation

--- 

🔧 Tech Stack

- Detection: YOLO (Ultralytics)
- Classification: ResNet-18 (PyTorch)
- Backend: FastAPI + WebSocket
- Frontend: HTML + JavaScript
- Training: PyTorch, Albumentations
- Deployment: Docker

--- 

📈 Key Insights

- Two-stage architecture significantly improves performance vs single-stage
- Class imbalance was the main bottleneck
- Specialized classifiers outperform a single multi-class model
- System achieves real-time performance

--- 

🔮 Future Work

- Improve mandatory signs classification
- Add tracking for video streams
- Deploy on mobile / embedded devices

---

👨‍💻 Author

Ivan
🌐 https://github.com/greencat1/Portfolio/

Portfolio project — feedback and stars are welcome ⭐


