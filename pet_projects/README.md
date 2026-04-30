# 🧠 Data Science & ML Portfolio

Collection of production-ready and research-oriented machine learning projects.

---

## 📁 Projects Overview

| Project | Description | Tech Stack | Status |
|--------|------------|------------|--------|
| **customer-churn-prediction** | End-to-end churn prediction system with API, dashboard, and deployment | FastAPI, CatBoost, Streamlit, Docker, SQLite | ✅ Production |
| **traffic-sign-detection** | Real-time traffic sign detection using YOLO + ResNet cascade (55 classes) | YOLO, PyTorch, Albumentations, FastAPI, WebSocket | ✅ Production |

---

## 🚀 Live Demo

### 🚦 Traffic Sign Detection
- **Live Demo:** https://greencat1.tech/  
- **Video Guide:** https://www.youtube.com/watch?v=XS1jED6fzpY  

> Real-time detection via browser camera (WebSocket streaming)

---

### 📊 Customer Churn Prediction
- **API Docs (Swagger):** http://207.231.105.113:8000/docs  
- **Dashboard (Streamlit):** http://207.231.105.113:8501  
- **Video Guide:** https://www.youtube.com/watch?v=zrF0L2pznhU  

> Fully deployed ML service with REST API + interactive UI

---

## 📊 Project Details

---

### 🔹 1. Customer Churn Prediction (Production)

**Goal:**  
Predict telecom customer churn and provide actionable insights for retention.

**Key Features:**
- End-to-end pipeline: training → API → UI → deployment
- REST API with **23 endpoints**
- **Role-based authentication**
- Interactive dashboard for business users

**Model Performance:**
- **Recall: 84.5%** (optimized for churn detection)

**Engineering Highlights:**
- Modular FastAPI backend
- Dockerized multi-service architecture
- SQLite for lightweight persistence
- Streamlit dashboard for visualization

**Impact:**
- Full **production-ready ML system**
- Covers complete ML lifecycle (MLOps mindset)

**Repository:** `customer-churn-prediction/`

---

### 🔹 2. Traffic Sign Detection (Production)

**Goal:**  
Build a **high-accuracy real-time detection system** for 55 traffic sign classes.

**Final Architecture:**
- **Stage 1:** YOLO (5 categories)
- **Stage 2:** 5× ResNet-18 (category-specific classification)

**Key Results:**
- **94.07% mAP50**
- **91.71% Recall**
- **+12.7% mAP improvement vs baseline YOLO**
- Real-time inference via WebSocket

**Key Challenges Solved:**
- Extreme class imbalance (**163:1**) → fixed via oversampling + augmentation
- Confusing classes → solved via **model specialization**
- Real-time constraints → optimized pipeline

**Engineering Highlights:**
- Two-stage cascade architecture
- FastAPI + WebSocket streaming
- Browser-based UI (camera input)
- Docker deployment

**Impact:**
- Significant quality boost over single-stage detector
- Demonstrates **ML system design, not just modeling**

**Repository:** `traffic-sign-detection/`

---

## 🧰 Tech Stack

**Machine Learning**
- PyTorch, CatBoost
- Scikit-learn
- Ultralytics YOLO

**Backend & Deployment**
- FastAPI
- Docker
- WebSocket

**Data Processing**
- Pandas, NumPy
- Albumentations

**Visualization**
- Streamlit
- Matplotlib / Seaborn

---

## 💡 What This Portfolio Demonstrates

- Building **end-to-end ML systems**
- Strong understanding of:
  - model vs system trade-offs
  - real-time constraints
  - class imbalance handling
- Experience with:
  - production APIs
  - deployment (Docker)
  - interactive ML interfaces
- Focus on **practical impact over toy models**

---

## 🔮 Future Work

- CI/CD for ML services  
- Model monitoring & logging  
- Experiment tracking (W&B / MLflow)  
- Edge / mobile deployment  

---

## 👨‍💻 Author

**Ivan Sazontov**

- GitHub: https://github.com/greencat1  
- Email: isazontov1@gmail.com  

---

⭐ If you find this portfolio interesting, feel free to explore the repositories and give feedback!
