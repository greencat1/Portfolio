# 🧠 Data Science & ML Portfolio

Collection of production-ready and research-oriented machine learning projects.

---

## 📁 Projects Overview

| Project | Description | Tech Stack | Status |
|--------|------------|------------|--------|
| **customer-churn-prediction** | End-to-end churn prediction system with API, dashboard, and deployment | FastAPI, CatBoost, Streamlit, Docker, SQLite | ✅ Production |
| **traffic-sign-detection** | Real-time traffic sign detection using YOLO + ResNet cascade (55 classes) | YOLO, PyTorch, Albumentations, FastAPI, WebSocket | ✅ Production |
| **rag-portfolio-assistant** | RAG system for semantic search over ML portfolio | FastAPI, FAISS, Sentence-Transformers, Ollama, Streamlit, Docker | ✅ Production |

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

### 🤖 RAG Portfolio Assistant 
- **Live Demo Dashboard:** http://207.231.105.113:8502/
- **Swagger API Docs:** http://207.231.105.113:8002/docs
- **GitHub:** [llm-portfolio-assistant](https://github.com/greencat1/Portfolio/tree/main/pet_projects/llm-portfolio-assistant)

> Ask questions about my portfolio — gets answers via semantic search + local LLM.

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

### 🔹 3. RAG Portfolio Assistant (Production) — *NEW*

**Goal:**  
Build a **question-answering system** over my entire ML portfolio using RAG (Retrieval-Augmented Generation).

**How It Works:**
1. User asks a question
2. System finds relevant chunks via semantic search (FAISS)
3. LLM generates answer based **only** on retrieved context

**Key Features:**
- 8,361 semantic chunks from 291 files (`.md`, `.ipynb`, `.py`, `.pdf`)
- **FAISS** vector search on CPU
- **BGE embeddings** (`BAAI/bge-small-en-v1.5`)
- **Local LLM** (`tinyllama` / `phi3:mini` via Ollama)
- FastAPI backend + Streamlit chat UI
- Docker Compose deployment (3 containers)


**Repository:** `llm-portfolio-assistant/`

---

## 🧰 Tech Stack

**Machine Learning**
- PyTorch, CatBoost, Sentence-Transformers
- Scikit-learn, FAISS
- Ultralytics YOLO

**Backend & Deployment**
- FastAPI, Uvicorn
- Docker, Docker Compose
- WebSocket
- Ollama (local LLM)

**Data Processing**
- Pandas, NumPy
- Albumentations

**Visualization**
- Streamlit
- Matplotlib / Seaborn

---

## 💡 What This Portfolio Demonstrates

- Building **end-to-end ML systems**
- **RAG system design** (embeddings + vector search + LLM)
- Strong understanding of:
  - model vs system trade-offs
  - real-time constraints
  - class imbalance handling
  - CPU optimization for LLMs
- Experience with:
  - production APIs
  - deployment (Docker)
  - interactive ML interfaces
  - multi-format document parsing
- Focus on **practical impact over toy models**

---

## 🔮 Future Work

- CI/CD for ML services  
- Model monitoring & logging  
- Experiment tracking (W&B / MLflow)  
- Edge / mobile deployment  
- RAG evaluation metrics (hit rate, MRR)

---

## 👨‍💻 Author

**Ivan Sazontov**

- GitHub: https://github.com/greencat1  
- Email: isazontov1@gmail.com  
- Telegram: @Hammerschmidt1

---

⭐ If you find this portfolio interesting, feel free to explore the repositories and give feedback!
