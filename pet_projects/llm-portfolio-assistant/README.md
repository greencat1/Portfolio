# 🚀 RAG Portfolio Assistant

A production-ready **RAG (Retrieval-Augmented Generation)** system that answers questions about my Data Science & Machine Learning portfolio. The system uses semantic search over 8,000+ document chunks and a local LLM to generate accurate, factual answers.

**🌐 Live Demo dashboard:** https://rag.greencat1.tech/
**🌐 Live Demo swagger:** https://rag-api.greencat1.tech/docs

---

## 📊 What It Does

| Feature | Description |
|---------|-------------|
| **Semantic Search** | FAISS vector search over 8,361 text chunks (BGE embeddings) |
| **Local LLM** | Runs `tinyllama` or `phi3:mini` via Ollama (CPU-only, no GPU needed) |
| **REST API** | FastAPI endpoints for asking questions |
| **Chat Interface** | Streamlit dashboard for interactive conversations |
| **Multi-format Parsing** | Supports `.md`, `.ipynb`, `.py`, `.pdf` files |
| **Docker Compose** | 3-container setup (Ollama + API + Dashboard) |

---

## 🔧 Architecture

User Question → Embedding (BGE) → FAISS Search → Deduplicate (top-5) → Context → LLM (tinyllama) → Answer + Sources

### Screenshot from the dashboard

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/5b557bae-aec2-4639-9168-2b4be681be84" />


### Container Services

| Container | Port | Purpose |
|-----------|------|---------|
| **rag-ollama** | 11435 | Local LLM server (tinyllama) |
| **rag-api** | 8002 | FastAPI RAG backend |
| **rag-dashboard** | 8502 | Streamlit chat UI |

---

## 💡 Example Questions

### Churn Prediction
- *"What model is used in churn prediction?"* → CatBoost with 95.7% recall
- *"Why was recall chosen as the main metric?"* → Critical to identify customers likely to leave, minimizes false negatives

### Traffic Sign Detection
- *"Describe the traffic sign detection architecture"* → Two-stage hybrid: YOLO → ResNet-18, 94.07% mAP50
- *"How is it better than baseline YOLO?"* → +12.7% improvement (94.07% vs 81.3% mAP50)

### Portfolio
- *"What technologies were used in the API?"* → FastAPI, CatBoost, Docker, Streamlit
- *"What makes this production-ready?"* → Dashboard, labeling system, ML pipeline, Docker, zero downtime

---

## 🚀 Quick Start (Local)

### Prerequisites
- Docker Desktop

### 1. Clone and run

```bash
git clone https://github.com/greencat1/Portfolio.git
cd Portfolio/pet_projects/llm-portfolio-assistant
docker compose up -d
```

### 2. Pull the model (first run only)

```bash
docker exec -it rag-ollama ollama pull tinyllama
```

### 3. 🌐 Open in Browser

| Interface | URL |
|-----------|-----|
| **Chat Dashboard** | http://localhost:8502 |
| **API Docs (Swagger)** | http://localhost:8002/docs |
| **Health Check** | http://localhost:8002/health |

---

## 📁 Project Structure

    lm-portfolio-assistant/
    ├── app/ # FastAPI backend
    │ ├── main.py # API routes
    │ ├── config.py # Configuration
    │ ├── rag_pipeline.py # RAG logic
    │ └── schemes.py # Pydantic models
    ├── data/ # FAISS index & chunks
    │ ├── chunks/
    │ └── embeddings/
    ├── frontend/ # Streamlit UI
    │ └── streamlit_app.py
    ├── notebooks # Jupiter notebooks
    ├── Dockerfile.api # FastAPI container
    ├── Dockerfile.dashboard # Streamlit container
    ├── docker-compose.yml # Orchestration
    └── requirements.txt # Python dependencies

---

## 🛠️ Technologies

| Category | Tools |
|----------|-------|
| **Backend** | FastAPI, Uvicorn |
| **ML** | Sentence-Transformers, FAISS |
| **LLM** | Ollama, tinyllama |
| **Frontend** | Streamlit |
| **Containerization** | Docker, Docker Compose |
| **Deployment** | Ubuntu VPS, Nginx (optional) |

---

## 📬 Contact

- **GitHub:** [greencat1](https://github.com/greencat1)
- **Email:** isazontov1@gmail.com
- **Telegram:** [@Hammerschmidt1](https://t.me/Hammerschmidt1)

---

## 📚 Source Code

[GitHub Repository](https://github.com/greencat1/Portfolio/tree/main/pet_projects/llm-portfolio-assistant)
