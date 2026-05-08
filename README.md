# 📊 Data Science & Machine Learning Portfolio

Welcome to my **Data Science & Machine Learning portfolio**.

This repository contains my coursework, applied projects, and independent pet projects, demonstrating my progression from data science fundamentals to advanced, real-world analytical and machine learning solutions.

The portfolio focuses on **practical problem-solving**, **clean analytical workflows**, and **reproducible Python-based data science**.

---

## 📂 Repository Structure

```text
Portfolio/  
├── Courses/  
│   ├── Applied Data Science with Python/  
│   ├── More Applied Data Science with Python/  
│   ├── Data Science Foundations/
│   ├── Data Science Fundamentals with Python and SQL Specialization/
│   ├── IBM Generative AI Engineering Professional Certificate/  
│   └── Certificates/  
│  
├── Pet_Projects/ # Hands-on ML projects
│   ├── customer-churn-prediction/     # Churn prediction service (FastAPI + Docker)
│   ├── traffic-sign-detection/        # Road signs detection service (FastAPI + Docker)
│   └── llm-portfolio-assistant/       # RAG assistant for portfolio (FastApi + Docker)
│  
├── scientific publications/
│   ├── Traffic_Sign_Detection_SSD/
│   ├── YOLOv7_Traffic_Sign_Detection/
│  
└── README.md
```

## 🚀 Featured Projects

---

### 🚦 Traffic Sign Detection (Production)

**Real-time traffic sign recognition | YOLO + ResNet cascade | 55 classes**

Real-time traffic sign recognition system using **YOLO + ResNet cascade architecture**. Two-stage design solves class imbalance (163:1) and confusing categories.

**🔥 Key Highlights:**
- **94.07% mAP50 / 91.71% Recall**
- **+12.7% improvement vs baseline YOLO**
- Real-time detection via **WebSocket + browser camera**
- Two-stage architecture:
  - YOLO → category detection (5 groups)
  - 5× ResNet → fine-grained classification

**🛠️ Tech Stack:** YOLOv7, PyTorch, Albumentations, FastAPI, WebSocket, Docker

**🔗 Links:**
- 🌐 [Live Demo](https://greencat1.tech/)
- ▶️ [Video Guide](https://www.youtube.com/watch?v=XS1jED6fzpY)

---

### 📊 Customer Churn Prediction (Production)

**End-to-end ML system | 23 REST endpoints | Production-ready**

Complete MLOps pipeline for telecom churn prediction: training → labeling → retraining → API → dashboard.

**🔥 Key Highlights:**
- **84.5% recall** (optimized for business impact)
- **23 REST endpoints** with role-based access (user/admin/dashboard)
- **Sliding window rate limiting** (100-1000 requests/minute)
- **Incremental + full retraining** (model CI/CD)
- **Manual labeling system** with SQLite persistence
- **Interactive Streamlit dashboard** for business users

**🛠️ Tech Stack:** FastAPI, CatBoost, Streamlit, SQLite, Docker Compose, pytest

**🔗 Links:**
- 📚 [API Docs](http://207.231.105.113:8000/docs)
- 📊 [Dashboard](http://207.231.105.113:8501)
- ▶️ [Video Guide](https://www.youtube.com/watch?v=zrF0L2pznhU)

---

### 🤖 RAG Portfolio Assistant

**Question-answering over ML portfolio | 8,361 chunks | 100% local**

Full RAG pipeline: parse → chunk → embed → search → generate. No cloud APIs — everything runs locally.

**🔥 Key Highlights:**
- **8,361 semantic chunks** from **291 files** (.md, .ipynb, .py, .pdf)
- **AST extraction** for Python code (functions, classes, imports, docstrings)
- **FAISS vector search** on CPU (cosine similarity)
- **Local LLM** (tinyllama / phi3:mini via Ollama)
- **Smart chunking** (sentence-aware, 500 chars, 10% overlap)
- **FastAPI backend** + **Streamlit chat interface**
- **Docker Compose** (3 containers: Ollama + API + Dashboard)


**🛠️ Tech Stack:** FastAPI, FAISS, Sentence-Transformers (BGE), Ollama, tinyllama, Streamlit, Docker

**🔗 Links:**
- 🌐 [Live Demo](http://207.231.105.113:8502/)
- 📚 [Swagger API](http://207.231.105.113:8002/docs)

---

## 🎓 Courses & Training

Completed structured data science programs with hands-on projects covering the full data science workflow:

**Core competencies:**
- **Data analysis:** Python (Pandas/NumPy) + SQL (JOINs, aggregations, window functions)
- **Machine learning:** Supervised/unsupervised, model validation, metrics (precision, recall, F1)
- **Advanced topics:** NLP (NLTK), graph analysis (NetworkX), Generative AI fundamentals
- **Best practices:** Clean code, reproducibility, experiment-driven development

📁 All coursework, assignments, and capstones are in the `Courses/` folder.

---

## 🧪 Pet Projects

Independent projects demonstrating **initiative** and **applied ML skills**:

- Work with real-world datasets
- Build end-to-end data science pipelines
- Experiment beyond coursework
- Focus on practical problem-solving

All pet projects are in the `Pet_Projects/` folder.

---

## 🧠 Core Skills & Competencies

### 📊 Data & Analytics
- **Processing:** Pandas, NumPy, EDA, feature engineering, missing values
- **Visualization:** Matplotlib, Seaborn, Plotly, Streamlit dashboards
- **SQL:** Complex queries, JOINs, aggregations, window functions, SQLite

### 🤖 Machine Learning
- **Core:** Regression, classification, clustering, dimensionality reduction
- **Models:** CatBoost, XGBoost, LightGBM, scikit-learn
- **Techniques:** Cross-validation, hyperparameter tuning, class imbalance, incremental learning

### 🖼 Computer Vision & Deep Learning
- **Object detection:** YOLOv7, SSD, bounding boxes, IoU, NMS, mAP
- **Deep learning:** CNNs, transfer learning, PyTorch

### ✅ RAG & LLM
- **RAG pipeline:** FAISS vector search, BGE embeddings, semantic chunking
- **LLM:** Ollama, tinyllama, phi3:mini (100% local, no cloud)
- **Parsing:** AST extraction, multi-format (`.md`, `.ipynb`, `.py`, `.pdf`)

### 🚀 Backend & DevOps
- **API:** FastAPI, authentication (API keys + RBAC), rate limiting (sliding window)
- **Containerization:** Docker, Docker Compose, multi-container orchestration
- **Deployment:** Ubuntu VPS, reverse proxy, zero-downtime updates

### 📊 Monitoring & Testing
- **Logging:** Python logging, JSON format, rotating files
- **Testing:** pytest, unit tests, integration tests
- **Dashboard:** Streamlit, real-time metrics, business KPIs

### 📐 Statistics & Probability
- **Statistics:** Descriptive stats, hypothesis testing (t-test, chi-square), correlation analysis
- **Probability:** Distributions (Normal, Poisson, Binomial), CLT, bootstrapping
- **Modeling:** Linear regression assumptions, bias-variance tradeoff, A/B testing

### 💼 Business Skills
- ROI calculation, economic impact (MRR), technical documentation, stakeholder communication

---

## 🛠 Tools & Technologies

<details>
<summary>📊 Click to expand: Core Skills (40+ items)</summary>

| Category | Tools |
|----------|-------|
| Programming | Python |
| Data Libraries | NumPy, Pandas, SciPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, MLXtend, CatBoost, XGBoost |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV, YOLOv7, SSD |
| NLP | NLTK, re (regular expressions) |
| **RAG & Vector Search** | **FAISS, Sentence-Transformers, BGE** |
| **LLM** | **Ollama, tinyllama, phi3:mini** |
| Graph Analysis | NetworkX |
| Web Scraping | BeautifulSoup4, Requests |
| API Development | FastAPI, Uvicorn, REST API |
| Dashboard & UI | Streamlit |
| Databases | SQL, SQLite |
| Model Serialization | cloudpickle, pickle |
| Authentication | API Keys, SHA-256 hashing, RBAC |
| Rate Limiting | Sliding window algorithm |
| Containerization | Docker, Docker Compose |
| Workflow | Jupyter Notebook, Git, GitHub |
| Testing | pytest, unit testing, integration testing |
| Logging | Python logging, RotatingFileHandler |
| Deployment | Ubuntu VPS, Docker Compose |
| Experimentation | Google Colab |


</details>

---

## 📜 Certificates

The **Certificates** folder contains official certificates and statements of accomplishment confirming successful completion of courses and specializations.

---

## 📚 Scientific Publications

- **Traffic Sign Detection Using SSD**  
  Conference proceedings paper (RSCI), SSD-based object detection

- **Application of YOLOv7 for Traffic Sign Detection**  
  Conference paper (RSCI), YOLOv7-based object detection


---

## 🎯 Purpose of This Repository

- Demonstrate practical data science and machine learning skills  
- Showcase growth from fundamentals to advanced applied analytics  
- Serve as a professional portfolio for **junior data-related roles**, internships, and collaborations  

---

## 📬 Contact & Connect

- **Email:** isazontov1@gmail.com  
- **Telegram:** https://t.me/Hammerschmidt1  
- **GitHub:** https://github.com/greencat1  
