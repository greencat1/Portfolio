# 📊 Data Science & Machine Learning Portfolio

Welcome to my **Data Science & Machine Learning portfolio**.

This repository contains my coursework, applied projects, and independent pet projects, demonstrating my progression from data science fundamentals to advanced, real-world analytical and machine learning solutions.

The portfolio focuses on **practical problem-solving**, **clean analytical workflows**, and **reproducible Python-based data science**.

My website - https://greencat1.tech/

---

## 📂 Repository Structure

```text
Portfolio/
├── CV/
│  
├── Courses/  
│   ├── Applied Data Science with Python/  
│   ├── More Applied Data Science with Python/  
│   ├── Data Science Foundations/
│   ├── Data Science Fundamentals with Python and SQL Specialization/
│   ├── IBM Generative AI Engineering Professional Certificate/
│   ├── Meta Front-End Developer Professional Certificate/ 
│   └── Certificates/  
│  
├── Projects/ # Hands-on ML projects
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

**🛠️ Tech Stack:** YOLO, PyTorch, Albumentations, FastAPI, WebSocket, Docker

**🔗 Links:**
- 🌐 [Live Demo](https://cv.greencat1.tech/)
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
- 📚 [API Docs](https://churn-api.greencat1.tech/docs)
- 📊 [Dashboard](https://churn.greencat1.tech/)
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
- 🌐 [Live Demo](https://rag.greencat1.tech/)
- 📚 [Swagger API](https://rag-api.greencat1.tech/docs)

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

## 🧪 Projects

Independent projects demonstrating **initiative** and **applied ML skills**:

- Work with real-world datasets
- Build end-to-end data science pipelines
- Experiment beyond coursework
- Focus on practical problem-solving

All  projects are in the `projects/` folder.

---

## 🧠 Core Skills & Competencies

### 📊 Data & Analytics
- **Processing:** Pandas, NumPy, EDA, feature engineering, missing values, data wrangling, data cleaning
- **Visualization:** Matplotlib, Seaborn, Plotly, Streamlit dashboards, data representation, charting
- **SQL:** Complex queries, JOINs, aggregations, window functions, SQLite, database design
- **Statistics:** Descriptive stats, hypothesis testing (t-test, chi-square), correlation analysis, A/B testing
- **Probability:** Distributions (Normal, Poisson, Binomial), CLT, bootstrapping

### 🤖 Machine Learning
- **Core:** Regression, classification, clustering, dimensionality reduction, unsupervised learning
- **Models:** CatBoost, XGBoost, LightGBM, scikit-learn, MLXtend
- **Techniques:** Cross-validation, hyperparameter tuning, class imbalance, incremental learning, feature engineering
- **Model Evaluation:** Confusion matrix, ROC-AUC, precision/recall, F1-score, MSE/MAE, R²
- **Interpretability:** SHAP, feature importance, model validation

### 🖼 Computer Vision & Deep Learning
- **Object Detection:** YOLOv7, SSD, bounding boxes, IoU, NMS, mAP
- **Deep Learning:** CNNs, transfer learning, PyTorch, model fine-tuning
- **Image Processing:** OpenCV, image augmentation, preprocessing

### 🔍 RAG & LLM (Generative AI)
- **RAG Pipeline:** FAISS vector search, BGE embeddings, semantic chunking, retrieval-augmented generation
- **LLM:** Ollama, tinyllama, phi3:mini (100% local, no cloud), Hugging Face Transformers
- **Parsing:** AST extraction, multi-format (`.md`, `.ipynb`, `.py`, `.pdf`), unstructured data extraction
- **Prompt Engineering:** In-context learning, few-shot prompting, system prompts
- **LLM Engineering:** Fine-tuning, PEFT (LoRA), embeddings, semantic search
- **Vector Databases:** FAISS, Chroma, Pinecone, Sentence-Transformers
- **Frameworks:** LangChain, Hugging Face ecosystem

### 🌐 Frontend Development
- **Markup & Styling:** HTML5, CSS3, Sass, Flexbox, Grid, responsive design, accessibility (a11y)
- **JavaScript:** ES6+, DOM manipulation, asynchronous programming, modern syntax
- **Frameworks:** React.js, React Router, component-based architecture, hooks, state management
- **State Management:** Context API, Redux
- **Testing:** Jest, React Testing Library, unit testing, integration testing
- **Build & Package:** Webpack, Vite, npm, Yarn
- **Design & Prototyping:** Figma, UI/UX principles

### 🚀 Backend & DevOps
- **API Development:** FastAPI, Uvicorn, REST API, authentication (API keys + RBAC), rate limiting (sliding window)
- **Containerization:** Docker, Docker Compose, multi-container orchestration
- **Deployment:** Ubuntu VPS, reverse proxy, zero-downtime updates, Netlify, Vercel, GitHub Pages
- **Web Scraping:** BeautifulSoup4, Requests
- **Model Serialization:** cloudpickle, pickle

### 📊 Monitoring & Testing
- **Logging:** Python logging, JSON format, rotating files (RotatingFileHandler)
- **Testing:** pytest, unit tests, integration tests, test coverage
- **Dashboard:** Streamlit, real-time metrics, business KPIs, monitoring dashboards
- **Dev Tools:** Chrome DevTools, ESLint, Prettier

### 📐 Statistics & Probability
- **Statistics:** Descriptive stats, hypothesis testing (t-test, chi-square), correlation analysis, regression assumptions
- **Probability:** Distributions (Normal, Poisson, Binomial), CLT, bootstrapping
- **Modeling:** Linear regression assumptions, bias-variance tradeoff, A/B testing, statistical inference

### 💼 Business & Soft Skills
- **Business:** ROI calculation, economic impact (MRR), technical documentation, stakeholder communication
- **Project Management:** Agile workflows, project planning, milestone tracking
- **Communication:** Data-driven insights, presenting to non-technical audiences, report writing
- **Collaboration:** Cross-functional teamwork, code review, pair programming

### 🛠 Workflow & Tools
- **Experimentation:** Google Colab, Jupyter Notebook
- **Version Control:** Git, GitHub, branching strategies, collaborative workflows
- **Containerization & Orchestration:** Docker, Docker Compose

---

## 🛠 Tools & Technologies

| Category | Tools |
|----------|-------|
| **Programming Languages** | Python, JavaScript (ES6+), TypeScript |
| **Data Science & Analysis** | NumPy, Pandas, SciPy |
| **Data Visualization** | Matplotlib, Seaborn, Plotly |
| **Machine Learning** | Scikit-learn, MLXtend, CatBoost, XGBoost |
| **Deep Learning** | PyTorch |
| **Computer Vision** | OpenCV, YOLOv7, SSD |
| **Natural Language Processing (NLP)** | NLTK, re (regular expressions) |
| **Generative AI & LLMs** | Hugging Face Transformers, LangChain, Prompt Engineering, RAG, Ollama, tinyllama, phi3:mini |
| **RAG & Vector Search** | FAISS, Chroma, Pinecone, Sentence-Transformers, BGE |
| **LLM Engineering** | Fine-tuning, PEFT (LoRA), Embeddings |
| **Graph Analysis** | NetworkX |
| **Web Scraping** | BeautifulSoup4, Requests |
| **API Development** | FastAPI, Uvicorn, REST API |
| **Dashboard & UI** | Streamlit |
| **Frontend: Markup & Styling** | HTML5, CSS3, Sass, Flexbox, Grid |
| **Frontend: Frameworks** | React.js, React Router |
| **Frontend: State Management** | Context API, Redux |
| **Frontend: Testing** | Jest, React Testing Library |
| **Frontend: Build Tools** | Webpack, Vite |
| **Frontend: Package Management** | npm, Yarn |
| **Frontend: Design & Prototyping** | Figma |
| **Frontend: Deployment** | Netlify, Vercel, GitHub Pages |
| **Frontend: Dev Tools** | Chrome DevTools, ESLint, Prettier |
| **Databases** | SQL, SQLite |
| **Model Serialization** | cloudpickle, pickle |
| **Authentication** | API Keys, SHA-256 hashing, RBAC |
| **Rate Limiting** | Sliding window algorithm |
| **Containerization** | Docker, Docker Compose |
| **Testing** | pytest, unit testing, integration testing |
| **Logging** | Python logging, RotatingFileHandler |
| **Deployment** | Ubuntu VPS, Docker Compose |
| **Experimentation** | Google Colab |
| **Workflow & Version Control** | Jupyter Notebook, Git, GitHub |


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
