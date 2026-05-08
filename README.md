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

---

## 🚀 Featured Projects

### 🚦 Traffic Sign Detection (Production)

Real-time traffic sign recognition system using **YOLO + ResNet cascade architecture (55 classes)**

- **94.07% mAP50 / 91.71% Recall**
- **+12.7% improvement vs baseline YOLO**
- Real-time detection via **WebSocket + browser camera**
- Two-stage architecture:
  - YOLO → category detection
  - 5× ResNet → fine classification

🌐 **Live Demo:** https://greencat1.tech/  
▶️ **Video Guide:** https://www.youtube.com/watch?v=XS1jED6fzpY  

---

### 📊 Customer Churn Prediction (Production)

End-to-end ML system for predicting telecom churn.

- **84.5% recall** (optimized for business impact)
- REST API (**23 endpoints**) with authentication & rate limiting
- Interactive **Streamlit dashboard**
- Dockerized multi-container deployment

🌐 **API Docs:** http://207.231.105.113:8000/docs  
📊 **Dashboard:** http://207.231.105.113:8501  
▶️ **Video Guide:** https://www.youtube.com/watch?v=zrF0L2pznhU  

--- 

### 🤖 RAG Portfolio Assistant 

Question-answering system over my entire ML portfolio using Retrieval-Augmented Generation (RAG).

- 8,361 semantic chunks from 291 files (.md, .ipynb, .py, .pdf)
- FAISS vector search on CPU
- Local LLM (tinyllama / phi3:mini via Ollama)
- FastAPI backend + Streamlit chat interface
- Fully local — no API calls to OpenAI/cloud

🌐 **Live Demo**: http://207.231.105.113:8502/
📚 **Swagger API**: http://207.231.105.113:8002/docs 
▶️ **Video Guide**: (coming soon)

---

## 🎓 Courses

The **Courses** folder contains coursework, assignments, and capstone projects completed as part of structured data science programs, including Python- and SQL-based analytics.

These materials demonstrate hands-on experience with real-world datasets, relational data sources, and end-to-end analytical workflows — from raw data extraction to insight generation, model development, and evaluation.

The projects typically follow a structured **data science workflow**, including:

- Data collection and ingestion  
- Data cleaning and preprocessing  
- Exploratory data analysis (EDA)  
- Feature engineering  
- Model development and evaluation  
- Result interpretation and visualization  

### Topics & Competencies Covered

- **Data analysis with Python** (Pandas, NumPy, data cleaning, feature engineering)  
- **Relational data analysis with SQL** (SELECT queries, JOINs, aggregations, subqueries, basic window functions)  
- **Exploratory data analysis (EDA) & visualization** (Matplotlib, Seaborn, insight-driven storytelling)  
- **Machine learning** (supervised & unsupervised methods, model selection, validation, and evaluation metrics such as accuracy, precision, recall, and F1-score)  
- **Natural Language Processing (NLP)** (text preprocessing, tokenization, feature extraction)  
- **Network & graph analysis** (graph construction, centrality measures, community detection)  
- **Information extraction from unstructured data** (regex-based parsing, structured data generation)  
- **Generative AI fundamentals** (large language models, transformer-based NLP, and prompt-based workflows)

### Additional Focus Areas

- Working with **real-world datasets**, including structured and semi-structured data sources  
- Integration of **SQL and Python** within analytical workflows  
- Reproducible research using **Jupyter notebooks** and structured project organization  
- Experiment-driven model development and performance evaluation  
- Clear documentation of assumptions, analytical decisions, and limitations  

### Project Principles

These projects emphasize:

- Clear problem formulation and analytical thinking  
- Metric-driven and reproducible analysis  
- Integration of SQL and Python in data workflows  
- Clean, well-documented, and reproducible code  

---

## 🧪 Pet Projects

The **Pet_Projects** folder contains independent projects and experiments where I:

- Work with real-world datasets  
- Build end-to-end data science pipelines  
- Experiment with models and techniques beyond coursework  
- Focus on practical, applied problem-solving  

Pet projects highlight **initiative**, **curiosity**, and **hands-on machine learning skills**.

---

## 🧠 Core Skills & Competencies

### 📊 Data Analysis & Processing
- Advanced Pandas data manipulation  
- Feature engineering  
- Data cleaning and preprocessing  
- Exploratory Data Analysis (EDA)  
- Handling missing values with domain logic
- Correlation analysis and multicollinearity reduction

### 📈 Data Visualization
- Matplotlib, Seaborn  
- Clear and interpretable visualizations  
- Data-driven storytelling  
- Plotly interactive visualizations 
- Dashboard development (Streamlit)

### 🤖 Machine Learning
- Regression and classification  
- Unsupervised learning (clustering, dimensionality reduction)  
- Model evaluation and validation  
- Cross-validation and metrics  
- Scikit-learn pipelines  
- Gradient boosting (CatBoost, XGBoost, LightGBM)
- Class imbalance handling (class weights, stratified sampling)
- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV) 
- Incremental / continued learning

### 🧩 Advanced & Unsupervised Methods
- Clustering techniques  
- Pattern discovery  
- Dimensionality reduction  
- Network-based modeling

### 🖼 Computer Vision & Object Detection
- Object detection pipelines
- Bounding box regression
- Intersection over Union (IoU)
- Non-Maximum Suppression (NMS)
- Mean Average Precision (mAP)
- Confusion matrix analysis
- YOLOv7 architecture
- SSD (Single Shot Detector)

### 🤖 Deep Learning
- Convolutional Neural Networks (CNNs)
- Single Shot Detector (SSD)
- YOLOv7
- Transfer learning
- Training and evaluation of deep models

### 🧪 Model Evaluation & Validation
- Train / validation / test splits
- Cross-validation
- Overfitting detection
- Metric-driven optimization
- Recall as primary metric for imbalanced data
- Precision-Recall tradeoff

### 📝 NLP & Information Extraction
- Text preprocessing  
- Tokenization and normalization  
- NLTK-based workflows  
- Information extraction from text data

### ✅ RAG (Retrieval-Augmented Generation)
- Semantic search with FAISS vector database
- Local LLM (tinyllama / phi3:mini via Ollama)
- Multi-format document parsing (`.md`, `.ipynb`, `.py`, `.pdf`)
- Smart chunking (sentence-aware with overlap)
- Embedding model (BGE-small-en-v1.5)
- Full RAG pipeline (retrieve → deduplicate → context → generate)

### 🕸 Graphs & Networks
- Network analysis with NetworkX  
- Centrality measures  
- Community detection  
- Graph structure analysis  

### 🌐 Data Collection
- Web scraping with BeautifulSoup  
- HTTP requests with Requests

### 🗄 SQL & Relational Databases
- Writing complex SQL queries (SELECT, WHERE, JOIN)
- Aggregations and GROUP BY
- Subqueries and Common Table Expressions (CTE)
- Window functions (ROW_NUMBER, RANK)
- Data filtering and transformation
- Connecting Python to SQL databases
- SQLite for embedded applications
- Database design (3 tables: api_keys, new_data, raw_data)

### 🚀 API Development & Backend
- REST API design (FastAPI)
- API key authentication (SHA-256 hashing)
- Role-Based Access Control (user/admin/dashboard)
- Rate limiting (sliding window algorithm)
- Request/response validation (Pydantic)
- Automatic API documentation (Swagger UI / OpenAPI)
- Batch processing endpoints

### 📊 Dashboard & Monitoring
- Interactive dashboard development (Streamlit)
- Real-time data visualization
- Business metric tracking (MRR, revenue at risk)
- Retention recommendations UI

### 🐳 DevOps & Deployment
- Containerization with Docker
- Multi-container orchestration (Docker Compose)
- Volume mounting for data persistence
- VPS deployment (Ubuntu)

### 🧪 Testing & Quality Assurance
- Unit testing with pytest

### 📝 Logging & Monitoring
- Application logging (Python logging module)
- Rotating file handlers
- JSON log format
- Console + file logging

### 🔐 Security
- SHA-256 hashing
- Soft deletion (is_active flag)
- Role-based endpoint protection
- Header-based authentication (X-API-Key)

### 📐 Statistics & Probability

#### Descriptive Statistics
- Mean, median, mode  
- Variance and standard deviation  
- Quantiles, percentiles, interquartile range (IQR)  
- Distribution shape analysis (skewness, kurtosis)

#### Probability Theory
- Random variables  
- Probability distributions  
- Bernoulli, Binomial, Poisson, Normal distributions  
- Probability mass and density functions (PMF, PDF)  
- Cumulative distribution functions (CDF)

#### Statistical Inference
- Sampling techniques and sampling bias  
- Point estimation  
- Confidence intervals  
- Central Limit Theorem (CLT)  
- Bootstrapping and resampling methods

#### Hypothesis Testing
- Null and alternative hypotheses  
- p-values and significance levels  
- One-sample and two-sample tests  
- t-test and z-test  
- Chi-square test  
- Non-parametric tests (Mann–Whitney U, Wilcoxon)

#### Correlation & Dependency Analysis
- Pearson correlation  
- Spearman rank correlation  
- Covariance analysis  
- Multicollinearity detection

#### Statistical Modeling Fundamentals
- Linear regression assumptions  
- Residual analysis  
- Bias–variance tradeoff  
- Statistical interpretation of machine learning models

#### Exploratory Statistical Analysis
- Distribution visualization (histograms, KDE, boxplots)  
- Outlier detection  
- Statistical comparison of groups

#### Applied Statistics for Data Science
- A/B testing fundamentals  
- Data-driven decision making  
- Statistical validation of model results

### 💼 Business & Product Skills
- ROI calculation for ML models
- Economic impact analysis (MRR, revenue saved)
- Technical documentation (100+ pages)
- Presentation and stakeholder communication
- Project portfolio management

---

## 🛠 Tools & Technologies

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
