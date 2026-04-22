# 🚀 Churn Prediction Service

Production-ready machine learning service for predicting customer churn in telecom companies.

Built with **FastAPI**, **CatBoost**, **Streamlit**, and fully containerized with **Docker**.

---

## 🔗 Live Demo

- API Docs (Swagger): http://207.231.105.113:8000/docs  
- Dashboard: http://207.231.105.113:8501  

### Test API Keys
user
admin
dashboard

---

## 📌 Overview

This project predicts whether a customer will churn using machine learning.

It provides:

- 🔮 Real-time and batch predictions via REST API  
- 📊 Interactive dashboard for analytics  
- 🏷️ Manual labeling system (ground truth collection)  
- 🔁 Retraining pipeline for continuous improvement  

---

## ⚙️ Tech Stack

- **Backend:** FastAPI  
- **ML Model:** CatBoost  
- **Frontend:** Streamlit  
- **Database:** SQLite  
- **Containerization:** Docker + Docker Compose  
- **Auth:** API Keys + RBAC  

---

## 🧠 Model Performance

| Metric      | Value |
|------------|------|
| Recall     | ~96% |
| Precision  | ~39% |
| ROC-AUC    | ~0.84 |
| Accuracy   | ~71% |

> Model optimized for **high recall** — critical for churn prediction.

---

## 💰 Business Impact

- Detects **96% of churners**
- Saves up to **$4.9M/year** (for 100k customers)
- ROI: **33,000%**
- Payback period: **< 1 hour**

---

## 📁 Project Structure
.
├── app/ # Main application (FastAPI backend)
├── data/ # Database and data files (SQLite, datasets)
├── documentation/ # Full technical documentation
├── models/ # Trained ML models (.pkl pipelines)
├── notebooks/ # EDA, experiments, model development
├── presentation/ # Slides / demo materials
├── Dockerfile.api # FastAPI container
├── Dockerfile.dashboard # Streamlit container
├── docker-compose.yml # Multi-container orchestration
├── requirements.txt # Dependencies
├── .dockerignore # Ignore rules for Docker
└── README.md # Project description


---

## 📂 Directory Details

### `app/`
Core backend of the system:

- FastAPI endpoints (`/predict`, `/label`, `/admin`, etc.)
- Authentication (API keys, RBAC)
- Rate limiting
- Model loading and inference

---

### `data/`

- Initial data

---

### `models/`

- Serialized ML pipelines (`.pkl`, cloudpickle)


---

### `notebooks/`

- Exploratory Data Analysis (EDA)
- Feature engineering experiments
- Model comparison & tuning
- Creating DB

---

### `documentation/`

- Full technical documentation:
  - EDA
  - Feature engineering
  - Modeling
  - API design
  - Business impact

---

### `presentation/`

- Slides for demos / interviews
- Business + technical explanation

---

## 📦 Features

### 🔮 Prediction API

- `POST /predict` — single prediction  
- `POST /predict/batch` — batch predictions  

---

### 🏷️ Labeling System

- Add ground truth labels  
- View labeled / unlabeled data  
- Track labeling progress  

---

### 🔁 Retraining

- Incremental retraining  
- Full retraining  
- Model comparison  
- Model switching (zero downtime)  

---

### 🔐 Security

- API Key authentication  
- Role-based access:
  - user
  - admin  
- Rate limiting  

---

## 📊 Dashboard

Streamlit dashboard provides:

- Churn risk segmentation  
- Revenue at risk (MRR)  
- High-risk customers  
- Retention recommendations  

---

## 🏗️ Architecture
Client → FastAPI → ML Pipeline → SQLite
↓
Streamlit Dashboard

---

## 🐳 Installation (Docker)

```bash
git clone https://github.com/greencat1/Portfolio.git
cd pet_projects/customer-churn-prediction

docker-compose up --build

---

🔑 Example API Request
curl -X POST "http://localhost:8000/predict" \
-H "X-API-Key: user" \
-H "Content-Type: application/json" \
-d '{
  "gender": "Female",
  "SeniorCitizen": 0,
  "tenure": 1,
  "Contract": "Month-to-month",
  "MonthlyCharges": 29.85
}'

---

## 🧩 ML Pipeline

End-to-end pipeline:

- Data cleaning  
- Feature engineering  
- Encoding
- DropRedundant
- Scaling  
- CatBoost model  

**Fully automated:** raw data → prediction in one call  

---

## 🧪 Testing

- Unit tests  
- Integration tests  
- Rate limit tests  
- Transformer tests  

**All tests passed ✅**

---

## 🚀 Future Improvements

- PostgreSQL instead of SQLite  
- Monitoring (Prometheus + Grafana)  
- A/B testing for models  
- Customer segmentation  

---

## 👤 Author

**Ivan Sazontov**  
GitHub: https://github.com/greencat1  

---

## ⭐ Why this project matters

This is not just a model — it's a full ML production system:

- End-to-end pipeline  
- API + dashboard  
- Retraining loop  
- Real business impact  

