# Pet Projects Portfolio

Collection of my data science and machine learning projects.

---

## 📁 Projects

| Project | Description | Tech Stack | Status |
|---------|-------------|------------|--------|
| **customer-churn-prediction** | Production-ready churn prediction service for telecom companies | FastAPI, CatBoost, Streamlit, Docker, SQLite | ✅ Complete |
| **traffic-sign-detection** | Traffic sign detection with YOLOv8 — 55 classes, cascade architecture, imbalance handling | YOLOv8, Python, Ultralytics, Albumentations | 🚧 In Progress |

---


## 🚀 Live Demo

- **Churn Prediction API:** [http://207.231.105.113:8000/docs](http://207.231.105.113:8000/docs)
- **Dashboard:** [http://207.231.105.113:8501](http://207.231.105.113:8501)

---

## 📊 Project Details

### 1. Customer Churn Prediction ✅

**Goal:** Predict which telecom customers are likely to churn.

**Key Results:**
- 84.5% recall in production
- 23 API endpoints with role-based auth
- Docker deployment (2 containers)

**Repository:** `customer-churn-prediction/`

---

### 2. Traffic Sign Detection 🚧

**Goal:** Detect and classify 55 types of traffic signs using YOLOv8.

**Experiments Planned:**
- Baseline on 55 classes
- Category-level detection (4 groups)
- Cascade architecture (category → specialized models)
- Oversampling + augmentations for rare classes

**Expected Results:**
- Compare 6 model configurations
- Analyze class imbalance impact
- Measure generalization gap (processed vs original test)

**Repository:** `traffic-sign-detection/` *(coming soon)*

---

## 👨‍💻 Author

**Ivan Sazontov**

- GitHub: [@greencat1](https://github.com/greencat1)
- Email: isazontov1@gmail.com

---
