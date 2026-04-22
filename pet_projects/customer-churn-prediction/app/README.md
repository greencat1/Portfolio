# `/app` — Core Application Directory

This directory contains the complete source code of the Churn Prediction Service — FastAPI backend, ML pipeline, dashboard, database layer, and tests.

---

## 📁 Directory Structure
app/
│
├── init.py # Package initializer
├── main.py # FastAPI entry point (23 endpoints)
├── auth.py # API key authentication + RBAC
├── rate_limit.py # Sliding window rate limiting (60 sec)
├── schemas.py # Pydantic models (request/response)
├── config.py # Environment variables & settings
│
├── core/ # Core infrastructure
│ ├── database.py # SQLite connection & table initialization
│ └── key_input.py # Interactive API key prompting (CLI)
│
├── scripts/ # Business logic modules
│ ├── predict.py # Single customer prediction
│ ├── predict_batch.py # Batch predictions
│ ├── put_lbl.py # Label CRUD operations
│ ├── model.py # Singleton model loader (cached)
│ ├── model_manager.py # Model list/switch/delete
│ ├── retrain.py # Incremental + full retraining
│ ├── evaluate_model.py # Model evaluation on new data
│ ├── transformers.py # 6 custom sklearn transformers
│ └── init_keys.py # Manual API key initialization
│
├── dashboard/ # Streamlit dashboard
│ └── dashboard.py # Real-time monitoring UI
│
├── data/ # Data storage
│ ├── DB/
│ │ └── churn.db # SQLite database (api_keys, new_data, raw_data)
│ ├── raw/ # Raw Telco dataset (CSV)
│ ├── processed/ # Processed data (if any)
│ └── new_data/ # New customer data
│
├── models/ # Trained ML models (.pkl)
│ ├── full_churn_pipeline.pkl
│ ├── full_churn_pipeline_cloud.pkl
│ ├── full_churn_pipeline_retrained_cloud.pkl
│ ├── inc_churn_pipeline_backup_cloud.pkl
│ └── inc_churn_pipeline_retrained_cloud.pkl
│
├── metrics/ # Model performance metrics (JSON)
│ ├── full_churn_pipeline_cloud_test_metrics.json
│ ├── full_churn_pipeline_retrained_cloud_test_metrics.json
│ ├── inc_churn_pipeline_backup_cloud_test_metrics.json
│ └── inc_churn_pipeline_retrained_cloud_test_metrics.json
│
├── logs/ # Application logs
│ └── app.log # Rotating JSON logs
│
├── tests/ # Unit tests (15 tests)
│ ├── test_auth.py # API key authentication tests
│ ├── test_integration.py # End-to-end API tests
│ ├── test_rate_limit.py # Rate limiting tests
│ └── test_transformers.py # Transformer tests
│
└── utils/ # Utilities
└── logger.py # Logging setup (console + file)


---

## 🔑 Key Modules Description

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `main.py` | FastAPI app with 23 endpoints | `@app.on_event("startup")`, endpoint definitions |
| `auth.py` | API key auth & RBAC | `verify_api_key()`, `require_user()`, `require_admin()` |
| `rate_limit.py` | Request throttling | `check_rate_limit()`, sliding window, 100-1000 req/min |
| `schemas.py` | Request/response validation | `PredictRequest`, `LabelUpdateRequest`, `ModelInfo` |
| `config.py` | Environment config | `Settings` class with `.env` support |
| `core/database.py` | SQLite connection | `get_db()`, `init_db()`, `hash_key()` |
| `core/key_input.py` | Key setup wizard | `input_and_save_keys()` — runs on first startup |
| `scripts/predict.py` | Single prediction | `make_prediction()` — saves to DB, returns result |
| `scripts/predict_batch.py` | Batch prediction | `predict_batch()` — efficient for multiple customers |
| `scripts/put_lbl.py` | Label management | `update_label()`, `batch_update_labels()`, `get_label_statistics()` |
| `scripts/model.py` | Model loader (singleton) | `load_model()` — cached in memory after first call |
| `scripts/model_manager.py` | Model lifecycle | `get_all_models()`, `switch_active_model()`, `delete_model()` |
| `scripts/retrain.py` | Retraining logic | `retrain_on_new_data()` (incremental), `full_retrain_on_combined_data()` |
| `scripts/evaluate_model.py` | Model testing | `test_model_on_new_data()`, `compare_models_on_new_data()` |
| `scripts/transformers.py` | Custom transformers | 6 classes: `TotalChargesCleaner`, `FeatureEngineer`, `CategoricalEncoder`, `DropRedundant`, `NumericalScaler` |
| `dashboard/dashboard.py` | Streamlit UI | Real-time metrics, charts, retention recommendations |
| `tests/` | Unit tests | 15 tests, 100% pass rate, 88% coverage |

---

## 🧪 Data Files

| Path | Content | Size |
|------|---------|------|
| `data/DB/churn.db` | SQLite with 3 tables | 
| `data/raw/telco_churn.csv` | Original dataset | 
| `models/*.pkl` | Trained CatBoost pipelines | 
| `metrics/*.json` | Test metrics (recall, precision, etc.) | 
| `logs/app.log` | Rotating JSON logs | 

---

👨‍💻 Author
Ivan Sazontov · GitHub

---

