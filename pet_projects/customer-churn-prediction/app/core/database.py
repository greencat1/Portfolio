# app/core/database.py
import sqlite3
import hashlib
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "DB" / "churn.db"

def get_db():
    """Get database connection"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create tables if they don't exist"""
    with get_db() as conn:
        # API keys table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                key_hash TEXT PRIMARY KEY,
                name TEXT,
                role TEXT,
                rate_limit INTEGER DEFAULT 100,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # New data table (for predictions and labels)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS new_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT UNIQUE,
                gender TEXT,
                SeniorCitizen INTEGER,
                Partner TEXT,
                Dependents TEXT,
                tenure INTEGER,
                PhoneService TEXT,
                MultipleLines TEXT,
                InternetService TEXT,
                OnlineSecurity TEXT,
                OnlineBackup TEXT,
                DeviceProtection TEXT,
                TechSupport TEXT,
                StreamingTV TEXT,
                StreamingMovies TEXT,
                Contract TEXT,
                PaperlessBilling TEXT,
                PaymentMethod TEXT,
                MonthlyCharges REAL,
                TotalCharges REAL,
                prediction INTEGER,
                probability REAL,
                churn_label TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                label_timestamp TIMESTAMP
            )
        ''')
        
        conn.commit()

def hash_key(api_key: str) -> str:
    """Hash API key using SHA-256"""
    return hashlib.sha256(api_key.encode()).hexdigest()

def load_keys_from_db():
    """Load keys from DB into memory cache (optional, for rate limiting)"""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT key_hash, role, rate_limit, is_active FROM api_keys WHERE is_active = 1"
        ).fetchall()
    return {row['key_hash']: dict(row) for row in rows}