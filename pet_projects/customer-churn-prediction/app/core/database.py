# app/core/database.py
"""
Database Core Module

WHAT THIS MODULE DOES:
Provides database connection, initialization, and utility functions for SQLite.
Acts as the data persistence layer for the entire application.

WHY SQLITE?
- Zero configuration (no separate database server needed)
- File-based (easy to backup, version control friendly)
- Sufficient for small to medium scale (thousands of customers)
- Perfect for prototyping and production up to ~100k records

DATABASE FILES:
- Location: app/data/DB/churn.db
- Created automatically on first use
- Persistent across container restarts (via volume mount)

TWO MAIN TABLES:
1. api_keys - Stores API key hashes for authentication
2. new_data - Stores customer data, predictions, and labels
"""

import sqlite3
import hashlib
from pathlib import Path


# ============================================================================
# DATABASE PATH CONFIGURATION
# ============================================================================

# Construct path to database file
# __file__ = /app/app/core/database.py
# .parent = /app/app/core
# .parent.parent = /app/app
# .parent.parent.parent = /app
# Then /data/DB/churn.db
#
# WHY THIS PATH?
# - Keeps database inside app directory (easy to mount in Docker)
# - Separate folder 'DB' for clarity
# - Path(__file__) ensures it works regardless of working directory
DB_PATH = Path(__file__).parent.parent / "data" / "DB" / "churn.db"


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def get_db():
    """
    Get database connection with row factory.
    
    WHAT IS ROW FACTORY?
    Converts SQLite rows from tuples to dictionary-like objects.
    Allows accessing columns by name: row['customerID'] instead of row[0]
    
    WHY CONTEXT MANAGER?
    Use with `with get_db() as conn:` for automatic commit/rollback.
    Connection closes automatically when exiting the block.
    
    EXAMPLE USAGE:
        with get_db() as conn:
            cursor = conn.execute("SELECT * FROM users")
            rows = cursor.fetchall()
            for row in rows:
                print(row['customerID'])  # Access by column name!
    
    RETURNS:
        sqlite3.Connection object with row_factory = sqlite3.Row
    
    NOTE:
        Caller is responsible for using the connection in a context manager.
        The connection will be automatically closed when the block exits.
    """
    # Create parent directories if they don't exist
    # parents=True creates all missing directories recursively
    # exist_ok=True prevents error if directory already exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Connect to SQLite database (creates file if doesn't exist)
    conn = sqlite3.connect(str(DB_PATH))
    
    # Set row factory to return dictionary-like rows
    # Without this, rows would be tuples (index-based access only)
    conn.row_factory = sqlite3.Row
    
    return conn


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def init_db():
    """
    Create all required tables if they don't exist.
    
    CALLED ON:
    - First application startup (via startup_event in main.py)
    - After database file deletion/recreation
    
    IDEMPOTENT:
    Safe to call multiple times (CREATE TABLE IF NOT EXISTS)
    
    TWO TABLES CREATED:
    
    1. api_keys - Authentication and authorization
       - Stores hashed API keys (never plain text!)
       - Each key has role (user/admin/dashboard) and rate_limit
       - is_active allows soft deletion (revocation)
    
    2. new_data - Customer predictions and labels
       - Stores all 19 customer features (same as training data)
       - prediction: model output (0 = stay, 1 = churn)
       - probability: model confidence (0.0 to 1.0)
       - churn_label: ground truth (NULL = unlabeled, "Yes"/"No" = labeled)
       - timestamps track when data was created and labeled
    """
    with get_db() as conn:
        
        # ========================================
        # TABLE 1: API Keys (Authentication)
        # ========================================
        conn.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                key_hash TEXT PRIMARY KEY,           -- SHA-256 hash of API key
                name TEXT,                           -- Human-readable identifier
                role TEXT,                           -- 'user', 'admin', or 'dashboard'
                rate_limit INTEGER DEFAULT 100,      -- Requests per minute
                is_active INTEGER DEFAULT 1,         -- 1 = active, 0 = revoked
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # ========================================
        # TABLE 2: Customer Data (Predictions & Labels)
        # ========================================
        conn.execute('''
            CREATE TABLE IF NOT EXISTS new_data (
                -- Primary key (auto-incrementing integer)
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                
                -- Customer identifier (unique, from Telco dataset)
                customer_id TEXT UNIQUE,
                
                -- ========== FEATURE COLUMNS (19 total) ==========
                -- These match exactly the Telco Customer Churn dataset
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
                
                -- ========== MODEL OUTPUTS ==========
                prediction INTEGER,       -- 0 = No Churn, 1 = Churn
                probability REAL,         -- Confidence score (0.0 to 1.0)
                
                -- ========== GROUND TRUTH LABELS ==========
                churn_label TEXT,         -- NULL (unlabeled), 'Yes', or 'No'
                
                -- ========== TIMESTAMPS ==========
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- When record was added
                label_timestamp TIMESTAMP                         -- When label was set
            )
        ''')
        
        # Commit all changes (automatic with context manager, but explicit is fine)
        conn.commit()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def hash_key(api_key: str) -> str:
    """
    Hash API key using SHA-256 for secure storage.
    
    WHY HASH?
    - Never store raw API keys in database
    - SHA-256 is one-way (can't recover original key from hash)
    - Rainbow table resistant (keys are random)
    
    HOW IT WORKS:
    Input:  "my_secret_key_123"
    Output: "a1b2c3d4e5f67890abcdef1234567890..." (64 hex chars)
    
    Same input always produces same hash (deterministic).
    This allows verification without storing original key.
    
    VERIFICATION PROCESS:
    1. User sends: X-API-Key: my_secret_key_123
    2. Server hashes: hash = sha256("my_secret_key_123")
    3. Server looks up hash in database
    4. If found → key is valid
    
    ARGS:
        api_key: Raw API key string (plain text)
    
    RETURNS:
        SHA-256 hash as 64-character hexadecimal string
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def load_keys_from_db():
    """
    Load all active API keys from database into memory cache.
    
    USED BY:
    - Rate limiting module (for fast lookups)
    - Auth module (as fallback when cache misses)
    
    WHAT IT RETURNS:
    Dictionary mapping key_hash → key_info
    
    STRUCTURE:
    {
        "a1b2c3d4...": {
            "key_hash": "a1b2c3d4...",
            "role": "admin",
            "rate_limit": 1000,
            "is_active": 1
        },
        ...
    }
    
    NOTE:
    This loads ALL active keys into memory.
    For 1000 keys, memory usage is negligible (~100KB).
    
    RETURNS:
        dict: key_hash -> row dict
    """
    with get_db() as conn:
        rows = conn.execute(
            "SELECT key_hash, role, rate_limit, is_active FROM api_keys WHERE is_active = 1"
        ).fetchall()
    
    # Convert list of rows to dictionary for O(1) lookups
    # key: key_hash, value: dict of row data
    return {row['key_hash']: dict(row) for row in rows}


