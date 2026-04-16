# app/core/key_input.py
import hashlib
import getpass
from app.core.database import get_db

def hash_key(api_key: str) -> str:
    """Hash API key using SHA-256"""
    return hashlib.sha256(api_key.encode()).hexdigest()

def input_and_save_keys():
    """Prompt user for API keys at startup and save hashes to database"""
    print("\n" + "="*60)
    print("API KEY INPUT (required for authentication)")
    print("="*60)
    
    # Input keys (hidden input)
    user_key = getpass.getpass("Enter USER_API_KEY: ").strip()
    admin_key = getpass.getpass("Enter ADMIN_API_KEY: ").strip()
    
    if not user_key or not admin_key:
        print("\n ERROR: Both API keys are required!")
        print("   Please restart the server and enter valid keys.\n")
        raise ValueError("Missing API keys")
    
    # Hash keys
    user_hash = hash_key(user_key)
    admin_hash = hash_key(admin_key)
    
    # Save to database (upsert)
    with get_db() as conn:
        # Insert or replace user key
        conn.execute('''
            INSERT OR REPLACE INTO api_keys (key_hash, name, role, rate_limit, is_active)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_hash, "default_user", "user", 100, 1))
        
        # Insert or replace admin key
        conn.execute('''
            INSERT OR REPLACE INTO api_keys (key_hash, name, role, rate_limit, is_active)
            VALUES (?, ?, ?, ?, ?)
        ''', (admin_hash, "admin", "admin", 1000, 1))
        
        conn.commit()
    
    print(f"\n Keys saved to database (stored as hashes)")
    print(f"   User key hash: {user_hash[:16]}...")
    print(f"   Admin key hash: {admin_hash[:16]}...")
    print("="*60 + "\n")
    
    return user_hash, admin_hash

def refresh_key_cache():
    """Load keys from database into memory cache"""
    from app.auth import KEY_CACHE
    with get_db() as conn:
        rows = conn.execute('''
            SELECT key_hash, name, role, rate_limit, is_active 
            FROM api_keys WHERE is_active = 1
        ''').fetchall()
    
    KEY_CACHE.clear()
    for row in rows:
        KEY_CACHE[row['key_hash']] = {
            'name': row['name'],
            'role': row['role'],
            'rate_limit': row['rate_limit'],
            'is_active': row['is_active']
        }
    
    print(f" Key cache refreshed: {len(KEY_CACHE)} active keys loaded")