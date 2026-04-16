# app/scripts/init_keys.py
import sys
from pathlib import Path
from app.utils.logger import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.database import get_db, hash_key

def init_keys():
    """Enter API keys manually and save hashes to DB"""
    print("\n" + "="*50)
    print("API KEY INITIALIZATION")
    print("="*50)
    
    user_key = input("Enter USER_API_KEY: ").strip()
    admin_key = input("Enter ADMIN_API_KEY: ").strip()
    
    if not user_key or not admin_key:
        print("Error: Both keys are required!")
        logger.info(f"Error: Both keys are required!")
        return False
    
    user_hash = hash_key(user_key)
    admin_hash = hash_key(admin_key)
    
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
    
    print("\n Keys saved to database (stored as hashes)")
    print("   Original keys are NOT stored anywhere")
    print("   Keep them safe! You won't see them again.\n")
    logger.info(f"Primary keys have been successfully created and placed into the database.!")
    return True

if __name__ == "__main__":
    init_keys()