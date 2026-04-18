# app/scripts/init_keys.py
"""
API Key Initialization Script

WHAT THIS SCRIPT DOES:
Manually enters API keys via command line and stores their hashes in the database.
Run this script once during initial setup or when keys need to be reset.

WHY THIS SCRIPT EXISTS:
The main application needs API keys to work, but we can't store default keys in code.
This script provides a secure way to create the initial admin and user keys.

WHEN TO RUN:
- First time setting up the application
- When keys were lost and need to be recreated
- When rotating keys (create new ones, then run this to replace old ones)

SECURITY NOTE:
Original keys are NEVER stored in database or files.
Only SHA-256 hashes are stored.
Once you close this terminal, you cannot recover the original keys.
Store them in a secure password manager!

HOW TO USE:
    python -m app.scripts.init_keys
    
Or from Docker:
    docker exec -it churn-api python -m app.scripts.init_keys
"""

import sys
from pathlib import Path
from app.utils.logger import logger

# Add project root to Python path so imports work
# This is needed when running script directly (not through module)
# Path(__file__) = /app/app/scripts/init_keys.py
# .parent = /app/app/scripts
# .parent.parent = /app/app
# .parent.parent.parent = /app
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.database import get_db, hash_key


# ============================================================================
# MAIN INITIALIZATION FUNCTION
# ============================================================================

def init_keys():
    """
    Enter API keys manually and save hashes to database.
    
    WORKFLOW:
    1. Prompt user for USER_API_KEY (paste or type)
    2. Prompt user for ADMIN_API_KEY (paste or type)
    3. Validate both keys are provided
    4. Hash both keys using SHA-256
    5. Store hashes in api_keys table
    6. Confirm success
    
    KEYS CREATED:
    - default_user: role="user", rate_limit=100
    - admin: role="admin", rate_limit=1000
    
    WHAT HAPPENS ON REPEAT RUN:
    - INSERT OR REPLACE overwrites existing keys
    - Old keys become invalid (hashes replaced)
    - New keys must be used from now on
    
    RETURNS:
        bool: True if successful, False otherwise
    
    EXAMPLE INTERACTION:
        ==================================================
        API KEY INITIALIZATION
        ==================================================
        Enter USER_API_KEY: user_mySecretKey123
        Enter ADMIN_API_KEY: admin_MySuperSecretKey456
        
        ✅ Keys saved to database (stored as hashes)
           Original keys are NOT stored anywhere
           Keep them safe! You won't see them again.
    """
    print("\n" + "="*50)
    print("API KEY INITIALIZATION")
    print("="*50)
    
    # ============================================
    # STEP 1: Get keys from user input
    # ============================================
    # Input() reads from standard input (keyboard)
    # .strip() removes leading/trailing whitespace
    user_key = input("Enter USER_API_KEY: ").strip()
    admin_key = input("Enter ADMIN_API_KEY: ").strip()
    
    # ============================================
    # STEP 2: Validate both keys are provided
    # ============================================
    if not user_key or not admin_key:
        print("❌ Error: Both keys are required!")
        logger.info(f"Error: Both keys are required!")
        return False
    
    # ============================================
    # STEP 3: Hash the keys
    # ============================================
    # Never store original keys!
    # Only store irreversible SHA-256 hashes
    user_hash = hash_key(user_key)
    admin_hash = hash_key(admin_key)
    
    # ============================================
    # STEP 4: Store hashes in database
    # ============================================
    with get_db() as conn:
        # Insert or replace user key
        # INSERT OR REPLACE: If key_hash exists, update; otherwise insert
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
    
    # ============================================
    # STEP 5: Confirm success
    # ============================================
    print("\n✅ Keys saved to database (stored as hashes)")
    print("   ⚠️  Original keys are NOT stored anywhere")
    print("   🔐 Keep them safe! You won't see them again.\n")
    logger.info(f"Primary keys have been successfully created and placed into the database.!")
    return True


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Run script directly:
        python app/scripts/init_keys.py
    
    This allows the script to be executed standalone.
    When run directly, __name__ == "__main__", so init_keys() is called.
    When imported as module, __name__ != "__main__", so nothing runs automatically.
    """
    init_keys()


