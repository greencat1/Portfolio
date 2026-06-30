"""
Simple logging setup for the API
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
import os

# Create logs folder if it doesn't exist
os.makedirs("app/logs", exist_ok=True)

# Configure logger
logger = logging.getLogger("churn_api")
logger.setLevel(logging.INFO)

# Format: time - name - level - message
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 1. Console output (for development and Docker logs)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 2. File output with rotation (10MB max, keep 3 backups)
file_handler = RotatingFileHandler(
    "app/logs/app.log",
    maxBytes=10_000_000,  # 10 MB
    backupCount=3,
    encoding="utf-8"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)