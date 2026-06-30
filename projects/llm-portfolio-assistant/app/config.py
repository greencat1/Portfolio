import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


class Config:
    # Paths
    INDEX_PATH = 'app/data/embeddings/faiss.index'
    METADATA_PATH = 'app/data/embeddings/metadata.json'
   
    
    # Model settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    LLM_MODEL = os.getenv("LLM_MODEL", "tinyllama")
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    
    # RAG settings
    SEARCH_K = int(os.getenv("SEARCH_K", "20"))
    MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
    
    # API settings
    API_TITLE = os.getenv("API_TITLE", "RAG Portfolio API")
    API_VERSION = os.getenv("API_VERSION", "1.0.0")
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))

config = Config()
