from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class QueryRequest(BaseModel):
    """Request model for /ask endpoint"""
    question: str = Field(..., description="User question", min_length=1, max_length=500)
    k: Optional[int] = Field(20, description="Number of chunks to retrieve", ge=1, le=50)
    max_results: Optional[int] = Field(5, description="Max unique results to use", ge=1, le=10)

class ChunkInfo(BaseModel):
    """Information about a retrieved chunk"""
    text: str
    project: str
    type: str

class QueryResponse(BaseModel):
    """Response model for /ask endpoint"""
    answer: str
    sources: List[ChunkInfo]
    used_chunks: int

class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str
    chunks_loaded: int
    embedding_model: str
    llm_model: str

class SourceResponse(BaseModel):
    """Response model for /sources endpoint"""
    total_chunks: int
    projects: Dict[str, int]
    types: Dict[str, int]
    sample_chunks: List[Dict[str, Any]]