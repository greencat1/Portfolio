from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .config import config
from .schemes import QueryRequest, QueryResponse, HealthResponse, SourceResponse
from .rag_pipeline import get_rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    print(" Starting RAG API...")
    get_rag()  # Initialize RAG pipeline
    print(" API ready!")
    yield
    print(" Shutting down...")

# Create FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description = """
# RAG Portfolio API

Ask questions about my Data Science & Machine Learning portfolio.

## 📊 Features
- **Semantic search** over 8,000+ chunks (FAISS on CPU)
- **Local LLM** (phi3:mini) for accurate, factual answers
- **BGE embeddings** (BAAI/bge-small-en-v1.5)
- **Sources included** in every response
- **Predefined questions** from the portfolio notebook

---

## 💡 Example Questions

### Churn Prediction
| Question | Expected Answer |
|----------|-----------------|
| "What model is used in churn prediction in api?" | CatBoost Classifier |
| "What is the recall of the best churn prediction model?" | 96% recall |
| "Why was recall chosen as the main metric for churn prediction?" | Critical to identify customers likely to leave, minimizes false negatives |
| "Describe the full pipeline of the churn prediction system." | TotalChargesCleaner → FeatureEngineer → CategoricalEncoder → DropRedundant → NumericalScaler |
| "What technologies were used to build the churn prediction API?" | FastAPI, CatBoost, Docker, Streamlit |

### Traffic Sign Detection
| Question | Expected Answer |
|----------|-----------------|
| "Describe the architecture of the traffic sign detection system." | Two-stage hybrid: YOLO + ResNet-18, 94.07% mAP50, 91.71% Recall |
| "How is the traffic sign detection system better than a baseline YOLO model?" | 94.07% mAP50 vs 81.3%, +12.7% improvement |
| "Why resnet?" | Proven effectiveness in fine-grained classification, pretrained on large dataset |
| "What technologies were used to build the sign detection API?" | FastAPI, Docker, YOLO + ResNet, WebSocket |

### Portfolio & Deployment
| Question | Expected Answer |
|----------|-----------------|
| "What machine learning skills does this portfolio demonstrate?" | Predictive modeling, Python-based data science workflows, real-world analytical problems |
| "How is the churn project deployed?" | Docker Compose on Ubuntu VPS, two containers (FastAPI + Streamlit) |
| "What makes this portfolio production-ready?" | Interactive dashboard, labeling system, ML pipeline, Docker, zero downtime |
| "Tell me about portfolio in general." | Collection of projects demonstrating progression from fundamentals to advanced ML solutions |

---
""",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "RAG Portfolio API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/ask", "/sources", "/health"]
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Check API health and status"""
    rag = get_rag()
    stats = rag.get_stats()
    return HealthResponse(
        status="ok",
        chunks_loaded=stats["total_chunks"],
        embedding_model=config.EMBEDDING_MODEL,
        llm_model=config.LLM_MODEL
    )

@app.get("/sources", response_model=SourceResponse, tags=["Info"])
async def get_sources():
    """Get information about loaded chunks"""
    rag = get_rag()
    stats = rag.get_stats()
    
    sample = [
        {"text": c["text"][:200], "project": c["project"], "type": c["type"]}
        for c in rag.chunks[:5]
    ]
    
    return SourceResponse(
        total_chunks=stats["total_chunks"],
        projects=stats["projects"],
        types=stats["types"],
        sample_chunks=sample
    )

@app.post("/ask", response_model=QueryResponse, tags=["RAG"])
async def ask_question(request: QueryRequest):
    """
    Ask a question about the portfolio.
    
    Returns answer based on retrieved chunks + sources.
    """
    try:
        rag = get_rag()
        answer, sources, used = rag.answer(
            question=request.question,
            k=request.k,
            max_results=request.max_results
        )
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            used_chunks=used
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

