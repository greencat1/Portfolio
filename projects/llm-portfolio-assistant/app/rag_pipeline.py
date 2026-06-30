import json
import numpy as np
import faiss
import requests
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from .config import config
from .schemes import ChunkInfo

class RAGPipeline:
    """Main RAG pipeline class"""
    
    def __init__(self):
        print(" Loading embedding model...")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        print(" Loading FAISS index...")
        
        
        self.index = faiss.read_index(config.INDEX_PATH)
        
        print(" Loading metadata...")
        with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        
        print(f" Ready! Loaded {len(self.chunks)} chunks")
    
    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks from FAISS"""
        q_emb = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(q_emb, k)
        return [self.chunks[i] for i in indices[0]]
    
    def deduplicate(self, results: List[Dict], max_results: int = 5) -> List[Dict]:
        """Remove duplicate chunks"""
        seen = set()
        unique = []
        
        for r in results:
            text_hash = r["text"][:200]  # First 200 chars as key
            if text_hash not in seen:
                seen.add(text_hash)
                unique.append(r)
            if len(unique) >= max_results:
                break
        
        return unique
    
    def build_context(self, results: List[Dict]) -> str:
        """Build context string from results"""
        return "\n\n---\n\n".join([r["text"] for r in results])
    
    def build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM with strict rules"""
        return f"""<|system|>
You are an ML project expert. Answer ONLY based on the context below.

RULES:
- Give direct, concise answers (1-3 sentences)
- Use exact numbers and model names
- If multiple options exist but one is "best" — state that one
- If answer not in context → say "I don't know"
- Never say "the documentation does not specify"
<|end|>

<|user|>
CONTEXT:
{context}

QUESTION: {query}

ANSWER (concise, factual):
<|end|>

<|assistant|>"""

    def ask_llm(self, prompt: str) -> str:
        """Call local Ollama LLM"""
        try:
            response = requests.post(
                f"{config.OLLAMA_URL}/api/generate",
                json={
                    "model": config.LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": config.TEMPERATURE,
                        "num_predict": config.MAX_TOKENS
                    }
                },
                timeout=60
            )
            return response.json()["response"]
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
    
    def answer(self, question: str, k: int = None, max_results: int = None) -> tuple:
        """Full RAG pipeline"""
        k = k or config.SEARCH_K
        max_results = max_results or config.MAX_RESULTS
        
        # 1. Search
        results = self.search(question, k=k)
        
        # 2. Deduplicate
        unique_results = self.deduplicate(results, max_results=max_results)
        
        # 3. Build context
        context = self.build_context(unique_results)
        
        # 4. Build prompt
        prompt = self.build_prompt(question, context)
        
        # 5. Get answer
        answer = self.ask_llm(prompt)
        
        # 6. Format sources
        sources = [ChunkInfo(
            text=r["text"][:500],  # Truncate for response
            project=r["project"],
            type=r["type"]
        ) for r in unique_results]
        
        return answer, sources, len(unique_results)
    
    def get_stats(self) -> dict:
        """Get statistics about loaded chunks"""
        projects = {}
        types = {}
        
        for chunk in self.chunks:
            proj = chunk.get("project", "unknown")
            typ = chunk.get("type", "general")
            projects[proj] = projects.get(proj, 0) + 1
            types[typ] = types.get(typ, 0) + 1
        
        return {
            "total_chunks": len(self.chunks),
            "projects": projects,
            "types": types
        }

# Global instance (load once at startup)
rag = None

def get_rag():
    """Singleton pattern for RAG pipeline"""
    global rag
    if rag is None:
        rag = RAGPipeline()
    return rag