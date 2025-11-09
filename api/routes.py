"""
FastAPI routes for the RAG application.
"""
import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add the parent directory to Python path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.vector_store import VectorStore
from app.services.synthesizer import Synthesizer

app = FastAPI(
    title="RAG with PostgreSQL API",
    description="API for RAG (Retrieval-Augmented Generation) with PostgreSQL/TimescaleDB",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    metadata_filter: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_count: int

class QARequest(BaseModel):
    question: str
    limit: int = 3
    metadata_filter: Optional[Dict[str, Any]] = None

class QAResponse(BaseModel):
    question: str
    answer: str
    thought_process: List[str]
    enough_context: bool
    sources: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    tables_available: List[str]
    timestamp: datetime

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "RAG with PostgreSQL API", "status": "running"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with detailed system status."""
    try:
        vec = VectorStore()
        
        # Check database connection and tables
        vec.cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('embeddings', 'test_embeddings')
        """)
        tables = [row[0] for row in vec.cur.fetchall()]
        
        vec.close()
        
        return HealthResponse(
            status="healthy",
            database_connected=True,
            tables_available=tables,
            timestamp=datetime.now()
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            database_connected=False,
            tables_available=[],
            timestamp=datetime.now()
        )

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search for similar documents using vector similarity."""
    try:
        vec = VectorStore()
        results = vec.search(
            query_text=request.query,
            limit=request.limit,
            metadata_filter=request.metadata_filter,
            return_dataframe=False
        )
        vec.close()
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_results.append({
                "id": str(result[0]),
                "metadata": result[1],
                "content": result[2],
                "distance": float(result[4]) if len(result) > 4 else None
            })
        
        return SearchResponse(
            results=serializable_results,
            total_count=len(serializable_results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/ask", response_model=QAResponse)
async def ask_question(request: QARequest):
    """Ask a question and get a synthesized answer with RAG."""
    try:
        vec = VectorStore()
        
        # Search for relevant context
        search_results = vec.search(
            query_text=request.question,
            limit=request.limit,
            metadata_filter=request.metadata_filter
        )
        vec.close()
        
        # Generate response using synthesizer
        response = Synthesizer.generate_response(
            question=request.question,
            context=search_results
        )
        
        # Prepare source documents
        sources = []
        if not search_results.empty:
            for _, row in search_results.head(request.limit).iterrows():
                source = {
                    "content": getattr(row, 'content', ''),
                    "category": getattr(row, 'category', 'Unknown') if hasattr(row, 'category') else 'Unknown'
                }
                sources.append(source)
        
        return QAResponse(
            question=request.question,
            answer=response.answer,
            thought_process=response.thought_process,
            enough_context=response.enough_context,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get database statistics."""
    try:
        vec = VectorStore()
        
        # Get total document count
        vec.cur.execute(f"SELECT COUNT(*) FROM embeddings")
        total_documents = vec.cur.fetchone()[0]
        
        # Get category distribution
        vec.cur.execute("""
            SELECT metadata->>'category' as category, COUNT(*) 
            FROM embeddings 
            WHERE metadata->>'category' IS NOT NULL 
            GROUP BY metadata->>'category'
        """)
        category_stats = {row[0]: row[1] for row in vec.cur.fetchall()}
        
        vec.close()
        
        return {
            "total_documents": total_documents,
            "category_distribution": category_stats,
            "vector_store": "pgvector",
            "embedding_dimensions": 1536
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

