"""
FastAPI main application for RAG with PostgreSQL
"""
import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()  # This loads from .env in the same directory as main.py

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
    allow_origins=["*"],
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

@app.get("/debug-env")
async def debug_env():
    """Check environment variables"""
    return {
        "DATABASE_URL_set": bool(os.getenv("DATABASE_URL")),
        "TIMESCALE_SERVICE_URL_set": bool(os.getenv("TIMESCALE_SERVICE_URL")),
        "OPENAI_API_KEY_set": bool(os.getenv("OPENAI_API_KEY")),
    }

@app.get("/debug-db")
async def debug_database():
    """Debug database connection issues"""
    try:
        from app.config.settings import get_settings
        
        settings = get_settings()
        db_url = settings.database.service_url
        
        # Show what URL we're trying to use
        return {
            "database_url_used": db_url,
            "settings_loaded": True,
            "connection_attempt": "will try with this URL"
        }
        
    except Exception as e:
        return {
            "status": "error_in_settings",
            "error": str(e),
            "error_type": type(e).__name__
        }

@app.get("/debug-settings")
async def debug_settings():
    """Check what settings are actually being used"""
    from app.config.settings import get_settings
    
    settings = get_settings()
    return {
        "database_url": settings.database.service_url,
        "openai_key_set": settings.openai.api_key != "MISSING_OPENAI_KEY",
        "env_timescale_url": os.getenv("TIMESCALE_SERVICE_URL", "NOT_FOUND_IN_OS"),
        "env_database_url": os.getenv("DATABASE_URL", "NOT_FOUND_IN_OS")
    }

@app.get("/debug-db-connection")
async def debug_db_connection():
    """Test database connection directly with psycopg2"""
    try:
        import psycopg2
        from app.config.settings import get_settings
        
        settings = get_settings()
        db_url = settings.database.service_url
        
        # Try to connect directly
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        # Test basic query
        cur.execute("SELECT 1 as test")
        result = cur.fetchone()
        
        # Test embeddings table
        cur.execute("SELECT COUNT(*) FROM embeddings")
        count = cur.fetchone()[0]
        
        cur.close()
        conn.close()
        
        return {
            "status": "success",
            "connection": "working",
            "test_query": result[0],
            "embeddings_count": count,
            "database_url_preview": db_url[:30] + "..."  # Hide full URL
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "database_url_attempted": settings.database.service_url[:30] + "..."
        }

@app.get("/debug-vectorstore")
async def debug_vectorstore():
    """Test VectorStore initialization and basic operations"""
    try:
        from app.database.vector_store import VectorStore
        
        # Test VectorStore initialization
        vec = VectorStore()
        
        # Test a simple query through VectorStore
        vec.cur.execute("SELECT COUNT(*) FROM embeddings")
        count = vec.cur.fetchone()[0]
        
        vec.close()
        
        return {
            "status": "success",
            "vectorstore_initialized": True,
            "embeddings_count": count,
            "message": "VectorStore is working correctly"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "vectorstore_initialized": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@app.get("/test-db-connection")
async def test_db_connection():
    """Test database connection from FastAPI"""
    try:
        vec = VectorStore()
        
        # Test a simple query
        vec.cur.execute("SELECT COUNT(*) as count FROM embeddings")
        count = vec.cur.fetchone()[0]
        
        vec.close()
        
        return {
            "status": "success",
            "database": "connected",
            "embeddings_count": count,
            "message": f"Database is connected with {count} embeddings"
        }
    except Exception as e:
        return {
            "status": "error",
            "database": "disconnected", 
            "error": str(e)
        }

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)