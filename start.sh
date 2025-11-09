#!/bin/bash
set -e

echo "Starting RAG System..."

# Start FastAPI on port 8001
echo "Starting FastAPI on port 8001..."
cd app
uvicorn main:app --host 0.0.0.0 --port 8001 &

# Wait for FastAPI to start
sleep 10

# Start Streamlit on port 8501
echo "Starting Streamlit on port 8501..."
cd ..
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true

# Keep both services running
wait