FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Create a proper startup script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting RAG System..."\n\
\n\
# Start FastAPI in background\n\
echo "Starting FastAPI on port 8001..."\n\
uvicorn main:app --host 0.0.0.0 --port 8001 &\n\
\n\
# Wait a moment for FastAPI to start\n\
sleep 5\n\
\n\
# Start Streamlit in foreground (so Docker can manage it)\n\
echo "Starting Streamlit on port 8501..."\n\
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true\n\
\n\
wait' > start.sh && chmod +x start.sh

EXPOSE 8001 8501

CMD ["./start.sh"]