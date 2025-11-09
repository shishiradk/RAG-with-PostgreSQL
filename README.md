# RAG with PostgreSQL

A Retrieval-Augmented Generation (RAG) system built with PostgreSQL and Timescale Vector for efficient vector storage and similarity search. This project implements a complete RAG pipeline that combines document ingestion, vector embeddings, and LLM-powered question answering.

## Features

- **Vector Storage**: Uses Timescale Vector for high-performance vector operations
- **Embedding Generation**: OpenAI embeddings for semantic search
- **LLM Integration**: Configurable LLM factory supporting multiple providers
- **RESTful API**: FastAPI-based endpoints for search and ingestion
- **Docker Support**: Containerized deployment with Docker Compose
- **Modular Architecture**: Clean separation of concerns with services and database layers

## Architecture

### System Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   API Layer     │───▶│  LLM Service    │
│                 │    │  (FastAPI)      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Vector Search  │◀──▶│ Vector Store    │    │  Embedding      │
│                 │    │ (Timescale Vec) │    │  Generation     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                ▲
                                │
┌─────────────────┐             │
│ Document Store  │─────────────┘
│ (CSV/Processed) │
└─────────────────┘
```

*Figure 1: High-level system architecture showing the flow from user queries through API, vector search, and LLM generation.*

### Data Flow Diagram

```
CSV Data ──▶ Data Processing ──▶ Embedding Generation ──▶ Vector Store
     │               │                       │                   │
     │               │                       │                   │
     ▼               ▼                       ▼                   ▼
   Raw Text     Structured Records     Vector Embeddings    PostgreSQL
   (Q&A pairs)   (JSON format)         (1536-dim)          (Timescale Vec)
```

*Figure 2: Data ingestion pipeline from CSV files to vectorized documents in the database.*

## Prerequisites

- Python 3.8+
- PostgreSQL with Timescale Vector extension
- OpenAI API key
- Docker (optional, for containerized deployment)

## Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RAG-with-PostgreSQL
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp app/.env app/.env
   # Edit .env with your configuration
   ```

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose -f docker/docker-compose.yml up --build
   ```

## Configuration

### Environment Variables

Create a `.env` file in the `app/` directory with the following variables:

```env
# Database Configuration
TIMESCALE_SERVICE_URL=postgres://user:password@localhost:5432/dbname

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Vector Store Settings
VECTOR_TABLE_NAME=documents
EMBEDDING_DIMENSIONS=1536
TIME_PARTITION_INTERVAL=7 days
```

### Settings Configuration

The application uses Pydantic settings for configuration. See `app/config/settings.py` for available options.

## Usage

### Running the Application

1. **Start the database**
   ```bash
   docker-compose -f docker/docker-compose.yml up db -d
   ```

2. **Run the application**
   ```bash
   python app.py
   ```

3. **Insert vector data**
   ```bash
   cd app
   python -c "from dotenv import load_dotenv; load_dotenv(); import insert_vectors"
   ```

### API Usage

The API provides endpoints for search and data management:

#### Search Endpoint

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are your shipping options?", "limit": 5}'
```

#### Insert Data Endpoint

```bash
curl -X POST "http://localhost:8000/insert" \
  -H "Content-Type: application/json" \
  -d '{"documents": [...]}'
```

### Python API

```python
from app.database.vector_store import VectorStore

# Initialize vector store
vec = VectorStore()

# Search for similar documents
results = vec.search("What are your return policies?", limit=3)
print(results)

# Insert new documents
import pandas as pd
df = pd.DataFrame([...])  # Your document data
vec.upsert(df)
```

## API Endpoints

### Search Endpoints

- `POST /search` - Perform semantic search
- `POST /search/filtered` - Search with metadata filters
- `POST /search/predicates` - Advanced search with predicates

### Data Management

- `POST /insert` - Insert new documents
- `DELETE /delete` - Delete documents by ID or filter
- `PUT /update` - Update existing documents

### Health Check

- `GET /health` - Application health status

## Project Structure

```
RAG-with-PostgreSQL/
├── app/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── database/
│   │   ├── __init__.py
│   │   └── vector_store.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_factory.py
│   │   └── synthesizer.py
│   ├── insert_vectors.py
│   ├── similarity_search.py
│   └── .env
├── api/
│   ├── __init__.py
│   ├── routes.py
│   └── logs/
├── data/
│   └── faq_dataset.csv
├── docker/
│   ├── docker-compose.yml
│   └── init.sql
├── logs/
├── requirements.txt
├── app.py
├── main.py
└── README.md
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
isort .
```

### Database Schema

The system creates the following tables:

- `documents` - Main document storage with vector embeddings
- Indexes for efficient similarity search

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure you're running from the correct directory or using proper Python paths.

2. **Database Connection Error**: Check your PostgreSQL connection string and ensure Timescale Vector is installed.

3. **OpenAI API Error**: Verify your API key and billing status.

4. **Embedding Generation Fails**: Check network connectivity and API rate limits.

### Logs

Application logs are stored in the `logs/` directory. Check these files for detailed error information.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Timescale Vector](https://github.com/timescale/timescale-vector) for vector database functionality
- [OpenAI](https://openai.com/) for embedding and LLM services
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework