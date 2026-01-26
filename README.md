# RAG Research Assistant

An LLM-powered research assistant using Retrieval-Augmented Generation (RAG) to answer questions from your document collection. Features semantic search, hybrid retrieval, response quality assessment, and both API and web interfaces.

## Features

- **Multi-Format Document Ingestion**: Load PDF, Markdown, HTML, and text files with intelligent chunking strategies
- **Hybrid Search**: Combine semantic embeddings with BM25 keyword search for optimal retrieval
- **Reranking**: Optional cross-encoder reranking for improved relevance
- **Quality Assessment**: Automatic evaluation of response quality and hallucination risk
- **Multiple Prompt Styles**: Concise, detailed, academic, and conversational response modes
- **Source Attribution**: Every answer includes citations to source documents
- **REST API**: Full-featured FastAPI backend with OpenAPI documentation
- **Web Interface**: Streamlit-based UI for interactive Q&A
- **Docker Support**: Production-ready containerization with docker-compose

## Quick Start

### Prerequisites

- Python 3.11+
- Anthropic API key (for Claude)
- OpenAI API key (for embeddings)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-research-assistant.git
cd rag-research-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys
```

### CLI Usage

```bash
# Ingest sample documents
rag-assistant ingest data/sample/

# Ask a question
rag-assistant query "What is machine learning?"

# Semantic search
rag-assistant search "neural networks" --top-k 10

# Start interactive mode
rag-assistant interactive

# View statistics
rag-assistant stats
```

### API Server

```bash
# Start the API server
uvicorn rag_assistant.api.app:app --reload

# API documentation available at http://localhost:8000/docs
```

### Streamlit Frontend

```bash
# Start the Streamlit app
streamlit run src/rag_assistant/frontend/app.py

# Access at http://localhost:8501
```

### Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Project Structure

```
rag-research-assistant/
├── src/rag_assistant/
│   ├── ingestion/          # Document loading and chunking
│   │   ├── loaders.py      # PDF, Markdown, HTML, Text loaders
│   │   ├── chunking.py     # Chunking strategies
│   │   ├── metadata.py     # Metadata extraction
│   │   └── batch.py        # Batch processing
│   ├── retrieval/          # Vector store and search
│   │   ├── vector_store.py # ChromaDB implementation
│   │   ├── embeddings.py   # Cached embeddings service
│   │   ├── hybrid.py       # BM25 + semantic hybrid search
│   │   └── reranker.py     # Reranking implementations
│   ├── generation/         # LLM response generation
│   │   ├── llm_service.py  # Claude integration
│   │   ├── prompts.py      # Prompt templates
│   │   ├── context.py      # Context building
│   │   └── quality.py      # Quality assessment
│   ├── evaluation/         # RAG evaluation
│   │   ├── metrics.py      # MRR, Precision@K, NDCG
│   │   └── dataset.py      # Evaluation datasets
│   ├── api/                # FastAPI backend
│   │   ├── app.py          # Application setup
│   │   ├── routes.py       # API endpoints
│   │   └── models.py       # Pydantic schemas
│   ├── frontend/           # Streamlit UI
│   │   ├── app.py          # Main application
│   │   ├── components.py   # UI components
│   │   └── api_client.py   # API client
│   ├── config.py           # Configuration
│   ├── pipeline.py         # Main RAG pipeline
│   └── cli.py              # Command-line interface
├── tests/                  # Test suite
├── data/sample/            # Sample documents
├── notebooks/              # Jupyter notebooks
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Service orchestration
└── .github/workflows/      # CI/CD pipelines
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/query/` | POST | Ask a question |
| `/search/` | POST | Semantic search |
| `/ingest/file` | POST | Ingest files |
| `/ingest/text` | POST | Ingest text |
| `/admin/stats` | GET | Get statistics |
| `/admin/clear` | DELETE | Clear index |

## Technology Stack

- **LLM Framework**: LangChain
- **LLM Provider**: Anthropic Claude (claude-3-haiku, claude-3-sonnet)
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Database**: ChromaDB
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Streamlit
- **HTTP Client**: HTTPX
- **Testing**: pytest + pytest-cov
- **Containerization**: Docker + docker-compose

## Configuration

Configuration via environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key | Required |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `LLM_MODEL` | Claude model | claude-3-haiku-20240307 |
| `EMBEDDING_MODEL` | Embedding model | text-embedding-3-small |
| `CHUNK_SIZE` | Document chunk size | 512 |
| `CHUNK_OVERLAP` | Chunk overlap | 50 |
| `TOP_K` | Results to retrieve | 5 |
| `CHROMA_PERSIST_DIRECTORY` | ChromaDB path | ./data/chroma |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -v

# Run tests with coverage
pytest --cov=src/rag_assistant --cov-report=html

# Linting
ruff check src/ tests/

# Formatting
ruff format src/ tests/

# Type checking
mypy src/
```

## Evaluation Metrics

The system includes evaluation tools for measuring retrieval quality:

- **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks
- **Precision@K**: Proportion of relevant docs in top K
- **Recall@K**: Proportion of relevant docs retrieved
- **NDCG (Normalized DCG)**: Graded relevance scoring

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

Joseph Lattanzi
