"""Pydantic models for API request/response schemas."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class PromptStyleEnum(str, Enum):
    """Available prompt styles."""

    CONCISE = "concise"
    DETAILED = "detailed"
    ACADEMIC = "academic"
    CONVERSATIONAL = "conversational"


class QualityLevel(str, Enum):
    """Response quality levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


# Request Models


class QueryRequest(BaseModel):
    """Request model for RAG query endpoint."""

    query: str = Field(..., min_length=1, max_length=2000, description="The question to ask")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of sources to retrieve")
    prompt_style: PromptStyleEnum = Field(
        default=PromptStyleEnum.DETAILED, description="Prompt style for response"
    )
    min_relevance: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum relevance score threshold"
    )
    use_reranking: bool = Field(default=False, description="Whether to use reranking")

    model_config = {"json_schema_extra": {"example": {"query": "What is machine learning?", "top_k": 5}}}


class SearchRequest(BaseModel):
    """Request model for semantic search endpoint."""

    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score")

    model_config = {"json_schema_extra": {"example": {"query": "neural networks", "top_k": 10}}}


class IngestRequest(BaseModel):
    """Request model for document ingestion endpoint."""

    file_path: str = Field(..., description="Path to file or directory to ingest")
    recursive: bool = Field(default=True, description="Recursively process directories")
    file_types: list[str] = Field(
        default=["txt", "pdf", "md", "html"],
        description="File extensions to process",
    )

    model_config = {"json_schema_extra": {"example": {"file_path": "/data/documents", "recursive": True}}}


class IngestTextRequest(BaseModel):
    """Request model for direct text ingestion."""

    content: str = Field(..., min_length=1, description="Text content to ingest")
    filename: str = Field(..., description="Filename for the document")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")

    model_config = {
        "json_schema_extra": {
            "example": {
                "content": "Machine learning is a subset of AI...",
                "filename": "ml_basics.txt",
                "metadata": {"source": "manual"},
            }
        }
    }


# Response Models


class SourceDocument(BaseModel):
    """A source document used in response generation."""

    source_number: int = Field(..., description="Citation number")
    content: str = Field(..., description="Document content")
    filename: str = Field(..., description="Source filename")
    chunk_index: int = Field(..., description="Chunk index within file")
    relevance_score: float = Field(..., description="Relevance score")


class QualityInfo(BaseModel):
    """Quality assessment information."""

    level: QualityLevel = Field(..., description="Quality level")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    has_citations: bool = Field(..., description="Whether response has citations")
    citation_count: int = Field(..., description="Number of citations")
    hallucination_risk: float = Field(..., ge=0.0, le=1.0, description="Hallucination risk")
    issues: list[str] = Field(default_factory=list, description="Identified issues")
    suggestions: list[str] = Field(default_factory=list, description="Improvement suggestions")


class QueryResponse(BaseModel):
    """Response model for RAG query endpoint."""

    answer: str = Field(..., description="Generated answer")
    sources: list[SourceDocument] = Field(..., description="Source documents used")
    quality: QualityInfo = Field(..., description="Response quality assessment")
    query: str = Field(..., description="Original query")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "Machine learning is a subset of AI [Source 1]...",
                "sources": [
                    {
                        "source_number": 1,
                        "content": "Machine learning...",
                        "filename": "ml_basics.txt",
                        "chunk_index": 0,
                        "relevance_score": 0.95,
                    }
                ],
                "quality": {
                    "level": "high",
                    "confidence": 0.92,
                    "has_citations": True,
                    "citation_count": 3,
                    "hallucination_risk": 0.1,
                    "issues": [],
                    "suggestions": [],
                },
                "query": "What is machine learning?",
                "processing_time_ms": 1234.5,
            }
        }
    }


class SearchResult(BaseModel):
    """A single search result."""

    content: str = Field(..., description="Document content")
    filename: str = Field(..., description="Source filename")
    chunk_index: int = Field(..., description="Chunk index")
    score: float = Field(..., description="Similarity score")
    metadata: dict = Field(default_factory=dict, description="Document metadata")


class SearchResponse(BaseModel):
    """Response model for search endpoint."""

    results: list[SearchResult] = Field(..., description="Search results")
    query: str = Field(..., description="Original query")
    total_results: int = Field(..., description="Number of results returned")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class IngestResponse(BaseModel):
    """Response model for ingestion endpoint."""

    success: bool = Field(..., description="Whether ingestion succeeded")
    documents_processed: int = Field(..., description="Number of documents processed")
    chunks_created: int = Field(..., description="Number of chunks created")
    errors: list[str] = Field(default_factory=list, description="Any errors encountered")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class StatsResponse(BaseModel):
    """Response model for stats endpoint."""

    total_documents: int = Field(..., description="Total documents in store")
    total_chunks: int = Field(..., description="Total chunks indexed")
    unique_files: int = Field(..., description="Number of unique source files")
    collection_name: str = Field(..., description="Vector store collection name")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Current timestamp")
    components: dict[str, bool] = Field(..., description="Component health status")


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Additional details")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Error timestamp")
