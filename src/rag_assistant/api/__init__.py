"""API module for the RAG Research Assistant."""

from .app import app, create_app, main
from .models import (
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    IngestTextRequest,
    PromptStyleEnum,
    QualityInfo,
    QualityLevel,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SourceDocument,
    StatsResponse,
)
from .routes import router

__all__ = [
    # App
    "app",
    "create_app",
    "main",
    "router",
    # Request Models
    "QueryRequest",
    "SearchRequest",
    "IngestRequest",
    "IngestTextRequest",
    # Response Models
    "QueryResponse",
    "SearchResponse",
    "IngestResponse",
    "StatsResponse",
    "HealthResponse",
    "ErrorResponse",
    # Data Models
    "SourceDocument",
    "SearchResult",
    "QualityInfo",
    # Enums
    "PromptStyleEnum",
    "QualityLevel",
]
