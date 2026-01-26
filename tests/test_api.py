"""Tests for the FastAPI endpoints."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from rag_assistant.api.app import app
from rag_assistant.api.models import QualityLevel
from rag_assistant.api.routes import get_pipeline
from rag_assistant.retrieval.vector_store import SearchResult


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline."""
    pipeline = MagicMock()
    pipeline.vector_store = MagicMock()
    pipeline.llm_service = MagicMock()
    return pipeline


@pytest.fixture
def client(mock_pipeline):
    """Create a test client with overridden dependencies."""
    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestHealthEndpoint:
    """Tests for the health endpoint."""

    def test_health_check(self, client, mock_pipeline):
        """Test health check returns expected structure."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert "components" in data

    def test_health_check_components(self, client, mock_pipeline):
        """Test health check with mock pipeline components."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["components"]["vector_store"] is True
        assert data["components"]["llm_service"] is True


class TestApiInfoEndpoint:
    """Tests for the API info endpoint."""

    def test_api_info(self, client, mock_pipeline):
        """Test API info returns expected data."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "RAG Research Assistant API"
        assert "version" in data
        assert "docs_url" in data


class TestQueryEndpoint:
    """Tests for the query endpoint."""

    def test_query_success(self, client, mock_pipeline):
        """Test successful query."""
        # Setup mock responses
        mock_pipeline.vector_store.search.return_value = [
            SearchResult(
                content="Machine learning is a subset of AI.",
                metadata={"filename": "ml.txt", "chunk_index": 0},
                score=0.95,
                chunk_id="test_0",
            )
        ]

        mock_response = MagicMock()
        mock_response.answer = "Machine learning is a subset of AI [Source 1]."
        mock_pipeline.llm_service.generate.return_value = mock_response

        response = client.post(
            "/query/",
            json={"query": "What is machine learning?", "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "quality" in data
        assert data["query"] == "What is machine learning?"

    def test_query_empty_results(self, client, mock_pipeline):
        """Test query with no results."""
        mock_pipeline.vector_store.search.return_value = []

        response = client.post(
            "/query/",
            json={"query": "What is quantum computing?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["quality"]["level"] == QualityLevel.INSUFFICIENT.value

    def test_query_validation_error(self, client):
        """Test query with invalid request."""
        response = client.post(
            "/query/",
            json={"query": ""},  # Empty query should fail validation
        )

        assert response.status_code == 422  # Validation error


class TestSearchEndpoint:
    """Tests for the search endpoint."""

    def test_search_success(self, client, mock_pipeline):
        """Test successful search."""
        mock_pipeline.vector_store.search.return_value = [
            SearchResult(
                content="Test content",
                metadata={"filename": "test.txt", "chunk_index": 0},
                score=0.9,
                chunk_id="test_0",
            ),
            SearchResult(
                content="More content",
                metadata={"filename": "test2.txt", "chunk_index": 0},
                score=0.8,
                chunk_id="test2_0",
            ),
        ]

        response = client.post(
            "/search/",
            json={"query": "test query", "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        assert data["total_results"] == 2
        assert data["query"] == "test query"

    def test_search_with_min_score(self, client, mock_pipeline):
        """Test search with minimum score filter."""
        mock_pipeline.vector_store.search.return_value = [
            SearchResult(
                content="High score",
                metadata={"filename": "test.txt", "chunk_index": 0},
                score=0.9,
                chunk_id="test_0",
            ),
            SearchResult(
                content="Low score",
                metadata={"filename": "test2.txt", "chunk_index": 0},
                score=0.3,
                chunk_id="test2_0",
            ),
        ]

        response = client.post(
            "/search/",
            json={"query": "test query", "min_score": 0.5},
        )

        assert response.status_code == 200
        data = response.json()
        # Only high-score result should be included
        assert data["total_results"] == 1


class TestIngestEndpoint:
    """Tests for the ingestion endpoints."""

    def test_ingest_text_success(self, client, mock_pipeline):
        """Test successful text ingestion."""
        mock_pipeline.ingest_text.return_value = {"documents": 1, "chunks": 3, "errors": []}

        response = client.post(
            "/ingest/text",
            json={
                "content": "This is test content for ingestion.",
                "filename": "test.txt",
                "metadata": {"source": "test"},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["documents_processed"] == 1
        assert data["chunks_created"] == 3

    def test_ingest_file_not_found(self, client, mock_pipeline):
        """Test file ingestion with non-existent path."""
        response = client.post(
            "/ingest/file",
            json={"file_path": "/nonexistent/path/file.txt"},
        )

        assert response.status_code == 404


class TestAdminEndpoints:
    """Tests for admin endpoints."""

    def test_get_stats(self, client, mock_pipeline):
        """Test getting stats."""
        mock_pipeline.vector_store.get_stats.return_value = {
            "total_documents": 100,
            "total_chunks": 100,
            "unique_files": 10,
            "collection_name": "test_collection",
        }

        response = client.get("/admin/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_documents"] == 100
        assert data["unique_files"] == 10

    def test_clear_without_confirmation(self, client, mock_pipeline):
        """Test clear index without confirmation fails."""
        response = client.delete("/admin/clear")

        assert response.status_code == 400
        assert "confirm" in response.json()["detail"].lower()

    def test_clear_with_confirmation(self, client, mock_pipeline):
        """Test clear index with confirmation succeeds."""
        response = client.delete("/admin/clear?confirm=true")

        assert response.status_code == 200
        mock_pipeline.vector_store.clear.assert_called_once()


class TestRequestValidation:
    """Tests for request validation."""

    def test_query_max_length(self, client):
        """Test query max length validation."""
        long_query = "a" * 2001  # Exceeds max length

        response = client.post(
            "/query/",
            json={"query": long_query},
        )

        assert response.status_code == 422

    def test_search_top_k_bounds(self, client):
        """Test top_k bounds validation."""
        # Test lower bound
        response = client.post(
            "/search/",
            json={"query": "test", "top_k": 0},
        )
        assert response.status_code == 422

        # Test upper bound
        response = client.post(
            "/search/",
            json={"query": "test", "top_k": 100},
        )
        assert response.status_code == 422

    def test_min_relevance_bounds(self, client):
        """Test min_relevance bounds validation."""
        response = client.post(
            "/query/",
            json={"query": "test", "min_relevance": 1.5},  # > 1.0
        )
        assert response.status_code == 422
