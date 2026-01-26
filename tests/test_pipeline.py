"""Tests for the RAG pipeline."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestRAGPipeline:
    """Tests for RAGPipeline class."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components for the pipeline."""
        with patch("rag_assistant.pipeline.ChromaVectorStore") as mock_vs, \
             patch("rag_assistant.pipeline.LLMService") as mock_llm, \
             patch("rag_assistant.pipeline.TextLoader") as mock_loader:

            # Setup mock vector store
            mock_vs_instance = MagicMock()
            mock_vs_instance.count.return_value = 10
            mock_vs_instance.collection_name = "test"
            mock_vs_instance.persist_directory = Path("/tmp/test")
            mock_vs.return_value = mock_vs_instance

            # Setup mock LLM service
            mock_llm_instance = MagicMock()
            mock_llm.return_value = mock_llm_instance

            # Setup mock loader
            mock_loader_instance = MagicMock()
            mock_loader.return_value = mock_loader_instance

            yield {
                "vector_store": mock_vs_instance,
                "llm_service": mock_llm_instance,
                "loader": mock_loader_instance,
            }

    @pytest.fixture
    def pipeline(self, mock_components):
        """Create a RAGPipeline with mocked components."""
        from rag_assistant.pipeline import RAGPipeline

        return RAGPipeline(
            vector_store=mock_components["vector_store"],
            llm_service=mock_components["llm_service"],
            text_loader=mock_components["loader"],
        )

    def test_get_stats(self, pipeline, mock_components):
        """Test getting pipeline statistics."""
        stats = pipeline.get_stats()

        assert "total_documents" in stats
        assert "collection_name" in stats
        assert "persist_directory" in stats
        assert stats["total_documents"] == 10

    def test_search(self, pipeline, mock_components):
        """Test search delegates to vector store."""
        from rag_assistant.retrieval.vector_store import SearchResult

        mock_results = [
            SearchResult(content="test", metadata={}, score=0.9, chunk_id="1")
        ]
        mock_components["vector_store"].search.return_value = mock_results

        results = pipeline.search("test query")

        mock_components["vector_store"].search.assert_called_once()
        assert results == mock_results

    def test_clear(self, pipeline, mock_components):
        """Test clearing the vector store."""
        pipeline.clear()

        mock_components["vector_store"].clear.assert_called_once()


class TestRAGPipelineIntegration:
    """Integration tests for RAG pipeline with sample data."""

    @pytest.fixture
    def sample_dir(self):
        """Create a temporary directory with sample files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create sample files
            (temp_path / "doc1.txt").write_text(
                "Machine learning is a field of artificial intelligence."
            )
            (temp_path / "doc2.txt").write_text(
                "Neural networks are computational models."
            )

            yield temp_path

    @pytest.mark.skip(reason="Requires API keys for full integration test")
    def test_full_pipeline(self, sample_dir):
        """Test the full RAG pipeline with real components."""
        from rag_assistant.pipeline import create_pipeline

        pipeline = create_pipeline()

        # Ingest documents
        chunks = pipeline.ingest_directory(sample_dir)
        assert chunks > 0

        # Query
        response = pipeline.query("What is machine learning?")
        assert response.answer
        assert len(response.sources) > 0
