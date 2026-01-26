"""Tests for vector store implementations."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_assistant.ingestion.loaders import DocumentChunk


class TestChromaVectorStore:
    """Tests for ChromaVectorStore class."""

    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings."""
        with patch("rag_assistant.retrieval.vector_store.HuggingFaceEmbeddings") as mock:
            mock_instance = MagicMock()
            # Return embeddings matching input length (384 dimensions for all-MiniLM-L6-v2)
            mock_instance.embed_documents.side_effect = lambda texts: [[0.1] * 384 for _ in texts]
            mock_instance.embed_query.return_value = [0.1] * 384
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def vector_store(self, mock_embeddings):
        """Create a ChromaVectorStore instance with mocked embeddings."""
        from rag_assistant.retrieval.vector_store import ChromaVectorStore

        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaVectorStore(
                collection_name="test_collection",
                persist_directory=Path(temp_dir),
            )
            yield store

    @pytest.fixture
    def sample_chunks(self):
        """Create sample document chunks."""
        return [
            DocumentChunk(
                content="Machine learning is a subset of AI.",
                metadata={"source": "doc1.txt", "chunk_index": 0},
            ),
            DocumentChunk(
                content="Neural networks are inspired by the brain.",
                metadata={"source": "doc2.txt", "chunk_index": 0},
            ),
            DocumentChunk(
                content="RAG combines retrieval with generation.",
                metadata={"source": "doc3.txt", "chunk_index": 0},
            ),
        ]

    def test_add_documents(self, vector_store, sample_chunks):
        """Test adding documents to the store."""
        ids = vector_store.add_documents(sample_chunks)

        assert len(ids) == 3
        assert vector_store.count() == 3

    def test_add_empty_list(self, vector_store):
        """Test adding empty list returns empty."""
        ids = vector_store.add_documents([])
        assert ids == []

    def test_search(self, vector_store, sample_chunks):
        """Test searching for documents."""
        vector_store.add_documents(sample_chunks)

        results = vector_store.search("What is machine learning?", top_k=2)

        assert len(results) == 2
        assert all(hasattr(r, "content") for r in results)
        assert all(hasattr(r, "score") for r in results)

    def test_search_empty_store(self, vector_store):
        """Test searching in empty store returns empty."""
        results = vector_store.search("test query")
        assert results == []

    def test_delete_documents(self, vector_store, sample_chunks):
        """Test deleting documents."""
        ids = vector_store.add_documents(sample_chunks)
        assert vector_store.count() == 3

        vector_store.delete([ids[0]])
        assert vector_store.count() == 2

    def test_clear(self, vector_store, sample_chunks):
        """Test clearing all documents."""
        vector_store.add_documents(sample_chunks)
        assert vector_store.count() == 3

        vector_store.clear()
        assert vector_store.count() == 0

    def test_count(self, vector_store, sample_chunks):
        """Test counting documents."""
        assert vector_store.count() == 0

        vector_store.add_documents(sample_chunks[:1])
        assert vector_store.count() == 1

        vector_store.add_documents(sample_chunks[1:])
        assert vector_store.count() == 3

    def test_get_all_ids(self, vector_store, sample_chunks):
        """Test getting all document IDs."""
        ids = vector_store.add_documents(sample_chunks)

        all_ids = vector_store.get_all_ids()
        assert set(all_ids) == set(ids)
