"""Vector store implementations for semantic search."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_huggingface import HuggingFaceEmbeddings

from rag_assistant.config import settings
from rag_assistant.ingestion.loaders import DocumentChunk


@dataclass
class SearchResult:
    """Represents a search result from the vector store."""

    content: str
    metadata: dict
    score: float
    chunk_id: str


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add_documents(self, chunks: list[DocumentChunk]) -> list[str]:
        """Add document chunks to the vector store.

        Args:
            chunks: List of DocumentChunk objects to add.

        Returns:
            List of IDs for the added documents.
        """
        pass

    @abstractmethod
    def search(
        self, query: str, top_k: int | None = None, filter_metadata: dict | None = None
    ) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            query: Search query text.
            top_k: Number of results to return. Defaults to settings.
            filter_metadata: Optional metadata filter.

        Returns:
            List of SearchResult objects.
        """
        pass

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        """Delete documents by their IDs.

        Args:
            chunk_ids: List of chunk IDs to delete.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Get the number of documents in the store.

        Returns:
            Number of documents.
        """
        pass


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of the vector store with HuggingFace embeddings."""

    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: Path | None = None,
        embedding_model: str | None = None,
    ) -> None:
        """Initialize the ChromaDB vector store.

        Args:
            collection_name: Name of the collection. Defaults to settings.
            persist_directory: Directory for persistence. Defaults to settings.
            embedding_model: HuggingFace embedding model name. Defaults to settings.
        """
        self.collection_name = collection_name or settings.collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        self.embedding_model = embedding_model or settings.embedding_model

        # Ensure persist directory exists
        self.persist_directory = Path(self.persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Initialize HuggingFace embeddings (free, local)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": "cpu"},  # Use "mps" for M1/M2/M3 GPU acceleration
            encode_kwargs={"normalize_embeddings": True},
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        return self.embeddings.embed_documents(texts)

    def _embed_query(self, query: str) -> list[float]:
        """Generate embedding for a query.

        Args:
            query: Query text to embed.

        Returns:
            Embedding vector.
        """
        return self.embeddings.embed_query(query)

    def add_documents(self, chunks: list[DocumentChunk]) -> list[str]:
        """Add document chunks to ChromaDB.

        Args:
            chunks: List of DocumentChunk objects to add.

        Returns:
            List of IDs for the added documents.
        """
        if not chunks:
            return []

        # Extract texts and metadata
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]

        # Generate embeddings
        embeddings = self._embed_texts(texts)

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        return ids

    def search(
        self, query: str, top_k: int | None = None, filter_metadata: dict | None = None
    ) -> list[SearchResult]:
        """Search for similar documents in ChromaDB.

        Args:
            query: Search query text.
            top_k: Number of results to return. Defaults to settings.
            filter_metadata: Optional metadata filter.

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        top_k = top_k or settings.top_k

        # Generate query embedding
        query_embedding = self._embed_query(query)

        # Build query arguments
        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if filter_metadata:
            query_args["where"] = filter_metadata

        # Execute search
        results = self.collection.query(**query_args)

        # Convert to SearchResult objects
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances, convert to similarity scores
                # For cosine distance: similarity = 1 - distance
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance

                search_results.append(
                    SearchResult(
                        content=results["documents"][0][i],
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        score=score,
                        chunk_id=chunk_id,
                    )
                )

        return search_results

    def delete(self, chunk_ids: list[str]) -> None:
        """Delete documents by their IDs.

        Args:
            chunk_ids: List of chunk IDs to delete.
        """
        if chunk_ids:
            self.collection.delete(ids=chunk_ids)

    def count(self) -> int:
        """Get the number of documents in the collection.

        Returns:
            Number of documents.
        """
        return self.collection.count()

    def clear(self) -> None:
        """Clear all documents from the collection."""
        # Delete and recreate the collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_all_ids(self) -> list[str]:
        """Get all document IDs in the collection.

        Returns:
            List of all chunk IDs.
        """
        result = self.collection.get(include=[])
        return result["ids"] if result["ids"] else []

    def get_stats(self) -> dict:
        """Get statistics about the vector store.

        Returns:
            Dictionary with statistics.
        """
        total_chunks = self.count()

        # Get unique files from metadata
        unique_files = set()
        if total_chunks > 0:
            result = self.collection.get(include=["metadatas"])
            if result["metadatas"]:
                for metadata in result["metadatas"]:
                    if metadata and "filename" in metadata:
                        unique_files.add(metadata["filename"])

        return {
            "total_documents": total_chunks,
            "total_chunks": total_chunks,
            "unique_files": len(unique_files),
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
        }
