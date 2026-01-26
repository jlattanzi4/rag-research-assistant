"""Embedding service with caching support."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Protocol

from langchain_openai import OpenAIEmbeddings

from rag_assistant.config import settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        ...


class EmbeddingCache:
    """Simple file-based cache for embeddings."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        """Initialize the embedding cache.

        Args:
            cache_dir: Directory to store cached embeddings.
        """
        self.cache_dir = cache_dir or Path(".embedding_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, list[float]] = {}

    def _hash_text(self, text: str, model: str) -> str:
        """Generate a hash for cache key.

        Args:
            text: Text to hash.
            model: Model name to include in hash.

        Returns:
            Hash string.
        """
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, text: str, model: str) -> list[float] | None:
        """Get cached embedding.

        Args:
            text: Text to look up.
            model: Model name.

        Returns:
            Cached embedding or None.
        """
        key = self._hash_text(text, model)

        # Check memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]

        # Check file cache
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    embedding = json.load(f)
                    self._memory_cache[key] = embedding
                    return embedding
            except Exception:
                pass

        return None

    def set(self, text: str, model: str, embedding: list[float]) -> None:
        """Store embedding in cache.

        Args:
            text: Original text.
            model: Model name.
            embedding: Embedding vector.
        """
        key = self._hash_text(text, model)

        # Store in memory
        self._memory_cache[key] = embedding

        # Store in file
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()


class CachedEmbeddingService:
    """Embedding service with caching support."""

    def __init__(
        self,
        model: str | None = None,
        cache_dir: Path | None = None,
        use_cache: bool = True,
    ) -> None:
        """Initialize the cached embedding service.

        Args:
            model: Embedding model name.
            cache_dir: Directory for cache storage.
            use_cache: Whether to enable caching.
        """
        self.model = model or settings.embedding_model
        self.use_cache = use_cache

        # Initialize OpenAI embeddings
        self._embeddings = OpenAIEmbeddings(
            model=self.model,
            openai_api_key=settings.openai_api_key,
        )

        # Initialize cache
        self._cache = EmbeddingCache(cache_dir) if use_cache else None

        # Stats
        self._cache_hits = 0
        self._cache_misses = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents with caching.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        embeddings = []
        texts_to_embed = []
        indices_to_embed = []

        # Check cache for each text
        for i, text in enumerate(texts):
            if self._cache:
                cached = self._cache.get(text, self.model)
                if cached is not None:
                    embeddings.append((i, cached))
                    self._cache_hits += 1
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
                    self._cache_misses += 1
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        # Embed non-cached texts
        if texts_to_embed:
            new_embeddings = self._embeddings.embed_documents(texts_to_embed)

            # Cache new embeddings
            for text, embedding in zip(texts_to_embed, new_embeddings):
                if self._cache:
                    self._cache.set(text, self.model, embedding)

            # Add to results
            for idx, embedding in zip(indices_to_embed, new_embeddings):
                embeddings.append((idx, embedding))

        # Sort by original index and return just embeddings
        embeddings.sort(key=lambda x: x[0])
        return [e[1] for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query with caching.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector.
        """
        if self._cache:
            cached = self._cache.get(text, self.model)
            if cached is not None:
                self._cache_hits += 1
                return cached

        self._cache_misses += 1
        embedding = self._embeddings.embed_query(text)

        if self._cache:
            self._cache.set(text, self.model, embedding)

        return embedding

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats.
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_requests": total,
            "hit_rate": hit_rate,
            "model": self.model,
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self._cache:
            self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
