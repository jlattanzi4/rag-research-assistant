"""Hybrid search combining semantic and keyword search."""

import math
import re
from collections import Counter
from dataclasses import dataclass

from rag_assistant.config import settings
from rag_assistant.retrieval.vector_store import ChromaVectorStore, SearchResult


@dataclass
class HybridSearchResult:
    """Result from hybrid search with combined score."""

    content: str
    metadata: dict
    semantic_score: float
    keyword_score: float
    combined_score: float
    chunk_id: str


class BM25:
    """BM25 ranking algorithm for keyword search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize BM25.

        Args:
            k1: Term frequency saturation parameter.
            b: Length normalization parameter.
        """
        self.k1 = k1
        self.b = b
        self.documents: list[list[str]] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_count: int = 0
        self.doc_freqs: dict[str, int] = {}  # term -> number of docs containing term
        self.idf_cache: dict[str, float] = {}

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens.
        """
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def fit(self, documents: list[str]) -> None:
        """Index documents for BM25.

        Args:
            documents: List of document texts.
        """
        self.documents = []
        self.doc_lengths = []
        self.doc_freqs = {}

        for doc in documents:
            tokens = self._tokenize(doc)
            self.documents.append(tokens)
            self.doc_lengths.append(len(tokens))

            # Count unique terms per document
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.doc_count = len(documents)
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 0
        )

        # Pre-compute IDF values
        self._compute_idf()

    def _compute_idf(self) -> None:
        """Pre-compute IDF values for all terms."""
        self.idf_cache = {}
        for term, df in self.doc_freqs.items():
            # IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
            self.idf_cache[term] = idf

    def score(self, query: str, doc_idx: int) -> float:
        """Calculate BM25 score for a document.

        Args:
            query: Query text.
            doc_idx: Index of document.

        Returns:
            BM25 score.
        """
        query_tokens = self._tokenize(query)
        doc_tokens = self.documents[doc_idx]
        doc_length = self.doc_lengths[doc_idx]

        # Count term frequencies in document
        doc_tf = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term not in self.idf_cache:
                continue

            idf = self.idf_cache[term]
            tf = doc_tf.get(term, 0)

            # BM25 TF component
            tf_component = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            )

            score += idf * tf_component

        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Search for top-k documents.

        Args:
            query: Query text.
            top_k: Number of results.

        Returns:
            List of (doc_idx, score) tuples.
        """
        scores = []
        for i in range(len(self.documents)):
            score = self.score(query, i)
            if score > 0:
                scores.append((i, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class HybridRetriever:
    """Combines semantic search with BM25 keyword search."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        """Initialize hybrid retriever.

        Args:
            vector_store: ChromaDB vector store.
            semantic_weight: Weight for semantic search scores (0-1).
            keyword_weight: Weight for keyword search scores (0-1).
            bm25_k1: BM25 k1 parameter.
            bm25_b: BM25 b parameter.
        """
        self.vector_store = vector_store
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

        # Initialize BM25
        self.bm25 = BM25(k1=bm25_k1, b=bm25_b)

        # Cache for document content
        self._documents: list[str] = []
        self._doc_ids: list[str] = []
        self._doc_metadata: list[dict] = []
        self._indexed = False

    def _build_bm25_index(self) -> None:
        """Build BM25 index from vector store documents."""
        # Get all documents from vector store
        all_ids = self.vector_store.get_all_ids()

        if not all_ids:
            self._indexed = True
            return

        # Retrieve documents in batches
        self._documents = []
        self._doc_ids = []
        self._doc_metadata = []

        # Get documents from ChromaDB
        result = self.vector_store.collection.get(
            ids=all_ids,
            include=["documents", "metadatas"],
        )

        if result["documents"]:
            self._documents = result["documents"]
            self._doc_ids = result["ids"]
            self._doc_metadata = result["metadatas"] or [{}] * len(self._documents)

        # Fit BM25
        self.bm25.fit(self._documents)
        self._indexed = True

    def search(
        self,
        query: str,
        top_k: int | None = None,
        semantic_top_k: int | None = None,
    ) -> list[HybridSearchResult]:
        """Perform hybrid search.

        Args:
            query: Search query.
            top_k: Number of final results.
            semantic_top_k: Number of candidates from semantic search.

        Returns:
            List of hybrid search results.
        """
        top_k = top_k or settings.top_k
        semantic_top_k = semantic_top_k or top_k * 2

        # Ensure BM25 index is built
        if not self._indexed:
            self._build_bm25_index()

        if not self._documents:
            return []

        # Get semantic search results
        semantic_results = self.vector_store.search(query, top_k=semantic_top_k)

        # Create a mapping of chunk_id to semantic result
        semantic_map: dict[str, SearchResult] = {
            r.chunk_id: r for r in semantic_results
        }

        # Get BM25 scores for all documents
        bm25_results = self.bm25.search(query, top_k=semantic_top_k)

        # Create a mapping of doc_idx to BM25 score
        bm25_map: dict[int, float] = {idx: score for idx, score in bm25_results}

        # Combine results
        combined_results: dict[str, HybridSearchResult] = {}

        # Normalize scores
        max_semantic = max((r.score for r in semantic_results), default=1.0)
        max_bm25 = max((score for _, score in bm25_results), default=1.0)

        # Add semantic results
        for result in semantic_results:
            normalized_semantic = result.score / max_semantic if max_semantic > 0 else 0

            # Find BM25 score if available
            try:
                doc_idx = self._doc_ids.index(result.chunk_id)
                bm25_score = bm25_map.get(doc_idx, 0)
                normalized_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0
            except ValueError:
                normalized_bm25 = 0

            combined_score = (
                self.semantic_weight * normalized_semantic
                + self.keyword_weight * normalized_bm25
            )

            combined_results[result.chunk_id] = HybridSearchResult(
                content=result.content,
                metadata=result.metadata,
                semantic_score=normalized_semantic,
                keyword_score=normalized_bm25,
                combined_score=combined_score,
                chunk_id=result.chunk_id,
            )

        # Add BM25 results not in semantic results
        for doc_idx, bm25_score in bm25_results:
            chunk_id = self._doc_ids[doc_idx]
            if chunk_id not in combined_results:
                normalized_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0
                combined_score = self.keyword_weight * normalized_bm25

                combined_results[chunk_id] = HybridSearchResult(
                    content=self._documents[doc_idx],
                    metadata=self._doc_metadata[doc_idx],
                    semantic_score=0.0,
                    keyword_score=normalized_bm25,
                    combined_score=combined_score,
                    chunk_id=chunk_id,
                )

        # Sort by combined score and return top_k
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x.combined_score,
            reverse=True,
        )

        return sorted_results[:top_k]

    def refresh_index(self) -> None:
        """Rebuild the BM25 index from vector store."""
        self._indexed = False
        self._build_bm25_index()
