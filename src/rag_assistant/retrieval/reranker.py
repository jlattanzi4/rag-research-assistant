"""Reranking module for improving retrieval quality."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from rag_assistant.config import settings
from rag_assistant.retrieval.vector_store import SearchResult


@dataclass
class RerankResult:
    """Result after reranking."""

    content: str
    metadata: dict
    original_score: float
    rerank_score: float
    chunk_id: str


class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(
        self, query: str, results: list[SearchResult], top_k: int | None = None
    ) -> list[RerankResult]:
        """Rerank search results.

        Args:
            query: Original search query.
            results: Search results to rerank.
            top_k: Number of results to return after reranking.

        Returns:
            Reranked results.
        """
        pass


class LLMReranker(BaseReranker):
    """Reranker using LLM to score relevance."""

    def __init__(
        self,
        model: str | None = None,
        batch_size: int = 5,
    ) -> None:
        """Initialize LLM reranker.

        Args:
            model: LLM model to use.
            batch_size: Number of documents to score at once.
        """
        self.model = model or settings.llm_model
        self.batch_size = batch_size
        self._llm = None

    def _get_llm(self):
        """Lazy load LLM."""
        if self._llm is None:
            from langchain_anthropic import ChatAnthropic

            self._llm = ChatAnthropic(
                model=self.model,
                temperature=0,
                max_tokens=100,
                anthropic_api_key=settings.anthropic_api_key,
            )
        return self._llm

    def _score_document(self, query: str, document: str) -> float:
        """Score a single document for relevance.

        Args:
            query: Search query.
            document: Document text.

        Returns:
            Relevance score (0-1).
        """
        prompt = f"""Rate how relevant this document is to the query on a scale of 0 to 10.

Query: {query}

Document: {document[:1000]}

Respond with ONLY a number from 0 to 10. Nothing else."""

        llm = self._get_llm()
        response = llm.invoke(prompt)

        try:
            score = float(response.content.strip())
            return min(max(score / 10.0, 0.0), 1.0)  # Normalize to 0-1
        except ValueError:
            return 0.5  # Default score if parsing fails

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int | None = None
    ) -> list[RerankResult]:
        """Rerank using LLM scoring.

        Args:
            query: Original search query.
            results: Search results to rerank.
            top_k: Number of results to return.

        Returns:
            Reranked results.
        """
        top_k = top_k or len(results)

        if not results:
            return []

        reranked = []
        for result in results:
            score = self._score_document(query, result.content)
            reranked.append(
                RerankResult(
                    content=result.content,
                    metadata=result.metadata,
                    original_score=result.score,
                    rerank_score=score,
                    chunk_id=result.chunk_id,
                )
            )

        # Sort by rerank score
        reranked.sort(key=lambda x: x.rerank_score, reverse=True)
        return reranked[:top_k]


class CohereReranker(BaseReranker):
    """Reranker using Cohere's rerank API (placeholder for future implementation)."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize Cohere reranker.

        Args:
            api_key: Cohere API key.
        """
        self.api_key = api_key

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int | None = None
    ) -> list[RerankResult]:
        """Rerank using Cohere (not implemented).

        This is a placeholder for future Cohere integration.
        """
        raise NotImplementedError("Cohere reranking not yet implemented")


class SimpleReranker(BaseReranker):
    """Simple reranker based on keyword matching and length."""

    def __init__(
        self,
        keyword_weight: float = 0.3,
        length_weight: float = 0.1,
        original_weight: float = 0.6,
    ) -> None:
        """Initialize simple reranker.

        Args:
            keyword_weight: Weight for keyword match score.
            length_weight: Weight for length score.
            original_weight: Weight for original retrieval score.
        """
        self.keyword_weight = keyword_weight
        self.length_weight = length_weight
        self.original_weight = original_weight

    def _keyword_score(self, query: str, document: str) -> float:
        """Calculate keyword overlap score.

        Args:
            query: Search query.
            document: Document text.

        Returns:
            Keyword overlap score (0-1).
        """
        import re

        query_terms = set(re.findall(r"\b\w+\b", query.lower()))
        doc_terms = set(re.findall(r"\b\w+\b", document.lower()))

        if not query_terms:
            return 0.0

        overlap = len(query_terms & doc_terms)
        return overlap / len(query_terms)

    def _length_score(self, document: str, target_length: int = 500) -> float:
        """Score document based on length (prefer medium-length docs).

        Args:
            document: Document text.
            target_length: Ideal document length.

        Returns:
            Length score (0-1).
        """
        length = len(document)
        if length == 0:
            return 0.0

        # Prefer documents around target length
        ratio = min(length, target_length) / max(length, target_length)
        return ratio

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int | None = None
    ) -> list[RerankResult]:
        """Rerank using simple heuristics.

        Args:
            query: Original search query.
            results: Search results to rerank.
            top_k: Number of results to return.

        Returns:
            Reranked results.
        """
        top_k = top_k or len(results)

        if not results:
            return []

        reranked = []
        for result in results:
            keyword_score = self._keyword_score(query, result.content)
            length_score = self._length_score(result.content)

            combined_score = (
                self.original_weight * result.score
                + self.keyword_weight * keyword_score
                + self.length_weight * length_score
            )

            reranked.append(
                RerankResult(
                    content=result.content,
                    metadata=result.metadata,
                    original_score=result.score,
                    rerank_score=combined_score,
                    chunk_id=result.chunk_id,
                )
            )

        # Sort by rerank score
        reranked.sort(key=lambda x: x.rerank_score, reverse=True)
        return reranked[:top_k]


def get_reranker(
    strategy: str = "simple",
    **kwargs,
) -> BaseReranker:
    """Factory function to get a reranker.

    Args:
        strategy: Reranking strategy ('simple', 'llm').
        **kwargs: Additional arguments for the reranker.

    Returns:
        Reranker instance.
    """
    strategies = {
        "simple": SimpleReranker,
        "llm": LLMReranker,
    }

    if strategy not in strategies:
        raise ValueError(
            f"Unknown reranking strategy: {strategy}. "
            f"Available: {list(strategies.keys())}"
        )

    return strategies[strategy](**kwargs)
