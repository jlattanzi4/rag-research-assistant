"""Context building and management for RAG generation."""

from dataclasses import dataclass

import tiktoken

from rag_assistant.config import settings
from rag_assistant.retrieval.vector_store import SearchResult


@dataclass
class ContextSource:
    """Represents a single source in the context."""

    source_number: int
    content: str
    filename: str
    chunk_index: int
    relevance_score: float
    token_count: int


@dataclass
class BuiltContext:
    """Result of context building."""

    formatted_context: str
    sources: list[ContextSource]
    total_tokens: int
    truncated: bool
    num_sources_used: int
    num_sources_available: int


class ContextBuilder:
    """Builds and manages context for LLM generation."""

    def __init__(
        self,
        max_context_tokens: int = 4000,
        encoding_name: str = "cl100k_base",
        include_metadata: bool = True,
        source_separator: str = "\n\n---\n\n",
    ) -> None:
        """Initialize the ContextBuilder.

        Args:
            max_context_tokens: Maximum tokens allowed for context.
            encoding_name: Tiktoken encoding name.
            include_metadata: Whether to include source metadata.
            source_separator: Separator between sources.
        """
        self.max_context_tokens = max_context_tokens
        self.include_metadata = include_metadata
        self.source_separator = source_separator

        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception:
            # Fallback to rough estimate
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens.
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimate: ~4 chars per token
            return len(text) // 4

    def _format_source(
        self,
        source_num: int,
        result: SearchResult,
    ) -> str:
        """Format a single source for inclusion in context.

        Args:
            source_num: Source number for citation.
            result: Search result to format.

        Returns:
            Formatted source string.
        """
        filename = result.metadata.get("filename", "Unknown")
        chunk_idx = result.metadata.get("chunk_index", 0)
        score = result.score

        if self.include_metadata:
            header = f"[Source {source_num}] {filename} (chunk {chunk_idx}, relevance: {score:.2f})"
            return f"{header}\n{result.content}"
        else:
            return f"[Source {source_num}]\n{result.content}"

    def build(
        self,
        search_results: list[SearchResult],
        query: str | None = None,
        min_score: float = 0.0,
    ) -> BuiltContext:
        """Build context from search results.

        Args:
            search_results: List of search results to include.
            query: Optional query for relevance filtering.
            min_score: Minimum relevance score to include.

        Returns:
            BuiltContext with formatted context and metadata.
        """
        if not search_results:
            return BuiltContext(
                formatted_context="No relevant documents found.",
                sources=[],
                total_tokens=0,
                truncated=False,
                num_sources_used=0,
                num_sources_available=0,
            )

        # Filter by minimum score
        filtered_results = [r for r in search_results if r.score >= min_score]

        if not filtered_results:
            return BuiltContext(
                formatted_context="No documents met the relevance threshold.",
                sources=[],
                total_tokens=0,
                truncated=False,
                num_sources_used=0,
                num_sources_available=len(search_results),
            )

        # Build context incrementally
        sources = []
        context_parts = []
        total_tokens = 0
        truncated = False

        for i, result in enumerate(filtered_results, 1):
            formatted = self._format_source(i, result)
            token_count = self.count_tokens(formatted)

            # Check if adding this source would exceed limit
            separator_tokens = self.count_tokens(self.source_separator) if context_parts else 0
            if total_tokens + token_count + separator_tokens > self.max_context_tokens:
                truncated = True
                break

            context_parts.append(formatted)
            total_tokens += token_count + separator_tokens

            sources.append(
                ContextSource(
                    source_number=i,
                    content=result.content,
                    filename=result.metadata.get("filename", "Unknown"),
                    chunk_index=result.metadata.get("chunk_index", 0),
                    relevance_score=result.score,
                    token_count=token_count,
                )
            )

        formatted_context = self.source_separator.join(context_parts)

        return BuiltContext(
            formatted_context=formatted_context,
            sources=sources,
            total_tokens=total_tokens,
            truncated=truncated,
            num_sources_used=len(sources),
            num_sources_available=len(search_results),
        )

    def build_with_priority(
        self,
        search_results: list[SearchResult],
        priority_keywords: list[str] | None = None,
    ) -> BuiltContext:
        """Build context with priority given to certain keywords.

        Args:
            search_results: List of search results.
            priority_keywords: Keywords to prioritize in results.

        Returns:
            BuiltContext with priority-sorted sources.
        """
        if not priority_keywords:
            return self.build(search_results)

        # Score results by keyword matches
        def keyword_score(result: SearchResult) -> float:
            content_lower = result.content.lower()
            matches = sum(1 for kw in priority_keywords if kw.lower() in content_lower)
            return matches / len(priority_keywords) if priority_keywords else 0

        # Sort by combined score (original + keyword bonus)
        sorted_results = sorted(
            search_results,
            key=lambda r: r.score + 0.2 * keyword_score(r),
            reverse=True,
        )

        return self.build(sorted_results)

    def estimate_response_tokens(
        self, context_tokens: int, prompt_tokens: int = 500
    ) -> int:
        """Estimate available tokens for response.

        Args:
            context_tokens: Tokens used by context.
            prompt_tokens: Estimated tokens for prompt template.

        Returns:
            Estimated tokens available for response.
        """
        # Assuming a model with 8192 context window
        model_context_limit = 8192
        return max(0, model_context_limit - context_tokens - prompt_tokens)


class DiversityAwareContextBuilder(ContextBuilder):
    """Context builder that ensures source diversity."""

    def __init__(
        self,
        max_context_tokens: int = 4000,
        max_chunks_per_source: int = 2,
        **kwargs,
    ) -> None:
        """Initialize with diversity constraints.

        Args:
            max_context_tokens: Maximum tokens for context.
            max_chunks_per_source: Maximum chunks from same source file.
            **kwargs: Additional arguments for parent class.
        """
        super().__init__(max_context_tokens=max_context_tokens, **kwargs)
        self.max_chunks_per_source = max_chunks_per_source

    def build(
        self,
        search_results: list[SearchResult],
        query: str | None = None,
        min_score: float = 0.0,
    ) -> BuiltContext:
        """Build context ensuring source diversity.

        Args:
            search_results: List of search results.
            query: Optional query.
            min_score: Minimum score threshold.

        Returns:
            BuiltContext with diverse sources.
        """
        # Group results by source file
        source_counts: dict[str, int] = {}
        diverse_results = []

        for result in search_results:
            if result.score < min_score:
                continue

            filename = result.metadata.get("filename", "Unknown")
            current_count = source_counts.get(filename, 0)

            if current_count < self.max_chunks_per_source:
                diverse_results.append(result)
                source_counts[filename] = current_count + 1

        return super().build(diverse_results, query, min_score=0.0)
