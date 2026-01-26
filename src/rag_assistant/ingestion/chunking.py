"""Text chunking strategies for document processing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_assistant.config import settings


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""

    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Split text into chunks.

        Args:
            text: Text to split.

        Returns:
            List of text chunks.
        """
        pass


class RecursiveChunker(BaseChunker):
    """Recursive character text splitter (default strategy)."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ) -> None:
        """Initialize the RecursiveChunker.

        Args:
            chunk_size: Maximum size of each chunk.
            chunk_overlap: Overlap between chunks.
            separators: Custom separators for splitting.
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        default_separators = ["\n\n", "\n", ". ", " ", ""]
        self.separators = separators or default_separators

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=self.separators,
        )

    def split(self, text: str) -> list[str]:
        """Split text using recursive character splitting."""
        return self.splitter.split_text(text)


class SentenceChunker(BaseChunker):
    """Sentence-based chunking that respects sentence boundaries."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_sentences: int = 1,
    ) -> None:
        """Initialize the SentenceChunker.

        Args:
            chunk_size: Target size for each chunk.
            chunk_overlap: Overlap between chunks in sentences.
            min_sentences: Minimum sentences per chunk.
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or 1  # sentences
        self.min_sentences = min_sentences

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re

        # Simple sentence splitting - handles common cases
        # Split on period, exclamation, question mark followed by space or end
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def split(self, text: str) -> list[str]:
        """Split text into chunks respecting sentence boundaries."""
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk_size and we have content
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_length

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


class SemanticChunker(BaseChunker):
    """Semantic chunking based on embedding similarity.

    This chunker groups sentences that are semantically similar together,
    creating more coherent chunks.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        similarity_threshold: float = 0.5,
        embedding_model: str | None = None,
    ) -> None:
        """Initialize the SemanticChunker.

        Args:
            chunk_size: Maximum size for each chunk.
            similarity_threshold: Threshold for grouping sentences (0-1).
            embedding_model: OpenAI embedding model to use.
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model or settings.embedding_model
        self._embeddings = None

    def _get_embeddings(self):
        """Lazy load embeddings model."""
        if self._embeddings is None:
            from langchain_openai import OpenAIEmbeddings

            self._embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=settings.openai_api_key,
            )
        return self._embeddings

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math

        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def split(self, text: str) -> list[str]:
        """Split text into semantically coherent chunks.

        This method:
        1. Splits text into sentences
        2. Embeds each sentence
        3. Groups adjacent sentences with high similarity
        4. Ensures chunks don't exceed max size
        """
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        if len(sentences) == 1:
            return sentences

        # Get embeddings for all sentences
        embeddings = self._get_embeddings()
        sentence_embeddings = embeddings.embed_documents(sentences)

        # Find breakpoints based on similarity drops
        breakpoints = []
        for i in range(len(sentences) - 1):
            similarity = self._cosine_similarity(
                sentence_embeddings[i], sentence_embeddings[i + 1]
            )
            if similarity < self.similarity_threshold:
                breakpoints.append(i + 1)

        # Create chunks based on breakpoints
        chunks = []
        start = 0

        for bp in breakpoints:
            chunk_text = " ".join(sentences[start:bp])

            # If chunk is too large, split it further
            if len(chunk_text) > self.chunk_size:
                # Use recursive splitter as fallback
                fallback = RecursiveChunker(
                    chunk_size=self.chunk_size, chunk_overlap=50
                )
                chunks.extend(fallback.split(chunk_text))
            else:
                chunks.append(chunk_text)

            start = bp

        # Handle remaining sentences
        if start < len(sentences):
            chunk_text = " ".join(sentences[start:])
            if len(chunk_text) > self.chunk_size:
                fallback = RecursiveChunker(
                    chunk_size=self.chunk_size, chunk_overlap=50
                )
                chunks.extend(fallback.split(chunk_text))
            else:
                chunks.append(chunk_text)

        return chunks


class ParagraphChunker(BaseChunker):
    """Paragraph-based chunking that keeps paragraphs together when possible."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """Initialize the ParagraphChunker.

        Args:
            chunk_size: Maximum size for each chunk.
            chunk_overlap: Character overlap between chunks.
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def split(self, text: str) -> list[str]:
        """Split text into chunks respecting paragraph boundaries."""
        # Split on double newlines (paragraph breaks)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if not paragraphs:
            return []

        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para_length = len(para)

            # If paragraph alone exceeds chunk_size, split it
            if para_length > self.chunk_size:
                # Save current chunk first
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Split large paragraph with recursive splitter
                splitter = RecursiveChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                chunks.extend(splitter.split(para))
                continue

            # If adding this paragraph exceeds limit
            if current_length + para_length + 2 > self.chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(para)
            current_length += para_length + 2  # +2 for \n\n

        # Don't forget last chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks


def get_chunker(
    strategy: str = "recursive",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    **kwargs,
) -> BaseChunker:
    """Factory function to get a chunker by strategy name.

    Args:
        strategy: Chunking strategy ('recursive', 'sentence', 'semantic', 'paragraph').
        chunk_size: Maximum chunk size.
        chunk_overlap: Overlap between chunks.
        **kwargs: Additional arguments for specific chunkers.

    Returns:
        Configured chunker instance.

    Raises:
        ValueError: If strategy is unknown.
    """
    strategies = {
        "recursive": RecursiveChunker,
        "sentence": SentenceChunker,
        "semantic": SemanticChunker,
        "paragraph": ParagraphChunker,
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown chunking strategy: {strategy}. "
                        f"Available: {list(strategies.keys())}")

    chunker_class = strategies[strategy]

    # Filter kwargs to only include valid arguments for this chunker
    import inspect
    valid_params = inspect.signature(chunker_class.__init__).parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return chunker_class(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **filtered_kwargs,
    )
