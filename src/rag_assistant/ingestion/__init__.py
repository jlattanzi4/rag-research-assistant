"""Document ingestion module for loading and processing documents."""

from .batch import BatchProcessor, BatchResult, ProcessingResult
from .chunking import (
    BaseChunker,
    ParagraphChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    get_chunker,
)
from .loaders import (
    BaseLoader,
    DocumentChunk,
    HTMLLoader,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
)
from .metadata import DocumentMetadata, MetadataExtractor

__all__ = [
    # Loaders
    "BaseLoader",
    "DocumentChunk",
    "HTMLLoader",
    "MarkdownLoader",
    "PDFLoader",
    "TextLoader",
    # Chunkers
    "BaseChunker",
    "ParagraphChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "SentenceChunker",
    "get_chunker",
    # Batch processing
    "BatchProcessor",
    "BatchResult",
    "ProcessingResult",
    # Metadata
    "DocumentMetadata",
    "MetadataExtractor",
]
