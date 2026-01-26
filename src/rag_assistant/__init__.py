"""RAG Research Assistant - An LLM-powered research assistant using RAG."""

__version__ = "0.1.0"

from rag_assistant.pipeline import RAGPipeline, create_pipeline

__all__ = ["RAGPipeline", "create_pipeline", "__version__"]
