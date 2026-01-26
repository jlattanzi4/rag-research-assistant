"""Configuration settings for RAG Research Assistant."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Provider: "ollama" (local) or "groq" (cloud, free)
    llm_provider: str = Field(
        default="ollama", description="LLM provider: 'ollama' or 'groq'"
    )

    # Ollama Settings (free, local)
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama server URL"
    )
    ollama_model: str = Field(
        default="llama3.2", description="Ollama model name"
    )

    # Groq Settings (free cloud API)
    groq_api_key: str = Field(
        default="", description="Groq API key (free at console.groq.com)"
    )
    groq_model: str = Field(
        default="llama-3.1-8b-instant", description="Groq model name"
    )

    # LLM Settings
    llm_temperature: float = Field(default=0.0, description="LLM temperature")
    llm_max_tokens: int = Field(default=1024, description="Max tokens for LLM response")

    # Embedding Settings (free, local via HuggingFace)
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="HuggingFace embedding model"
    )
    embedding_dimensions: int = Field(default=384, description="Embedding dimensions")

    # Chunking Settings
    chunk_size: int = Field(default=512, description="Chunk size in tokens")
    chunk_overlap: int = Field(default=50, description="Chunk overlap in tokens")

    # Vector Store Settings
    chroma_persist_directory: Path = Field(
        default=Path("./data/chroma"), description="ChromaDB persistence directory"
    )
    collection_name: str = Field(
        default="research_papers", description="ChromaDB collection name"
    )

    # Retrieval Settings
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    similarity_threshold: float = Field(
        default=0.7, description="Minimum similarity score"
    )

    # Optional: Paid API keys (if user wants to switch later)
    anthropic_api_key: str = Field(default="", description="Anthropic API key (optional)")
    openai_api_key: str = Field(default="", description="OpenAI API key (optional)")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Global settings instance
settings = Settings()


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
