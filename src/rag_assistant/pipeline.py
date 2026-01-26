"""Main RAG pipeline orchestrating ingestion, retrieval, and generation."""

from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm import tqdm

from rag_assistant.config import settings
from rag_assistant.generation.llm_service import LLMService, RAGResponse
from rag_assistant.ingestion.loaders import DocumentChunk, TextLoader
from rag_assistant.retrieval.vector_store import ChromaVectorStore, SearchResult

console = Console()


class RAGPipeline:
    """Main RAG pipeline for document Q&A."""

    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        llm_service: LLMService | None = None,
        text_loader: TextLoader | None = None,
    ) -> None:
        """Initialize the RAG pipeline.

        Args:
            vector_store: Vector store instance. Creates default if None.
            llm_service: LLM service instance. Creates default if None.
            text_loader: Text loader instance. Creates default if None.
        """
        self.vector_store = vector_store or ChromaVectorStore()
        self.llm_service = llm_service or LLMService()
        self.text_loader = text_loader or TextLoader()

    def ingest_file(self, file_path: Path | str) -> dict:
        """Ingest a single file into the vector store.

        Args:
            file_path: Path to the file to ingest.

        Returns:
            Dictionary with ingestion results.
        """
        file_path = Path(file_path)
        console.print(f"[blue]Loading file:[/blue] {file_path.name}")

        # Load and chunk the document
        chunks = self.text_loader.load(file_path)

        if not chunks:
            console.print(f"[yellow]Warning:[/yellow] No content found in {file_path.name}")
            return {"documents": 1, "chunks": 0, "errors": []}

        # Add to vector store
        self.vector_store.add_documents(chunks)

        console.print(f"[green]Ingested {len(chunks)} chunks from {file_path.name}[/green]")
        return {"documents": 1, "chunks": len(chunks), "errors": []}

    def ingest_directory(
        self,
        directory: Path | str,
        recursive: bool = True,
        show_progress: bool = True,
        file_types: list[str] | None = None,
    ) -> dict:
        """Ingest all documents from a directory.

        Args:
            directory: Path to the directory.
            recursive: Whether to search recursively.
            show_progress: Whether to show progress bar.
            file_types: List of file extensions to process (e.g., ["txt", "md"]).

        Returns:
            Dictionary with ingestion results.
        """
        directory = Path(directory)
        console.print(f"[blue]Scanning directory:[/blue] {directory}")

        file_types = file_types or ["txt"]
        files = []

        # Find files matching the specified types
        for ext in file_types:
            pattern = f"**/*.{ext}" if recursive else f"*.{ext}"
            files.extend(directory.glob(pattern))

        if not files:
            console.print(f"[yellow]No files found with extensions: {file_types}[/yellow]")
            return {"documents": 0, "chunks": 0, "errors": []}

        console.print(f"[blue]Found {len(files)} files to process[/blue]")

        total_chunks = 0
        total_docs = 0
        errors = []
        iterator = tqdm(files, desc="Ingesting files") if show_progress else files

        for file_path in iterator:
            try:
                chunks = self.text_loader.load(file_path)
                if chunks:
                    self.vector_store.add_documents(chunks)
                    total_chunks += len(chunks)
                    total_docs += 1
            except Exception as e:
                error_msg = f"Error processing {file_path.name}: {e}"
                console.print(f"[red]{error_msg}[/red]")
                errors.append(error_msg)

        console.print(f"[green]Total chunks ingested: {total_chunks}[/green]")
        return {"documents": total_docs, "chunks": total_chunks, "errors": errors}

    def ingest_text(
        self,
        content: str,
        filename: str = "direct_input.txt",
        metadata: dict | None = None,
    ) -> dict:
        """Ingest text content directly.

        Args:
            content: Text content to ingest.
            filename: Filename to use for the document.
            metadata: Additional metadata.

        Returns:
            Dictionary with ingestion results.
        """
        metadata = metadata or {}
        metadata["filename"] = filename
        metadata["source"] = "direct_input"

        # Create chunks from content
        chunks = self.text_loader.chunk_text(content, metadata)

        if not chunks:
            return {"documents": 1, "chunks": 0, "errors": []}

        # Add to vector store
        self.vector_store.add_documents(chunks)

        console.print(f"[green]Ingested {len(chunks)} chunks from direct input[/green]")
        return {"documents": 1, "chunks": len(chunks), "errors": []}

    def search(
        self, query: str, top_k: int | None = None
    ) -> list[SearchResult]:
        """Search for relevant documents.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            List of search results.
        """
        return self.vector_store.search(query, top_k=top_k)

    def query(
        self,
        question: str,
        top_k: int | None = None,
        show_sources: bool = True,
    ) -> RAGResponse:
        """Query the RAG system with a question.

        Args:
            question: User's question.
            top_k: Number of documents to retrieve.
            show_sources: Whether to display source information.

        Returns:
            RAGResponse with answer and sources.
        """
        top_k = top_k or settings.top_k

        # Retrieve relevant documents
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Searching documents...", total=None)
            search_results = self.vector_store.search(question, top_k=top_k)

        if not search_results:
            console.print("[yellow]No relevant documents found[/yellow]")
            return RAGResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                query=question,
            )

        # Generate response
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Generating response...", total=None)
            response = self.llm_service.generate(question, search_results)

        return response

    def get_stats(self) -> dict:
        """Get statistics about the vector store.

        Returns:
            Dictionary with statistics.
        """
        return {
            "total_documents": self.vector_store.count(),
            "collection_name": self.vector_store.collection_name,
            "persist_directory": str(self.vector_store.persist_directory),
        }

    def clear(self) -> None:
        """Clear all documents from the vector store."""
        self.vector_store.clear()
        console.print("[green]Vector store cleared[/green]")


def create_pipeline() -> RAGPipeline:
    """Factory function to create a RAG pipeline with default settings.

    Returns:
        Configured RAGPipeline instance.
    """
    return RAGPipeline()
