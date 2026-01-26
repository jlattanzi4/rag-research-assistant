"""Command-line interface for RAG Research Assistant."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from rag_assistant import __version__
from rag_assistant.pipeline import create_pipeline

app = typer.Typer(
    name="rag-assistant",
    help="RAG Research Assistant - Query your documents with AI",
    add_completion=False,
)
console = Console()


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or directory to ingest"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r", help="Recursively process directories"),
) -> None:
    """Ingest documents into the vector store."""
    pipeline = create_pipeline()

    if path.is_file():
        chunks = pipeline.ingest_file(path)
        console.print(f"[green]Successfully ingested {chunks} chunks[/green]")
    elif path.is_dir():
        chunks = pipeline.ingest_directory(path, recursive=recursive)
        console.print(f"[green]Successfully ingested {chunks} chunks from directory[/green]")
    else:
        console.print(f"[red]Error: Path not found: {path}[/red]")
        raise typer.Exit(1)


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of documents to retrieve"),
    show_sources: bool = typer.Option(True, "--sources/--no-sources", "-s", help="Show source documents"),
) -> None:
    """Query the RAG system with a question."""
    pipeline = create_pipeline()

    # Check if there are documents
    stats = pipeline.get_stats()
    if stats["total_documents"] == 0:
        console.print("[yellow]No documents in the vector store. Run 'ingest' first.[/yellow]")
        raise typer.Exit(1)

    console.print(f"\n[bold blue]Question:[/bold blue] {question}\n")

    # Get response
    response = pipeline.query(question, top_k=top_k)

    # Display answer
    console.print(Panel(
        Markdown(response.answer),
        title="[bold green]Answer[/bold green]",
        border_style="green",
    ))

    # Display sources if requested
    if show_sources and response.sources:
        console.print("\n[bold blue]Sources:[/bold blue]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", style="dim", width=3)
        table.add_column("Source", style="cyan")
        table.add_column("Relevance", justify="right")
        table.add_column("Preview", max_width=50)

        for i, source in enumerate(response.sources, 1):
            filename = source.metadata.get("filename", "Unknown")
            score = f"{source.score:.2%}"
            preview = source.content[:100].replace("\n", " ") + "..."
            table.add_row(str(i), filename, score, preview)

        console.print(table)


@app.command()
def search(
    query_text: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
) -> None:
    """Search documents without generating a response."""
    pipeline = create_pipeline()

    results = pipeline.search(query_text, top_k=top_k)

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    console.print(f"\n[bold blue]Search results for:[/bold blue] {query_text}\n")

    for i, result in enumerate(results, 1):
        filename = result.metadata.get("filename", "Unknown")
        chunk_idx = result.metadata.get("chunk_index", 0)
        score = result.score

        console.print(Panel(
            result.content,
            title=f"[cyan]{filename}[/cyan] (chunk {chunk_idx}, score: {score:.2%})",
            border_style="blue",
        ))
        console.print()


@app.command()
def stats() -> None:
    """Show statistics about the vector store."""
    pipeline = create_pipeline()
    stats_data = pipeline.get_stats()

    table = Table(title="Vector Store Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Documents", str(stats_data["total_documents"]))
    table.add_row("Collection Name", stats_data["collection_name"])
    table.add_row("Persist Directory", stats_data["persist_directory"])

    console.print(table)


@app.command()
def clear(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Clear all documents from the vector store."""
    if not confirm:
        confirm = typer.confirm("Are you sure you want to clear all documents?")

    if confirm:
        pipeline = create_pipeline()
        pipeline.clear()
        console.print("[green]Vector store cleared successfully[/green]")
    else:
        console.print("[yellow]Operation cancelled[/yellow]")


@app.command()
def interactive() -> None:
    """Start an interactive Q&A session."""
    pipeline = create_pipeline()

    stats_data = pipeline.get_stats()
    if stats_data["total_documents"] == 0:
        console.print("[yellow]No documents in the vector store. Run 'ingest' first.[/yellow]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]RAG Research Assistant[/bold]\n\n"
        f"Documents loaded: {stats_data['total_documents']}\n\n"
        f"Type your questions below. Type 'quit' or 'exit' to end the session.",
        title="Interactive Mode",
        border_style="blue",
    ))

    while True:
        console.print()
        question = typer.prompt("Question", default="")

        if question.lower() in ["quit", "exit", "q"]:
            console.print("[blue]Goodbye![/blue]")
            break

        if not question.strip():
            continue

        response = pipeline.query(question)

        console.print(Panel(
            Markdown(response.answer),
            title="[bold green]Answer[/bold green]",
            border_style="green",
        ))


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"[bold]RAG Research Assistant[/bold] v{__version__}")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
