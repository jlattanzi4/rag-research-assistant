"""Batch processing utilities for large document sets."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm import tqdm

from rag_assistant.ingestion.loaders import (
    BaseLoader,
    DocumentChunk,
    HTMLLoader,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
)

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class ProcessingResult:
    """Result of processing a single file."""

    file_path: Path
    success: bool
    chunks_created: int = 0
    error: str = ""
    processing_time: float = 0.0


@dataclass
class BatchResult:
    """Result of batch processing."""

    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    total_time: float = 0.0
    results: list[ProcessingResult] = field(default_factory=list)
    errors: list[tuple[Path, str]] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a summary string."""
        return (
            f"Processed {self.total_files} files: "
            f"{self.successful_files} successful, {self.failed_files} failed. "
            f"Created {self.total_chunks} chunks in {self.total_time:.2f}s"
        )


class BatchProcessor:
    """Process large batches of documents with progress tracking and error handling."""

    # Map file extensions to loaders
    LOADER_MAP: dict[str, type[BaseLoader]] = {
        ".txt": TextLoader,
        ".text": TextLoader,
        ".pdf": PDFLoader,
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
        ".html": HTMLLoader,
        ".htm": HTMLLoader,
    }

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        show_progress: bool = True,
        continue_on_error: bool = True,
        save_state: bool = False,
        state_file: Path | None = None,
    ) -> None:
        """Initialize the BatchProcessor.

        Args:
            chunk_size: Chunk size for all loaders.
            chunk_overlap: Chunk overlap for all loaders.
            show_progress: Whether to show progress bar.
            continue_on_error: Continue processing if a file fails.
            save_state: Save processing state for resumability.
            state_file: Path to save state file.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.show_progress = show_progress
        self.continue_on_error = continue_on_error
        self.save_state = save_state
        self.state_file = state_file or Path(".batch_state.json")

        # Initialize loaders
        self._loaders: dict[str, BaseLoader] = {}

    def _get_loader(self, extension: str) -> BaseLoader | None:
        """Get or create a loader for the given extension.

        Args:
            extension: File extension (with dot).

        Returns:
            Loader instance or None if unsupported.
        """
        extension = extension.lower()

        if extension not in self.LOADER_MAP:
            return None

        if extension not in self._loaders:
            loader_class = self.LOADER_MAP[extension]
            kwargs = {}
            if self.chunk_size:
                kwargs["chunk_size"] = self.chunk_size
            if self.chunk_overlap:
                kwargs["chunk_overlap"] = self.chunk_overlap
            self._loaders[extension] = loader_class(**kwargs)

        return self._loaders[extension]

    def _discover_files(
        self,
        directory: Path,
        recursive: bool = True,
        extensions: list[str] | None = None,
    ) -> list[Path]:
        """Discover all processable files in a directory.

        Args:
            directory: Directory to scan.
            recursive: Whether to scan recursively.
            extensions: Limit to specific extensions (e.g., ['.pdf', '.txt']).

        Returns:
            List of file paths.
        """
        directory = Path(directory)

        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        # Determine which extensions to look for
        target_extensions = extensions or list(self.LOADER_MAP.keys())

        files = []
        for ext in target_extensions:
            pattern = f"**/*{ext}" if recursive else f"*{ext}"
            files.extend(directory.glob(pattern))

        # Sort for consistent ordering
        return sorted(files)

    def _load_state(self) -> set[str]:
        """Load previously processed files from state file.

        Returns:
            Set of already processed file paths.
        """
        if not self.save_state or not self.state_file.exists():
            return set()

        try:
            with open(self.state_file) as f:
                state = json.load(f)
                return set(state.get("processed_files", []))
        except Exception as e:
            logger.warning(f"Failed to load state file: {e}")
            return set()

    def _save_state(self, processed_files: set[str]) -> None:
        """Save processed files to state file.

        Args:
            processed_files: Set of processed file paths.
        """
        if not self.save_state:
            return

        try:
            with open(self.state_file, "w") as f:
                json.dump({
                    "processed_files": list(processed_files),
                    "last_updated": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save state file: {e}")

    def process_file(self, file_path: Path) -> tuple[list[DocumentChunk], ProcessingResult]:
        """Process a single file.

        Args:
            file_path: Path to the file.

        Returns:
            Tuple of (chunks, processing result).
        """
        import time

        file_path = Path(file_path)
        start_time = time.time()

        try:
            loader = self._get_loader(file_path.suffix)
            if loader is None:
                return [], ProcessingResult(
                    file_path=file_path,
                    success=False,
                    error=f"Unsupported file type: {file_path.suffix}",
                )

            chunks = loader.load(file_path)
            elapsed = time.time() - start_time

            return chunks, ProcessingResult(
                file_path=file_path,
                success=True,
                chunks_created=len(chunks),
                processing_time=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed to process {file_path}: {e}")

            return [], ProcessingResult(
                file_path=file_path,
                success=False,
                error=str(e),
                processing_time=elapsed,
            )

    def process_directory(
        self,
        directory: Path,
        recursive: bool = True,
        extensions: list[str] | None = None,
        callback: Callable[[list[DocumentChunk]], None] | None = None,
    ) -> BatchResult:
        """Process all documents in a directory.

        Args:
            directory: Directory to process.
            recursive: Whether to process recursively.
            extensions: Limit to specific extensions.
            callback: Optional callback called with chunks from each file.

        Returns:
            BatchResult with processing statistics.
        """
        import time

        start_time = time.time()

        # Discover files
        files = self._discover_files(directory, recursive, extensions)

        if not files:
            console.print("[yellow]No processable files found[/yellow]")
            return BatchResult()

        # Load previous state if resuming
        processed_files = self._load_state()
        files_to_process = [
            f for f in files if str(f.absolute()) not in processed_files
        ]

        if len(files_to_process) < len(files):
            skipped = len(files) - len(files_to_process)
            console.print(f"[blue]Skipping {skipped} already processed files[/blue]")

        # Process files
        result = BatchResult(total_files=len(files_to_process))

        if self.show_progress:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            )

            with progress:
                task = progress.add_task(
                    "Processing documents...", total=len(files_to_process)
                )

                for file_path in files_to_process:
                    chunks, proc_result = self.process_file(file_path)
                    result.results.append(proc_result)

                    if proc_result.success:
                        result.successful_files += 1
                        result.total_chunks += proc_result.chunks_created

                        if callback:
                            callback(chunks)

                        # Update state
                        processed_files.add(str(file_path.absolute()))
                        if self.save_state:
                            self._save_state(processed_files)
                    else:
                        result.failed_files += 1
                        result.errors.append((file_path, proc_result.error))

                        if not self.continue_on_error:
                            break

                    progress.update(task, advance=1)
        else:
            for file_path in files_to_process:
                chunks, proc_result = self.process_file(file_path)
                result.results.append(proc_result)

                if proc_result.success:
                    result.successful_files += 1
                    result.total_chunks += proc_result.chunks_created

                    if callback:
                        callback(chunks)

                    processed_files.add(str(file_path.absolute()))
                    if self.save_state:
                        self._save_state(processed_files)
                else:
                    result.failed_files += 1
                    result.errors.append((file_path, proc_result.error))

                    if not self.continue_on_error:
                        break

        result.total_time = time.time() - start_time
        return result

    def process_files(
        self,
        file_paths: list[Path],
        callback: Callable[[list[DocumentChunk]], None] | None = None,
    ) -> BatchResult:
        """Process a list of files.

        Args:
            file_paths: List of file paths to process.
            callback: Optional callback called with chunks from each file.

        Returns:
            BatchResult with processing statistics.
        """
        import time

        start_time = time.time()
        result = BatchResult(total_files=len(file_paths))

        iterator = (
            tqdm(file_paths, desc="Processing files")
            if self.show_progress
            else file_paths
        )

        for file_path in iterator:
            chunks, proc_result = self.process_file(Path(file_path))
            result.results.append(proc_result)

            if proc_result.success:
                result.successful_files += 1
                result.total_chunks += proc_result.chunks_created

                if callback:
                    callback(chunks)
            else:
                result.failed_files += 1
                result.errors.append((Path(file_path), proc_result.error))

                if not self.continue_on_error:
                    break

        result.total_time = time.time() - start_time
        return result

    def iter_chunks(
        self,
        directory: Path,
        recursive: bool = True,
        extensions: list[str] | None = None,
    ) -> Iterator[DocumentChunk]:
        """Iterate over chunks from all documents in a directory.

        This is memory-efficient as it yields chunks one at a time.

        Args:
            directory: Directory to process.
            recursive: Whether to process recursively.
            extensions: Limit to specific extensions.

        Yields:
            DocumentChunk objects.
        """
        files = self._discover_files(directory, recursive, extensions)

        for file_path in files:
            try:
                chunks, _ = self.process_file(file_path)
                yield from chunks
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                if not self.continue_on_error:
                    raise

    def clear_state(self) -> None:
        """Clear the saved processing state."""
        if self.state_file.exists():
            self.state_file.unlink()
            console.print("[green]State file cleared[/green]")
