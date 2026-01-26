"""Document loaders for ingesting various file types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_assistant.config import settings


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""

    content: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            # Generate a unique chunk ID based on source and content hash
            source = self.metadata.get("source", "unknown")
            content_hash = hash(self.content) % 10**8
            chunk_index = self.metadata.get("chunk_index", 0)
            self.chunk_id = f"{Path(source).stem}_{chunk_index}_{content_hash}"


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, file_path: Path) -> list[DocumentChunk]:
        """Load a document and return chunks."""
        pass

    @abstractmethod
    def load_directory(self, directory: Path) -> list[DocumentChunk]:
        """Load all documents from a directory."""
        pass


class TextLoader(BaseLoader):
    """Loader for plain text files (.txt)."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        encoding: str = "utf-8",
    ) -> None:
        """Initialize the TextLoader.

        Args:
            chunk_size: Size of each chunk in characters. Defaults to settings.
            chunk_overlap: Overlap between chunks in characters. Defaults to settings.
            encoding: File encoding. Defaults to utf-8.
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.encoding = encoding

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _extract_metadata(self, file_path: Path) -> dict:
        """Extract metadata from a file.

        Args:
            file_path: Path to the file.

        Returns:
            Dictionary of metadata.
        """
        stat = file_path.stat()
        return {
            "source": str(file_path.absolute()),
            "filename": file_path.name,
            "file_type": file_path.suffix,
            "file_size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

    def load(self, file_path: Path) -> list[DocumentChunk]:
        """Load a text file and split into chunks.

        Args:
            file_path: Path to the text file.

        Returns:
            List of DocumentChunk objects.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file is not a text file.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in [".txt", ".text"]:
            raise ValueError(f"Expected .txt file, got: {file_path.suffix}")

        # Read the file content
        content = file_path.read_text(encoding=self.encoding)

        # Skip empty files
        if not content.strip():
            return []

        # Extract base metadata
        base_metadata = self._extract_metadata(file_path)

        # Split into chunks
        texts = self.text_splitter.split_text(content)

        # Create DocumentChunk objects
        chunks = []
        for i, text in enumerate(texts):
            chunk_metadata = {
                **base_metadata,
                "chunk_index": i,
                "total_chunks": len(texts),
            }
            chunks.append(DocumentChunk(content=text, metadata=chunk_metadata))

        return chunks

    def load_directory(
        self, directory: Path, recursive: bool = True
    ) -> list[DocumentChunk]:
        """Load all text files from a directory.

        Args:
            directory: Path to the directory.
            recursive: Whether to search recursively. Defaults to True.

        Returns:
            List of DocumentChunk objects from all files.

        Raises:
            NotADirectoryError: If the path is not a directory.
        """
        directory = Path(directory)

        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        # Find all text files
        pattern = "**/*.txt" if recursive else "*.txt"
        files = list(directory.glob(pattern))

        # Load all files
        all_chunks = []
        for file_path in files:
            try:
                chunks = self.load(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")

        return all_chunks

    def load_many(self, file_paths: list[Path]) -> Iterator[DocumentChunk]:
        """Load multiple files and yield chunks.

        Args:
            file_paths: List of file paths to load.

        Yields:
            DocumentChunk objects from each file.
        """
        for file_path in file_paths:
            try:
                chunks = self.load(file_path)
                yield from chunks
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")

    def chunk_text(self, content: str, metadata: dict | None = None) -> list[DocumentChunk]:
        """Chunk raw text content directly.

        Args:
            content: Text content to chunk.
            metadata: Optional metadata to include.

        Returns:
            List of DocumentChunk objects.
        """
        if not content.strip():
            return []

        metadata = metadata or {}

        # Split into chunks
        texts = self.text_splitter.split_text(content)

        # Create DocumentChunk objects
        chunks = []
        for i, text in enumerate(texts):
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(texts),
            }
            chunks.append(DocumentChunk(content=text, metadata=chunk_metadata))

        return chunks


class PDFLoader(BaseLoader):
    """Loader for PDF files using PyPDF2 with pdfplumber fallback."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        use_pdfplumber: bool = False,
    ) -> None:
        """Initialize the PDFLoader.

        Args:
            chunk_size: Size of each chunk in characters. Defaults to settings.
            chunk_overlap: Overlap between chunks in characters. Defaults to settings.
            use_pdfplumber: Force use of pdfplumber instead of PyPDF2.
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.use_pdfplumber = use_pdfplumber

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _extract_with_pypdf2(self, file_path: Path) -> tuple[str, dict]:
        """Extract text and metadata using PyPDF2.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Tuple of (extracted text, metadata dict).
        """
        import PyPDF2

        text_parts = []
        metadata = {}

        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)

            # Extract metadata
            if reader.metadata:
                metadata = {
                    "title": reader.metadata.get("/Title", ""),
                    "author": reader.metadata.get("/Author", ""),
                    "subject": reader.metadata.get("/Subject", ""),
                    "creator": reader.metadata.get("/Creator", ""),
                }

            metadata["page_count"] = len(reader.pages)

            # Extract text from each page
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}")

        return "\n\n".join(text_parts), metadata

    def _extract_with_pdfplumber(self, file_path: Path) -> tuple[str, dict]:
        """Extract text and metadata using pdfplumber (better for complex PDFs).

        Args:
            file_path: Path to the PDF file.

        Returns:
            Tuple of (extracted text, metadata dict).
        """
        import pdfplumber

        text_parts = []
        metadata = {}

        with pdfplumber.open(file_path) as pdf:
            metadata["page_count"] = len(pdf.pages)

            if pdf.metadata:
                metadata.update({
                    "title": pdf.metadata.get("Title", ""),
                    "author": pdf.metadata.get("Author", ""),
                    "subject": pdf.metadata.get("Subject", ""),
                    "creator": pdf.metadata.get("Creator", ""),
                })

            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}")

        return "\n\n".join(text_parts), metadata

    def _extract_metadata(self, file_path: Path) -> dict:
        """Extract file metadata."""
        stat = file_path.stat()
        return {
            "source": str(file_path.absolute()),
            "filename": file_path.name,
            "file_type": ".pdf",
            "file_size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

    def load(self, file_path: Path) -> list[DocumentChunk]:
        """Load a PDF file and split into chunks.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of DocumentChunk objects.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file is not a PDF.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected .pdf file, got: {file_path.suffix}")

        # Try extraction
        try:
            if self.use_pdfplumber:
                content, pdf_metadata = self._extract_with_pdfplumber(file_path)
            else:
                content, pdf_metadata = self._extract_with_pypdf2(file_path)
                # Fallback to pdfplumber if PyPDF2 extracts little content
                if len(content.strip()) < 100:
                    content, pdf_metadata = self._extract_with_pdfplumber(file_path)
        except Exception as e:
            # Try the other method as fallback
            try:
                if self.use_pdfplumber:
                    content, pdf_metadata = self._extract_with_pypdf2(file_path)
                else:
                    content, pdf_metadata = self._extract_with_pdfplumber(file_path)
            except Exception:
                raise ValueError(f"Failed to extract text from PDF: {e}")

        if not content.strip():
            return []

        # Combine metadata
        base_metadata = self._extract_metadata(file_path)
        base_metadata.update({k: v for k, v in pdf_metadata.items() if v})

        # Split into chunks
        texts = self.text_splitter.split_text(content)

        # Create DocumentChunk objects
        chunks = []
        for i, text in enumerate(texts):
            chunk_metadata = {
                **base_metadata,
                "chunk_index": i,
                "total_chunks": len(texts),
            }
            chunks.append(DocumentChunk(content=text, metadata=chunk_metadata))

        return chunks

    def load_directory(
        self, directory: Path, recursive: bool = True
    ) -> list[DocumentChunk]:
        """Load all PDF files from a directory."""
        directory = Path(directory)

        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        pattern = "**/*.pdf" if recursive else "*.pdf"
        files = list(directory.glob(pattern))

        all_chunks = []
        for file_path in files:
            try:
                chunks = self.load(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")

        return all_chunks


class MarkdownLoader(BaseLoader):
    """Loader for Markdown files."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        encoding: str = "utf-8",
        preserve_code_blocks: bool = True,
    ) -> None:
        """Initialize the MarkdownLoader.

        Args:
            chunk_size: Size of each chunk in characters. Defaults to settings.
            chunk_overlap: Overlap between chunks in characters. Defaults to settings.
            encoding: File encoding. Defaults to utf-8.
            preserve_code_blocks: Try to keep code blocks intact. Defaults to True.
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.encoding = encoding
        self.preserve_code_blocks = preserve_code_blocks

        # Use markdown-aware separators
        separators = [
            "\n## ",      # H2 headers
            "\n### ",     # H3 headers
            "\n#### ",    # H4 headers
            "\n```\n",    # Code blocks
            "\n\n",       # Paragraphs
            "\n",         # Lines
            ". ",         # Sentences
            " ",          # Words
            "",           # Characters
        ]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=separators,
        )

    def _extract_metadata(self, file_path: Path, content: str) -> dict:
        """Extract metadata from markdown file."""
        import re

        stat = file_path.stat()
        metadata = {
            "source": str(file_path.absolute()),
            "filename": file_path.name,
            "file_type": ".md",
            "file_size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

        # Try to extract title from first H1
        h1_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if h1_match:
            metadata["title"] = h1_match.group(1).strip()

        # Extract YAML frontmatter if present
        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            for line in frontmatter.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[f"frontmatter_{key.strip()}"] = value.strip()

        return metadata

    def load(self, file_path: Path) -> list[DocumentChunk]:
        """Load a Markdown file and split into chunks.

        Args:
            file_path: Path to the Markdown file.

        Returns:
            List of DocumentChunk objects.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in [".md", ".markdown"]:
            raise ValueError(f"Expected .md file, got: {file_path.suffix}")

        content = file_path.read_text(encoding=self.encoding)

        if not content.strip():
            return []

        base_metadata = self._extract_metadata(file_path, content)

        # Split into chunks
        texts = self.text_splitter.split_text(content)

        chunks = []
        for i, text in enumerate(texts):
            chunk_metadata = {
                **base_metadata,
                "chunk_index": i,
                "total_chunks": len(texts),
            }
            chunks.append(DocumentChunk(content=text, metadata=chunk_metadata))

        return chunks

    def load_directory(
        self, directory: Path, recursive: bool = True
    ) -> list[DocumentChunk]:
        """Load all Markdown files from a directory."""
        directory = Path(directory)

        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        all_chunks = []
        for pattern in ["**/*.md", "**/*.markdown"] if recursive else ["*.md", "*.markdown"]:
            for file_path in directory.glob(pattern):
                try:
                    chunks = self.load(file_path)
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")

        return all_chunks


class HTMLLoader(BaseLoader):
    """Loader for HTML files using BeautifulSoup."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        encoding: str = "utf-8",
        remove_scripts: bool = True,
        remove_styles: bool = True,
    ) -> None:
        """Initialize the HTMLLoader.

        Args:
            chunk_size: Size of each chunk in characters. Defaults to settings.
            chunk_overlap: Overlap between chunks in characters. Defaults to settings.
            encoding: File encoding. Defaults to utf-8.
            remove_scripts: Remove script tags. Defaults to True.
            remove_styles: Remove style tags. Defaults to True.
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.encoding = encoding
        self.remove_scripts = remove_scripts
        self.remove_styles = remove_styles

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _extract_text(self, file_path: Path) -> tuple[str, dict]:
        """Extract text and metadata from HTML file.

        Args:
            file_path: Path to the HTML file.

        Returns:
            Tuple of (extracted text, metadata dict).
        """
        from bs4 import BeautifulSoup

        content = file_path.read_text(encoding=self.encoding)
        soup = BeautifulSoup(content, "html.parser")

        metadata = {}

        # Extract title
        if soup.title:
            metadata["title"] = soup.title.string

        # Extract meta tags
        for meta in soup.find_all("meta"):
            name = meta.get("name", meta.get("property", ""))
            content_val = meta.get("content", "")
            if name and content_val:
                metadata[f"meta_{name}"] = content_val

        # Remove unwanted elements
        if self.remove_scripts:
            for script in soup.find_all("script"):
                script.decompose()

        if self.remove_styles:
            for style in soup.find_all("style"):
                style.decompose()

        # Remove navigation, header, footer elements
        for tag in soup.find_all(["nav", "header", "footer", "aside"]):
            tag.decompose()

        # Get text content
        text = soup.get_text(separator="\n", strip=True)

        # Clean up multiple newlines
        import re
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text, metadata

    def _extract_metadata(self, file_path: Path) -> dict:
        """Extract file metadata."""
        stat = file_path.stat()
        return {
            "source": str(file_path.absolute()),
            "filename": file_path.name,
            "file_type": file_path.suffix,
            "file_size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

    def load(self, file_path: Path) -> list[DocumentChunk]:
        """Load an HTML file and split into chunks.

        Args:
            file_path: Path to the HTML file.

        Returns:
            List of DocumentChunk objects.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in [".html", ".htm"]:
            raise ValueError(f"Expected .html file, got: {file_path.suffix}")

        content, html_metadata = self._extract_text(file_path)

        if not content.strip():
            return []

        # Combine metadata
        base_metadata = self._extract_metadata(file_path)
        base_metadata.update({k: v for k, v in html_metadata.items() if v})

        # Split into chunks
        texts = self.text_splitter.split_text(content)

        chunks = []
        for i, text in enumerate(texts):
            chunk_metadata = {
                **base_metadata,
                "chunk_index": i,
                "total_chunks": len(texts),
            }
            chunks.append(DocumentChunk(content=text, metadata=chunk_metadata))

        return chunks

    def load_directory(
        self, directory: Path, recursive: bool = True
    ) -> list[DocumentChunk]:
        """Load all HTML files from a directory."""
        directory = Path(directory)

        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        all_chunks = []
        for pattern in ["**/*.html", "**/*.htm"] if recursive else ["*.html", "*.htm"]:
            for file_path in directory.glob(pattern):
                try:
                    chunks = self.load(file_path)
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")

        return all_chunks
