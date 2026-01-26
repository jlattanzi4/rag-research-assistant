"""Metadata extraction utilities for documents."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class DocumentMetadata:
    """Structured metadata for a document."""

    source: str
    filename: str
    file_type: str
    file_size: int
    created_at: str
    modified_at: str

    # Optional fields
    title: str = ""
    author: str = ""
    subject: str = ""
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    language: str = ""
    page_count: int = 0

    # Custom metadata
    custom: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        result = {
            "source": self.source,
            "filename": self.filename,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
        }

        # Add optional fields if present
        if self.title:
            result["title"] = self.title
        if self.author:
            result["author"] = self.author
        if self.subject:
            result["subject"] = self.subject
        if self.description:
            result["description"] = self.description
        if self.keywords:
            result["keywords"] = ",".join(self.keywords)
        if self.language:
            result["language"] = self.language
        if self.page_count:
            result["page_count"] = self.page_count

        # Add custom metadata
        result.update(self.custom)

        return result


class MetadataExtractor:
    """Extracts and normalizes metadata from various document types."""

    def __init__(self) -> None:
        """Initialize the MetadataExtractor."""
        pass

    def extract_file_metadata(self, file_path: Path) -> dict:
        """Extract basic file system metadata.

        Args:
            file_path: Path to the file.

        Returns:
            Dictionary of file metadata.
        """
        file_path = Path(file_path)
        stat = file_path.stat()

        return {
            "source": str(file_path.absolute()),
            "filename": file_path.name,
            "file_type": file_path.suffix.lower(),
            "file_size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

    def extract_from_text(self, content: str, file_path: Path) -> DocumentMetadata:
        """Extract metadata from plain text content.

        Args:
            content: Text content.
            file_path: Path to the source file.

        Returns:
            DocumentMetadata object.
        """
        base = self.extract_file_metadata(file_path)

        # Try to extract title from first line
        lines = content.strip().split("\n")
        title = ""
        if lines:
            first_line = lines[0].strip()
            # If first line looks like a title (short, no punctuation at end)
            if len(first_line) < 200 and not first_line.endswith((".", ",", ";")):
                title = first_line

        return DocumentMetadata(
            source=base["source"],
            filename=base["filename"],
            file_type=base["file_type"],
            file_size=base["file_size"],
            created_at=base["created_at"],
            modified_at=base["modified_at"],
            title=title,
        )

    def extract_from_markdown(self, content: str, file_path: Path) -> DocumentMetadata:
        """Extract metadata from Markdown content.

        Args:
            content: Markdown content.
            file_path: Path to the source file.

        Returns:
            DocumentMetadata object.
        """
        base = self.extract_file_metadata(file_path)
        custom = {}

        # Extract title from first H1
        title = ""
        h1_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if h1_match:
            title = h1_match.group(1).strip()

        # Extract YAML frontmatter
        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            for line in frontmatter.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip().strip("\"'")

                    if key == "title":
                        title = value
                    elif key == "author":
                        custom["author"] = value
                    elif key == "date":
                        custom["date"] = value
                    elif key == "tags" or key == "keywords":
                        # Handle comma or space separated tags
                        tags = re.split(r"[,\s]+", value.strip("[]"))
                        custom["keywords"] = ",".join(t.strip() for t in tags if t.strip())
                    else:
                        custom[f"frontmatter_{key}"] = value

        return DocumentMetadata(
            source=base["source"],
            filename=base["filename"],
            file_type=base["file_type"],
            file_size=base["file_size"],
            created_at=base["created_at"],
            modified_at=base["modified_at"],
            title=title,
            author=custom.pop("author", ""),
            keywords=custom.pop("keywords", "").split(",") if "keywords" in custom else [],
            custom=custom,
        )

    def extract_from_html(self, content: str, file_path: Path) -> DocumentMetadata:
        """Extract metadata from HTML content.

        Args:
            content: HTML content.
            file_path: Path to the source file.

        Returns:
            DocumentMetadata object.
        """
        from bs4 import BeautifulSoup

        base = self.extract_file_metadata(file_path)
        soup = BeautifulSoup(content, "html.parser")

        # Extract title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Extract meta tags
        author = ""
        description = ""
        keywords = []

        for meta in soup.find_all("meta"):
            name = meta.get("name", meta.get("property", "")).lower()
            content_val = meta.get("content", "")

            if name == "author":
                author = content_val
            elif name == "description" or name == "og:description":
                description = content_val
            elif name == "keywords":
                keywords = [k.strip() for k in content_val.split(",")]
            elif name == "og:title" and not title:
                title = content_val

        return DocumentMetadata(
            source=base["source"],
            filename=base["filename"],
            file_type=base["file_type"],
            file_size=base["file_size"],
            created_at=base["created_at"],
            modified_at=base["modified_at"],
            title=title,
            author=author,
            description=description,
            keywords=keywords,
        )

    def extract_from_pdf(
        self, file_path: Path, pdf_metadata: dict | None = None
    ) -> DocumentMetadata:
        """Extract metadata from PDF file.

        Args:
            file_path: Path to the PDF file.
            pdf_metadata: Optional pre-extracted PDF metadata dict.

        Returns:
            DocumentMetadata object.
        """
        base = self.extract_file_metadata(file_path)

        # If PDF metadata not provided, try to extract it
        if pdf_metadata is None:
            pdf_metadata = self._extract_pdf_metadata(file_path)

        return DocumentMetadata(
            source=base["source"],
            filename=base["filename"],
            file_type=base["file_type"],
            file_size=base["file_size"],
            created_at=base["created_at"],
            modified_at=base["modified_at"],
            title=pdf_metadata.get("title", ""),
            author=pdf_metadata.get("author", ""),
            subject=pdf_metadata.get("subject", ""),
            page_count=pdf_metadata.get("page_count", 0),
            custom={k: v for k, v in pdf_metadata.items()
                   if k not in ["title", "author", "subject", "page_count"] and v},
        )

    def _extract_pdf_metadata(self, file_path: Path) -> dict:
        """Extract metadata directly from PDF file.

        Args:
            file_path: Path to PDF file.

        Returns:
            Dictionary of PDF metadata.
        """
        try:
            import PyPDF2

            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                metadata = {}

                if reader.metadata:
                    metadata = {
                        "title": reader.metadata.get("/Title", ""),
                        "author": reader.metadata.get("/Author", ""),
                        "subject": reader.metadata.get("/Subject", ""),
                        "creator": reader.metadata.get("/Creator", ""),
                    }

                metadata["page_count"] = len(reader.pages)
                return metadata
        except Exception:
            return {}

    def extract(self, file_path: Path, content: str | None = None) -> DocumentMetadata:
        """Auto-detect file type and extract appropriate metadata.

        Args:
            file_path: Path to the file.
            content: Optional pre-loaded content (for text-based files).

        Returns:
            DocumentMetadata object.
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self.extract_from_pdf(file_path)

        # For text-based files, load content if not provided
        if content is None:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

        if suffix in [".md", ".markdown"]:
            return self.extract_from_markdown(content, file_path)
        elif suffix in [".html", ".htm"]:
            return self.extract_from_html(content, file_path)
        else:
            return self.extract_from_text(content, file_path)
