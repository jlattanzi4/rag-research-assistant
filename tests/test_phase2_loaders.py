"""Tests for Phase 2 document loaders and processing."""

import tempfile
from pathlib import Path

import pytest

from rag_assistant.ingestion import (
    DocumentChunk,
    HTMLLoader,
    MarkdownLoader,
    PDFLoader,
    BatchProcessor,
    MetadataExtractor,
    RecursiveChunker,
    SentenceChunker,
    ParagraphChunker,
    get_chunker,
)


class TestMarkdownLoader:
    """Tests for MarkdownLoader class."""

    @pytest.fixture
    def loader(self):
        """Create a MarkdownLoader instance."""
        return MarkdownLoader(chunk_size=200, chunk_overlap=20)

    @pytest.fixture
    def sample_markdown(self):
        """Create a temporary markdown file."""
        content = """---
title: Test Document
author: Test Author
tags: python, testing
---

# Main Title

This is an introduction paragraph.

## Section 1

Content of section 1 goes here.

### Subsection 1.1

More detailed content.

```python
def hello():
    print("Hello, World!")
```

## Section 2

Another section with more content.
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(content)
            return Path(f.name)

    def test_load_markdown(self, loader, sample_markdown):
        """Test loading a markdown file."""
        chunks = loader.load(sample_markdown)

        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_extracts_title(self, loader, sample_markdown):
        """Test that title is extracted from H1 or frontmatter."""
        chunks = loader.load(sample_markdown)

        # Title should be in metadata
        assert any(
            c.metadata.get("title") or c.metadata.get("frontmatter_title")
            for c in chunks
        )

    def test_wrong_extension_raises(self, loader):
        """Test that wrong extension raises error."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            with pytest.raises(ValueError):
                loader.load(Path(f.name))


class TestHTMLLoader:
    """Tests for HTMLLoader class."""

    @pytest.fixture
    def loader(self):
        """Create an HTMLLoader instance."""
        return HTMLLoader(chunk_size=200, chunk_overlap=20)

    @pytest.fixture
    def sample_html(self):
        """Create a temporary HTML file."""
        content = """<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <meta name="description" content="A test page for testing">
    <meta name="author" content="Test Author">
</head>
<body>
    <nav>Navigation content to remove</nav>
    <header>Header content to remove</header>
    <main>
        <h1>Main Content</h1>
        <p>This is the main content of the page.</p>
        <p>It has multiple paragraphs.</p>
    </main>
    <footer>Footer content to remove</footer>
    <script>console.log('should be removed');</script>
</body>
</html>
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False
        ) as f:
            f.write(content)
            return Path(f.name)

    def test_load_html(self, loader, sample_html):
        """Test loading an HTML file."""
        chunks = loader.load(sample_html)

        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_removes_scripts(self, loader, sample_html):
        """Test that script content is removed."""
        chunks = loader.load(sample_html)

        content = " ".join(c.content for c in chunks)
        assert "console.log" not in content

    def test_extracts_title(self, loader, sample_html):
        """Test that title is extracted from meta tags."""
        chunks = loader.load(sample_html)

        assert chunks[0].metadata.get("title") == "Test Page"


class TestChunkers:
    """Tests for various chunking strategies."""

    @pytest.fixture
    def sample_text(self):
        """Sample text for chunking tests."""
        return """
This is the first paragraph. It contains multiple sentences. Each sentence should be handled properly.

This is the second paragraph. It also has several sentences. We want to test paragraph boundaries.

This is the third paragraph. More content here. Testing the chunking behavior.
""".strip()

    def test_recursive_chunker(self, sample_text):
        """Test recursive character chunking."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split(sample_text)

        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)

    def test_sentence_chunker(self, sample_text):
        """Test sentence-based chunking."""
        chunker = SentenceChunker(chunk_size=200, chunk_overlap=1)
        chunks = chunker.split(sample_text)

        assert len(chunks) > 0
        # Each chunk should contain complete sentences
        for chunk in chunks:
            # Should end with sentence-ending punctuation or be last chunk
            assert chunk.rstrip().endswith((".", "!", "?")) or chunk == chunks[-1]

    def test_paragraph_chunker(self, sample_text):
        """Test paragraph-based chunking."""
        chunker = ParagraphChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.split(sample_text)

        assert len(chunks) > 0

    def test_get_chunker_factory(self):
        """Test the chunker factory function."""
        chunker = get_chunker("recursive", chunk_size=100)
        assert isinstance(chunker, RecursiveChunker)

        chunker = get_chunker("sentence", chunk_size=200)
        assert isinstance(chunker, SentenceChunker)

        chunker = get_chunker("paragraph", chunk_size=500)
        assert isinstance(chunker, ParagraphChunker)

    def test_get_chunker_invalid(self):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError):
            get_chunker("invalid_strategy")


class TestMetadataExtractor:
    """Tests for MetadataExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create a MetadataExtractor instance."""
        return MetadataExtractor()

    def test_extract_file_metadata(self, extractor):
        """Test basic file metadata extraction."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            f.flush()

            metadata = extractor.extract_file_metadata(Path(f.name))

            assert "source" in metadata
            assert "filename" in metadata
            assert "file_type" in metadata
            assert metadata["file_type"] == ".txt"

    def test_extract_from_markdown(self, extractor):
        """Test metadata extraction from markdown."""
        content = """---
title: My Document
author: John Doe
---

# Heading

Content here.
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(content)
            f.flush()

            metadata = extractor.extract_from_markdown(content, Path(f.name))

            assert metadata.title == "My Document"
            assert metadata.author == "John Doe"


class TestBatchProcessor:
    """Tests for BatchProcessor class."""

    @pytest.fixture
    def temp_dir_with_files(self):
        """Create a temporary directory with various files."""
        import shutil

        temp_dir = tempfile.mkdtemp()

        # Create text file
        (Path(temp_dir) / "doc1.txt").write_text("This is a text document.")

        # Create markdown file
        (Path(temp_dir) / "doc2.md").write_text("# Markdown\n\nContent here.")

        # Create HTML file
        (Path(temp_dir) / "doc3.html").write_text(
            "<html><body><p>HTML content</p></body></html>"
        )

        yield Path(temp_dir)

        shutil.rmtree(temp_dir)

    def test_discover_files(self, temp_dir_with_files):
        """Test file discovery."""
        processor = BatchProcessor(show_progress=False)

        files = processor._discover_files(temp_dir_with_files)

        # Should find all 3 files
        assert len(files) == 3

    def test_discover_files_filtered(self, temp_dir_with_files):
        """Test file discovery with extension filter."""
        processor = BatchProcessor(show_progress=False)

        files = processor._discover_files(
            temp_dir_with_files, extensions=[".txt"]
        )

        assert len(files) == 1
        assert files[0].suffix == ".txt"

    def test_process_file(self, temp_dir_with_files):
        """Test processing a single file."""
        processor = BatchProcessor(show_progress=False)
        txt_file = temp_dir_with_files / "doc1.txt"

        chunks, result = processor.process_file(txt_file)

        assert result.success
        assert len(chunks) > 0

    def test_process_directory(self, temp_dir_with_files):
        """Test processing a directory."""
        processor = BatchProcessor(show_progress=False)

        result = processor.process_directory(temp_dir_with_files)

        assert result.total_files == 3
        assert result.successful_files == 3
        assert result.failed_files == 0
        assert result.total_chunks > 0

    def test_continue_on_error(self, temp_dir_with_files):
        """Test that processing continues on error when configured."""
        processor = BatchProcessor(
            show_progress=False, continue_on_error=True
        )

        # Create an invalid file
        (temp_dir_with_files / "invalid.txt").write_text("")  # Empty file

        result = processor.process_directory(temp_dir_with_files)

        # Should still process valid files
        assert result.successful_files >= 2
