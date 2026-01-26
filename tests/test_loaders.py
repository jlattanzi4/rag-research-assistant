"""Tests for document loaders."""

from pathlib import Path
import tempfile

import pytest

from rag_assistant.ingestion.loaders import TextLoader, DocumentChunk


class TestDocumentChunk:
    """Tests for DocumentChunk dataclass."""

    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunk = DocumentChunk(
            content="Test content",
            metadata={"source": "test.txt", "chunk_index": 0},
        )
        assert chunk.content == "Test content"
        assert chunk.metadata["source"] == "test.txt"
        assert chunk.chunk_id  # Should be auto-generated

    def test_chunk_id_generation(self):
        """Test that chunk IDs are generated correctly."""
        chunk = DocumentChunk(
            content="Test content",
            metadata={"source": "/path/to/test.txt", "chunk_index": 0},
        )
        assert "test_" in chunk.chunk_id
        assert "_0_" in chunk.chunk_id


class TestTextLoader:
    """Tests for TextLoader class."""

    @pytest.fixture
    def text_loader(self):
        """Create a TextLoader instance."""
        return TextLoader(chunk_size=100, chunk_overlap=10)

    @pytest.fixture
    def sample_file(self):
        """Create a temporary sample file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("This is a test document.\n\n")
            f.write("It contains multiple paragraphs.\n\n")
            f.write("Each paragraph will be processed.")
            return Path(f.name)

    @pytest.fixture
    def sample_dir(self, sample_file):
        """Create a temporary directory with sample files."""
        return sample_file.parent

    def test_load_single_file(self, text_loader, sample_file):
        """Test loading a single text file."""
        chunks = text_loader.load(sample_file)

        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(c.metadata.get("source") for c in chunks)

    def test_load_extracts_metadata(self, text_loader, sample_file):
        """Test that metadata is properly extracted."""
        chunks = text_loader.load(sample_file)

        assert len(chunks) > 0
        metadata = chunks[0].metadata

        assert "source" in metadata
        assert "filename" in metadata
        assert "file_type" in metadata
        assert "file_size" in metadata
        assert "created_at" in metadata
        assert "modified_at" in metadata

    def test_load_nonexistent_file(self, text_loader):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            text_loader.load(Path("/nonexistent/file.txt"))

    def test_load_wrong_extension(self, text_loader):
        """Test that loading non-txt file raises error."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            with pytest.raises(ValueError):
                text_loader.load(Path(f.name))

    def test_load_empty_file(self, text_loader):
        """Test that empty files return empty list."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("")
            chunks = text_loader.load(Path(f.name))
            assert chunks == []

    def test_chunk_indices_sequential(self, text_loader, sample_file):
        """Test that chunk indices are sequential."""
        chunks = text_loader.load(sample_file)

        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_total_chunks_metadata(self, text_loader, sample_file):
        """Test that total_chunks metadata is correct."""
        chunks = text_loader.load(sample_file)

        total = chunks[0].metadata["total_chunks"]
        assert all(c.metadata["total_chunks"] == total for c in chunks)
        assert total == len(chunks)


class TestTextLoaderDirectory:
    """Tests for TextLoader directory loading."""

    @pytest.fixture
    def temp_dir_with_files(self):
        """Create a temporary directory with multiple files."""
        import tempfile
        import os

        temp_dir = tempfile.mkdtemp()
        files = []

        # Create some text files
        for i in range(3):
            file_path = Path(temp_dir) / f"doc_{i}.txt"
            file_path.write_text(f"Content of document {i}")
            files.append(file_path)

        # Create a subdirectory with a file
        subdir = Path(temp_dir) / "subdir"
        subdir.mkdir()
        sub_file = subdir / "nested.txt"
        sub_file.write_text("Nested document content")
        files.append(sub_file)

        yield Path(temp_dir), files

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    def test_load_directory(self, temp_dir_with_files):
        """Test loading all files from a directory."""
        temp_dir, files = temp_dir_with_files
        loader = TextLoader()

        chunks = loader.load_directory(temp_dir, recursive=True)

        # Should load from all 4 files
        sources = set(c.metadata["filename"] for c in chunks)
        assert len(sources) == 4

    def test_load_directory_non_recursive(self, temp_dir_with_files):
        """Test non-recursive directory loading."""
        temp_dir, files = temp_dir_with_files
        loader = TextLoader()

        chunks = loader.load_directory(temp_dir, recursive=False)

        # Should only load from top-level files (3)
        sources = set(c.metadata["filename"] for c in chunks)
        assert len(sources) == 3

    def test_load_directory_not_found(self):
        """Test that loading nonexistent directory raises error."""
        loader = TextLoader()
        with pytest.raises(NotADirectoryError):
            loader.load_directory(Path("/nonexistent/directory"))
