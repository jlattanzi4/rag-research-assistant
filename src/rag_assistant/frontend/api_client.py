"""API client for communicating with the RAG backend."""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class RAGAPIClient:
    """Client for the RAG Research Assistant API."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 60.0):
        """Initialize the API client.

        Args:
            base_url: Base URL of the API server.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: dict | None = None,
        params: dict | None = None,
    ) -> dict | None:
        """Make an HTTP request to the API.

        Args:
            method: HTTP method.
            endpoint: API endpoint.
            data: Request body data.
            params: Query parameters.

        Returns:
            Response data or None on error.
        """
        url = f"{self.base_url}{endpoint}"

        try:
            if method.upper() == "GET":
                response = self.client.get(url, params=params)
            elif method.upper() == "POST":
                response = self.client.post(url, json=data, params=params)
            elif method.upper() == "DELETE":
                response = self.client.delete(url, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()

        except httpx.ConnectError:
            logger.error(f"Failed to connect to API at {self.base_url}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"API request failed: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

    def health_check(self) -> dict | None:
        """Check API health status.

        Returns:
            Health status data or None.
        """
        return self._make_request("GET", "/health")

    def get_stats(self) -> dict | None:
        """Get index statistics.

        Returns:
            Statistics data or None.
        """
        return self._make_request("GET", "/admin/stats")

    def query(
        self,
        query: str,
        top_k: int = 5,
        prompt_style: str = "detailed",
        min_relevance: float = 0.0,
        use_reranking: bool = False,
    ) -> dict | None:
        """Submit a query to the RAG system.

        Args:
            query: The question to ask.
            top_k: Number of sources to retrieve.
            prompt_style: Response style (concise, detailed, academic, conversational).
            min_relevance: Minimum relevance score.
            use_reranking: Whether to use reranking.

        Returns:
            Query response data or None.
        """
        return self._make_request(
            "POST",
            "/query/",
            data={
                "query": query,
                "top_k": top_k,
                "prompt_style": prompt_style,
                "min_relevance": min_relevance,
                "use_reranking": use_reranking,
            },
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> dict | None:
        """Perform semantic search.

        Args:
            query: Search query.
            top_k: Number of results to return.
            min_score: Minimum similarity score.

        Returns:
            Search results data or None.
        """
        return self._make_request(
            "POST",
            "/search/",
            data={
                "query": query,
                "top_k": top_k,
                "min_score": min_score,
            },
        )

    def ingest_text(
        self,
        content: str,
        filename: str,
        metadata: dict | None = None,
    ) -> dict | None:
        """Ingest text content.

        Args:
            content: Text content to ingest.
            filename: Filename for the document.
            metadata: Additional metadata.

        Returns:
            Ingestion result or None.
        """
        return self._make_request(
            "POST",
            "/ingest/text",
            data={
                "content": content,
                "filename": filename,
                "metadata": metadata or {},
            },
        )

    def ingest_file(
        self,
        file_path: str,
        recursive: bool = True,
        file_types: list[str] | None = None,
    ) -> dict | None:
        """Ingest files from a path.

        Args:
            file_path: Path to file or directory.
            recursive: Process directories recursively.
            file_types: File extensions to process.

        Returns:
            Ingestion result or None.
        """
        return self._make_request(
            "POST",
            "/ingest/file",
            data={
                "file_path": file_path,
                "recursive": recursive,
                "file_types": file_types or ["txt", "pdf", "md", "html"],
            },
        )

    def clear_index(self, confirm: bool = False) -> dict | None:
        """Clear the document index.

        Args:
            confirm: Must be True to actually clear.

        Returns:
            Result or None.
        """
        return self._make_request("DELETE", "/admin/clear", params={"confirm": confirm})

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
