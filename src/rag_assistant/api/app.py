"""FastAPI application for the RAG Research Assistant."""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag_assistant.api.models import ErrorResponse
from rag_assistant.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rag_assistant.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    logger.info("Starting RAG Research Assistant API")
    yield
    # Shutdown
    logger.info("Shutting down RAG Research Assistant API")


# Create FastAPI app
app = FastAPI(
    title="RAG Research Assistant API",
    description="""
## RAG Research Assistant

A Retrieval-Augmented Generation (RAG) API for intelligent document Q&A.

### Features

- **Semantic Search**: Find relevant documents using vector similarity
- **AI-Powered Q&A**: Get accurate answers with source citations
- **Document Ingestion**: Index PDF, Markdown, HTML, and text files
- **Quality Assessment**: Automatic response quality evaluation

### Getting Started

1. **Ingest Documents**: Use `/ingest/file` or `/ingest/text` to add documents
2. **Search**: Use `/search` for semantic document search
3. **Ask Questions**: Use `/query` to get AI-generated answers

### Authentication

Currently no authentication required. Add your own auth middleware for production.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and their processing time."""
    start_time = time.time()
    request_id = f"{datetime.now(timezone.utc).timestamp()}"

    # Log request
    logger.info(
        f"Request {request_id}: {request.method} {request.url.path} "
        f"- Client: {request.client.host if request.client else 'unknown'}"
    )

    # Process request
    response = await call_next(request)

    # Log response
    process_time = (time.time() - start_time) * 1000
    logger.info(
        f"Response {request_id}: Status {response.status_code} "
        f"- Time: {process_time:.2f}ms"
    )

    # Add timing header
    response.headers["X-Process-Time-MS"] = str(round(process_time, 2))

    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.exception(f"Unhandled exception: {exc}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            detail=str(exc) if app.debug else None,
            timestamp=datetime.now(timezone.utc),
        ).model_dump(mode="json"),
    )


# Value error handler
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="ValidationError",
            message=str(exc),
            timestamp=datetime.now(timezone.utc),
        ).model_dump(mode="json"),
    )


# Include main router
app.include_router(router)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    return app


# CLI entry point for running with uvicorn
def main():
    """Run the API server."""
    import uvicorn

    uvicorn.run(
        "rag_assistant.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
