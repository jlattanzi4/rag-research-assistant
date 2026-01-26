"""API routes for the RAG Research Assistant."""

import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from rag_assistant.api.models import (
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    IngestTextRequest,
    QualityInfo,
    QualityLevel,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SourceDocument,
    StatsResponse,
)
from rag_assistant.config import settings
from rag_assistant.generation import ContextBuilder, EdgeCaseHandler, ResponseQualityChecker
from rag_assistant.generation.prompts import PromptStyle

# API Version
API_VERSION = "1.0.0"

# Create routers
router = APIRouter()
query_router = APIRouter(prefix="/query", tags=["Query"])
search_router = APIRouter(prefix="/search", tags=["Search"])
ingest_router = APIRouter(prefix="/ingest", tags=["Ingestion"])
admin_router = APIRouter(prefix="/admin", tags=["Administration"])


# Dependency to get pipeline (lazy initialization)
_pipeline = None


def get_pipeline():
    """Get or create the RAG pipeline instance."""
    global _pipeline
    if _pipeline is None:
        from rag_assistant import create_pipeline

        _pipeline = create_pipeline()
    return _pipeline


# Health and Info Endpoints


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the API and its components.",
)
async def health_check():
    """Check API health status."""
    components = {
        "api": True,
        "vector_store": False,
        "llm_service": False,
    }

    try:
        pipeline = get_pipeline()
        components["vector_store"] = pipeline.vector_store is not None
        components["llm_service"] = pipeline.llm_service is not None
    except Exception:
        pass

    overall_status = "healthy" if all(components.values()) else "degraded"

    return HealthResponse(
        status=overall_status,
        version=API_VERSION,
        timestamp=datetime.now(timezone.utc),
        components=components,
    )


@router.get(
    "/",
    summary="API Info",
    description="Get basic API information.",
)
async def api_info():
    """Get API information."""
    return {
        "name": "RAG Research Assistant API",
        "version": API_VERSION,
        "description": "A Retrieval-Augmented Generation API for document Q&A",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
    }


# Query Endpoints


@query_router.post(
    "/",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Ask a Question",
    description="Submit a question and receive an AI-generated answer based on indexed documents.",
)
async def query(request: QueryRequest, pipeline=Depends(get_pipeline)):
    """Process a RAG query and return a generated response."""
    start_time = time.time()

    try:
        # Search for relevant documents
        search_results = pipeline.vector_store.search(
            query=request.query,
            top_k=request.top_k,
        )

        # Filter by minimum relevance
        filtered_results = [r for r in search_results if r.score >= request.min_relevance]

        # Handle edge cases
        edge_handler = EdgeCaseHandler(min_relevance_threshold=request.min_relevance)
        should_proceed, edge_response = edge_handler.process(request.query, filtered_results)

        if not should_proceed and edge_response:
            # Return edge case response
            processing_time = (time.time() - start_time) * 1000
            return QueryResponse(
                answer=edge_response,
                sources=[],
                quality=QualityInfo(
                    level=QualityLevel.INSUFFICIENT,
                    confidence=0.0,
                    has_citations=False,
                    citation_count=0,
                    hallucination_risk=1.0,
                    issues=["No relevant sources found"],
                    suggestions=["Try rephrasing your query"],
                ),
                query=request.query,
                processing_time_ms=processing_time,
            )

        # Build context
        context_builder = ContextBuilder()
        built_context = context_builder.build(filtered_results)

        # Map prompt style
        prompt_style_map = {
            "concise": PromptStyle.CONCISE,
            "detailed": PromptStyle.DETAILED,
            "academic": PromptStyle.ACADEMIC,
            "conversational": PromptStyle.CONVERSATIONAL,
        }
        prompt_style = prompt_style_map.get(request.prompt_style.value, PromptStyle.DETAILED)

        # Generate response
        response = pipeline.llm_service.generate(
            query=request.query,
            context=built_context.formatted_context,
        )

        # Assess quality
        quality_checker = ResponseQualityChecker()
        assessment = quality_checker.assess(response.answer, filtered_results, request.query)

        # Prepare source documents
        sources = [
            SourceDocument(
                source_number=src.source_number,
                content=src.content[:500] + "..." if len(src.content) > 500 else src.content,
                filename=src.filename,
                chunk_index=src.chunk_index,
                relevance_score=src.relevance_score,
            )
            for src in built_context.sources
        ]

        processing_time = (time.time() - start_time) * 1000

        return QueryResponse(
            answer=response.answer,
            sources=sources,
            quality=QualityInfo(
                level=QualityLevel(assessment.quality.value),
                confidence=assessment.confidence,
                has_citations=assessment.has_citations,
                citation_count=assessment.citation_count,
                hallucination_risk=assessment.estimated_hallucination_risk,
                issues=assessment.issues,
                suggestions=assessment.suggestions,
            ),
            query=request.query,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}",
        )


# Search Endpoints


@search_router.post(
    "/",
    response_model=SearchResponse,
    summary="Semantic Search",
    description="Search for relevant documents using semantic similarity.",
)
async def search(request: SearchRequest, pipeline=Depends(get_pipeline)):
    """Perform semantic search on indexed documents."""
    start_time = time.time()

    try:
        search_results = pipeline.vector_store.search(
            query=request.query,
            top_k=request.top_k,
        )

        # Filter by minimum score
        filtered_results = [r for r in search_results if r.score >= request.min_score]

        results = [
            SearchResult(
                content=r.content[:500] + "..." if len(r.content) > 500 else r.content,
                filename=r.metadata.get("filename", "Unknown"),
                chunk_index=r.metadata.get("chunk_index", 0),
                score=r.score,
                metadata=r.metadata,
            )
            for r in filtered_results
        ]

        processing_time = (time.time() - start_time) * 1000

        return SearchResponse(
            results=results,
            query=request.query,
            total_results=len(results),
            processing_time_ms=processing_time,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@search_router.get(
    "/similar/{doc_id}",
    response_model=SearchResponse,
    summary="Find Similar Documents",
    description="Find documents similar to a specific document by ID.",
)
async def find_similar(doc_id: str, top_k: int = 5, pipeline=Depends(get_pipeline)):
    """Find documents similar to a given document."""
    start_time = time.time()

    try:
        # Get the original document
        # This would require implementing get_by_id in vector store
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Similar document search not yet implemented",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similar search failed: {str(e)}",
        )


# Ingestion Endpoints


@ingest_router.post(
    "/file",
    response_model=IngestResponse,
    summary="Ingest Files",
    description="Ingest documents from a file path or directory.",
)
async def ingest_files(request: IngestRequest, pipeline=Depends(get_pipeline)):
    """Ingest documents from file system."""
    start_time = time.time()
    errors = []

    try:
        path = Path(request.file_path)

        if not path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Path not found: {request.file_path}",
            )

        docs_processed = 0
        chunks_created = 0

        if path.is_file():
            # Single file ingestion
            try:
                result = pipeline.ingest_file(str(path))
                docs_processed = 1
                chunks_created = result.get("chunks", 0)
            except Exception as e:
                errors.append(f"Failed to ingest {path.name}: {str(e)}")

        elif path.is_dir():
            # Directory ingestion
            result = pipeline.ingest_directory(
                str(path),
                recursive=request.recursive,
                file_types=request.file_types,
            )
            docs_processed = result.get("documents", 0)
            chunks_created = result.get("chunks", 0)
            if result.get("errors"):
                errors.extend(result["errors"])

        processing_time = (time.time() - start_time) * 1000

        return IngestResponse(
            success=len(errors) == 0,
            documents_processed=docs_processed,
            chunks_created=chunks_created,
            errors=errors,
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )


@ingest_router.post(
    "/text",
    response_model=IngestResponse,
    summary="Ingest Text",
    description="Directly ingest text content.",
)
async def ingest_text(request: IngestTextRequest, pipeline=Depends(get_pipeline)):
    """Ingest text content directly."""
    start_time = time.time()

    try:
        result = pipeline.ingest_text(
            content=request.content,
            filename=request.filename,
            metadata=request.metadata,
        )

        processing_time = (time.time() - start_time) * 1000

        return IngestResponse(
            success=True,
            documents_processed=1,
            chunks_created=result.get("chunks", 1),
            errors=[],
            processing_time_ms=processing_time,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text ingestion failed: {str(e)}",
        )


# Admin Endpoints


@admin_router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Get Statistics",
    description="Get statistics about the indexed documents.",
)
async def get_stats(pipeline=Depends(get_pipeline)):
    """Get document store statistics."""
    try:
        stats = pipeline.vector_store.get_stats()

        return StatsResponse(
            total_documents=stats.get("total_documents", 0),
            total_chunks=stats.get("total_chunks", 0),
            unique_files=stats.get("unique_files", 0),
            collection_name=stats.get("collection_name", "default"),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}",
        )


@admin_router.delete(
    "/clear",
    summary="Clear Index",
    description="Clear all indexed documents (requires confirmation).",
)
async def clear_index(confirm: bool = False, pipeline=Depends(get_pipeline)):
    """Clear all indexed documents."""
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Set confirm=true to clear the index",
        )

    try:
        pipeline.vector_store.clear()
        return {"success": True, "message": "Index cleared successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear index: {str(e)}",
        )


# Include all routers
router.include_router(query_router)
router.include_router(search_router)
router.include_router(ingest_router)
router.include_router(admin_router)
