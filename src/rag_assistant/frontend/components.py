"""Reusable UI components for the Streamlit frontend."""

import streamlit as st

from rag_assistant.frontend.api_client import RAGAPIClient


def sidebar_settings() -> dict:
    """Render sidebar settings and return configuration.

    Returns:
        Dictionary with current settings.
    """
    # API endpoint configuration
    api_url = st.text_input(
        "API URL",
        value=st.session_state.get("api_url", "http://localhost:8000"),
        help="URL of the RAG API server",
    )
    st.session_state.api_url = api_url

    # Initialize API client
    api_client = RAGAPIClient(base_url=api_url)

    st.divider()

    # Query settings
    st.subheader("Query Settings")

    top_k = st.slider(
        "Number of sources",
        min_value=1,
        max_value=20,
        value=st.session_state.get("top_k", 5),
        help="Number of source documents to retrieve",
    )
    st.session_state.top_k = top_k

    prompt_style = st.selectbox(
        "Response style",
        options=["detailed", "concise", "academic", "conversational"],
        index=["detailed", "concise", "academic", "conversational"].index(
            st.session_state.get("prompt_style", "detailed")
        ),
        help="Style of the generated response",
    )
    st.session_state.prompt_style = prompt_style

    min_relevance = st.slider(
        "Minimum relevance",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get("min_relevance", 0.0),
        step=0.1,
        help="Minimum relevance score for sources",
    )
    st.session_state.min_relevance = min_relevance

    return {
        "api_client": api_client,
        "top_k": top_k,
        "prompt_style": prompt_style,
        "min_relevance": min_relevance,
    }


def file_uploader_section(api_client: RAGAPIClient):
    """Render file upload section.

    Args:
        api_client: API client instance.
    """
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["txt", "pdf", "md", "html"],
        accept_multiple_files=True,
        help="Upload documents to index",
    )

    if uploaded_files and st.button("Index Documents"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing: {file.name}")
            progress_bar.progress((i + 1) / len(uploaded_files))

            # Read file content
            content = file.read().decode("utf-8", errors="ignore")

            # Ingest via API
            result = api_client.ingest_text(
                content=content,
                filename=file.name,
                metadata={"source": "upload"},
            )

            if result and result.get("success"):
                st.success(f"Indexed {file.name}: {result.get('chunks_created', 0)} chunks")
            else:
                st.error(f"Failed to index {file.name}")

        progress_bar.empty()
        status_text.empty()
        st.success(f"Finished indexing {len(uploaded_files)} files")

        # Refresh stats
        stats = api_client.get_stats()
        if stats:
            st.session_state.stats = stats


def display_quality_metrics(quality: dict):
    """Display quality assessment metrics.

    Args:
        quality: Quality assessment dictionary.
    """
    level = quality.get("level", "unknown")
    confidence = quality.get("confidence", 0)
    hallucination_risk = quality.get("hallucination_risk", 0)

    # Quality badge color mapping
    badge_colors = {
        "high": "green",
        "medium": "orange",
        "low": "red",
        "insufficient": "red",
    }
    color = badge_colors.get(level, "gray")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Quality", level.upper())

    with col2:
        st.metric("Confidence", f"{confidence:.0%}")

    with col3:
        st.metric("Citations", quality.get("citation_count", 0))

    with col4:
        st.metric("Hallucination Risk", f"{hallucination_risk:.0%}")

    # Issues and suggestions
    issues = quality.get("issues", [])
    suggestions = quality.get("suggestions", [])

    if issues:
        with st.expander("Issues", expanded=False):
            for issue in issues:
                st.warning(issue)

    if suggestions:
        with st.expander("Suggestions", expanded=False):
            for suggestion in suggestions:
                st.info(suggestion)


def display_sources(sources: list):
    """Display source documents.

    Args:
        sources: List of source documents.
    """
    for source in sources:
        with st.expander(
            f"[Source {source.get('source_number', '?')}] "
            f"{source.get('filename', 'Unknown')} "
            f"(relevance: {source.get('relevance_score', 0):.0%})",
            expanded=False,
        ):
            st.markdown(source.get("content", "No content available."))

            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"Chunk: {source.get('chunk_index', 'N/A')}")
            with col2:
                st.caption(f"Relevance: {source.get('relevance_score', 0):.2%}")


def display_search_results(results: list):
    """Display search results.

    Args:
        results: List of search results.
    """
    if not results:
        st.info("No results found.")
        return

    for i, result in enumerate(results):
        with st.container():
            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"**{result.get('filename', 'Unknown')}**")
                content = result.get("content", "")
                # Truncate long content
                if len(content) > 300:
                    content = content[:300] + "..."
                st.markdown(content)

            with col2:
                score = result.get("score", 0)
                st.metric("Score", f"{score:.0%}")

            st.divider()


def chat_message(role: str, content: str, sources: list | None = None):
    """Display a chat message.

    Args:
        role: Message role ('user' or 'assistant').
        content: Message content.
        sources: Optional list of sources for assistant messages.
    """
    with st.chat_message(role):
        st.markdown(content)

        if sources and role == "assistant":
            with st.expander("View Sources", expanded=False):
                display_sources(sources)
