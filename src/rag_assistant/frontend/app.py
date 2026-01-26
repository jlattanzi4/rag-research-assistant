"""Streamlit frontend for the RAG Research Assistant."""

import time
from pathlib import Path

import streamlit as st

from rag_assistant.frontend.components import (
    display_quality_metrics,
    display_search_results,
    display_sources,
    file_uploader_section,
    sidebar_settings,
)
from rag_assistant.frontend.state import init_session_state
from rag_assistant.frontend.api_client import RAGAPIClient


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="RAG Research Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    init_session_state()

    # Custom CSS
    st.markdown(
        """
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .source-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .quality-badge {
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-weight: bold;
        }
        .quality-high { background-color: #d4edda; color: #155724; }
        .quality-medium { background-color: #fff3cd; color: #856404; }
        .quality-low { background-color: #f8d7da; color: #721c24; }
        .quality-insufficient { background-color: #f5c6cb; color: #721c24; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.title("Settings")
        settings = sidebar_settings()

        st.divider()

        # File upload section
        st.subheader("Document Upload")
        file_uploader_section(settings["api_client"])

        st.divider()

        # Stats section
        if st.button("Refresh Stats"):
            with st.spinner("Loading stats..."):
                stats = settings["api_client"].get_stats()
                if stats:
                    st.session_state.stats = stats

        if "stats" in st.session_state and st.session_state.stats:
            st.subheader("Index Statistics")
            stats = st.session_state.stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats.get("total_chunks", 0))
            with col2:
                st.metric("Files", stats.get("unique_files", 0))

    # Main content
    st.title("RAG Research Assistant")
    st.markdown(
        "Ask questions about your documents and get AI-powered answers with citations."
    )

    # Tabs for different functionality
    tab1, tab2 = st.tabs(["Ask Questions", "Search Documents"])

    with tab1:
        query_tab(settings)

    with tab2:
        search_tab(settings)


def query_tab(settings: dict):
    """Query tab for asking questions."""
    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="What would you like to know about your documents?",
        height=100,
        key="query_input",
    )

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        submit = st.button("Ask", type="primary", use_container_width=True)
    with col2:
        clear = st.button("Clear", use_container_width=True)

    if clear:
        st.session_state.query_history = []
        st.session_state.current_response = None
        st.rerun()

    if submit and query:
        with st.spinner("Generating answer..."):
            start_time = time.time()
            response = settings["api_client"].query(
                query=query,
                top_k=settings["top_k"],
                prompt_style=settings["prompt_style"],
                min_relevance=settings["min_relevance"],
            )
            elapsed_time = time.time() - start_time

            if response:
                st.session_state.current_response = response
                st.session_state.current_response["client_time"] = elapsed_time

                # Add to history
                st.session_state.query_history.append({
                    "query": query,
                    "response": response,
                    "timestamp": time.strftime("%H:%M:%S"),
                })

    # Display current response
    if st.session_state.get("current_response"):
        response = st.session_state.current_response
        display_query_response(response)

    # Query history
    if st.session_state.get("query_history"):
        with st.expander("Query History", expanded=False):
            for i, item in enumerate(reversed(st.session_state.query_history[:-1])):
                st.markdown(f"**{item['timestamp']}**: {item['query']}")
                if st.button(f"Show Answer", key=f"history_{i}"):
                    st.session_state.current_response = item["response"]
                    st.rerun()


def display_query_response(response: dict):
    """Display a query response."""
    st.markdown("### Answer")
    st.markdown(response.get("answer", "No answer available."))

    # Quality metrics
    quality = response.get("quality", {})
    if quality:
        st.markdown("### Quality Assessment")
        display_quality_metrics(quality)

    # Sources
    sources = response.get("sources", [])
    if sources:
        st.markdown("### Sources")
        display_sources(sources)

    # Timing
    if "processing_time_ms" in response:
        st.caption(
            f"Server processing: {response['processing_time_ms']:.0f}ms | "
            f"Total time: {response.get('client_time', 0):.2f}s"
        )


def search_tab(settings: dict):
    """Search tab for semantic search."""
    search_query = st.text_input(
        "Search documents:",
        placeholder="Enter search terms...",
        key="search_input",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        search = st.button("Search", type="primary", use_container_width=True)

    if search and search_query:
        with st.spinner("Searching..."):
            results = settings["api_client"].search(
                query=search_query,
                top_k=settings["top_k"],
                min_score=settings["min_relevance"],
            )

            if results:
                st.session_state.search_results = results

    # Display search results
    if st.session_state.get("search_results"):
        results = st.session_state.search_results
        st.markdown(f"### Results ({results.get('total_results', 0)} found)")
        display_search_results(results.get("results", []))


if __name__ == "__main__":
    main()
