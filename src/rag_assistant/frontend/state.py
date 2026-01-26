"""Session state management for Streamlit frontend."""

import streamlit as st


def init_session_state():
    """Initialize Streamlit session state with default values."""
    defaults = {
        # API settings
        "api_url": "http://localhost:8000",
        # Query settings
        "top_k": 5,
        "prompt_style": "detailed",
        "min_relevance": 0.0,
        # Query state
        "query_history": [],
        "current_response": None,
        # Search state
        "search_results": None,
        # Stats
        "stats": None,
        # Chat mode
        "chat_messages": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_query_state():
    """Clear query-related state."""
    st.session_state.query_history = []
    st.session_state.current_response = None


def clear_search_state():
    """Clear search-related state."""
    st.session_state.search_results = None


def clear_all_state():
    """Clear all state except settings."""
    clear_query_state()
    clear_search_state()
    st.session_state.chat_messages = []


def add_chat_message(role: str, content: str, sources: list | None = None):
    """Add a message to chat history.

    Args:
        role: Message role ('user' or 'assistant').
        content: Message content.
        sources: Optional sources for assistant messages.
    """
    st.session_state.chat_messages.append({
        "role": role,
        "content": content,
        "sources": sources,
    })


def get_chat_messages() -> list:
    """Get all chat messages.

    Returns:
        List of chat messages.
    """
    return st.session_state.chat_messages


def update_stats(stats: dict):
    """Update statistics in session state.

    Args:
        stats: Statistics dictionary.
    """
    st.session_state.stats = stats
