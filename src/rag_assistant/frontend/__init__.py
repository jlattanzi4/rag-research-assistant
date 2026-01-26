"""Frontend module for the RAG Research Assistant Streamlit app."""

from .api_client import RAGAPIClient
from .app import main
from .components import (
    chat_message,
    display_quality_metrics,
    display_search_results,
    display_sources,
    file_uploader_section,
    sidebar_settings,
)
from .state import (
    add_chat_message,
    clear_all_state,
    clear_query_state,
    clear_search_state,
    get_chat_messages,
    init_session_state,
    update_stats,
)

__all__ = [
    # Main app
    "main",
    # API Client
    "RAGAPIClient",
    # Components
    "sidebar_settings",
    "file_uploader_section",
    "display_quality_metrics",
    "display_sources",
    "display_search_results",
    "chat_message",
    # State management
    "init_session_state",
    "clear_query_state",
    "clear_search_state",
    "clear_all_state",
    "add_chat_message",
    "get_chat_messages",
    "update_stats",
]
