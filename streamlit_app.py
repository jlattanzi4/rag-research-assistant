"""Streamlit Cloud app for RAG Research Assistant - Portfolio Edition."""

import os
import time

import streamlit as st

# Page config
st.set_page_config(
    page_title="RAG Research Assistant | Joseph Lattanzi",
    page_icon="./assets/favicon.ico" if os.path.exists("./assets/favicon.ico") else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1.2rem;
        color: #94a3b8;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    .hero-description {
        font-size: 1rem;
        color: #cbd5e1;
        line-height: 1.7;
        max-width: 600px;
    }

    /* Tech badges */
    .tech-stack {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1.5rem;
    }

    .tech-badge {
        background: rgba(255,255,255,0.1);
        color: #e2e8f0;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Chat container */
    .chat-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
    }

    /* Message styles */
    .user-message {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 12px 12px 4px 12px;
        margin: 1rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }

    .assistant-message {
        background: #f8fafc;
        color: #1e293b;
        padding: 1.25rem;
        border-radius: 12px 12px 12px 4px;
        margin: 1rem 0;
        max-width: 85%;
        border: 1px solid #e2e8f0;
        line-height: 1.7;
    }

    /* Source cards */
    .source-card {
        background: #f1f5f9;
        border-left: 3px solid #3b82f6;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }

    .source-title {
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }

    .source-content {
        color: #64748b;
        font-size: 0.85rem;
        line-height: 1.6;
    }

    .relevance-badge {
        background: #dbeafe;
        color: #1d4ed8;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    /* Sidebar styling */
    .sidebar-section {
        background: #f8fafc;
        padding: 1.25rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }

    .sidebar-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #374151;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }

    /* Stats display */
    .stat-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.25rem;
        border-radius: 10px;
        text-align: center;
    }

    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #60a5fa;
    }

    .stat-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }

    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: border-color 0.2s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #374151;
    }

    /* Response time badge */
    .response-time {
        display: inline-block;
        background: #f0fdf4;
        color: #166534;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }

    /* Loading animation */
    .loading-text {
        color: #64748b;
        font-style: italic;
    }

    /* Link styling */
    a {
        color: #3b82f6;
        text-decoration: none;
    }

    a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False


@st.cache_resource
def init_pipeline():
    """Initialize the RAG pipeline (cached)."""
    if hasattr(st, "secrets"):
        os.environ["LLM_PROVIDER"] = st.secrets.get("LLM_PROVIDER", "groq")
        os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")
        os.environ["GROQ_MODEL"] = st.secrets.get("GROQ_MODEL", "llama-3.1-8b-instant")
        os.environ["EMBEDDING_MODEL"] = st.secrets.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    from rag_assistant import create_pipeline
    return create_pipeline()


def render_hero():
    """Render the hero section."""
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">RAG Research Assistant</div>
        <div class="hero-subtitle">Intelligent Document Q&A System</div>
        <div class="hero-description">
            A production-ready Retrieval-Augmented Generation system that searches through documents,
            retrieves relevant context, and generates accurate answers with source citations.
        </div>
        <div class="tech-stack">
            <span class="tech-badge">Python</span>
            <span class="tech-badge">LangChain</span>
            <span class="tech-badge">Llama 3.1</span>
            <span class="tech-badge">ChromaDB</span>
            <span class="tech-badge">HuggingFace</span>
            <span class="tech-badge">FastAPI</span>
            <span class="tech-badge">Streamlit</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar."""
    with st.sidebar:
        st.markdown('<div class="sidebar-title">System Status</div>', unsafe_allow_html=True)

        if not st.session_state.initialized:
            if st.button("Initialize System", type="primary", use_container_width=True):
                with st.spinner("Loading models..."):
                    try:
                        st.session_state.pipeline = init_pipeline()

                        if st.session_state.pipeline.vector_store.count() == 0:
                            from pathlib import Path
                            sample_dir = Path(__file__).parent / "data" / "sample"
                            if sample_dir.exists():
                                st.session_state.pipeline.ingest_directory(
                                    sample_dir, show_progress=False
                                )

                        st.session_state.initialized = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Initialization failed: {e}")
        else:
            st.success("System Online")

            stats = st.session_state.pipeline.vector_store.get_stats()

            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{stats.get("total_chunks", 0)}</div>
                <div class="stat-label">Documents Indexed</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-title">Configuration</div>', unsafe_allow_html=True)

        top_k = st.slider("Sources to retrieve", 1, 10, 5,
                         help="Number of relevant document chunks to use for context")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">Architecture</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-section">
            <strong>Retrieval</strong><br>
            <span style="color: #64748b; font-size: 0.85rem;">
            Semantic search using sentence embeddings to find relevant document chunks
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-section">
            <strong>Generation</strong><br>
            <span style="color: #64748b; font-size: 0.85rem;">
            Llama 3.1 8B via Groq API for fast, accurate response generation
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-section">
            <strong>Vector Store</strong><br>
            <span style="color: #64748b; font-size: 0.85rem;">
            ChromaDB for persistent storage and efficient similarity search
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">About</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="color: #64748b; font-size: 0.85rem; line-height: 1.6;">
            Built by <strong>Joseph Lattanzi</strong><br><br>
            <a href="https://github.com/jlattanzi4/rag-research-assistant" target="_blank">View Source Code</a>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.initialized:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Reset System", use_container_width=True):
                st.session_state.initialized = False
                st.session_state.messages = []
                st.cache_resource.clear()
                st.rerun()

    return top_k


def render_chat_interface(top_k: int):
    """Render the chat interface."""

    if not st.session_state.initialized:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #64748b;">
            <p style="font-size: 1.1rem; margin-bottom: 1rem;">
                Initialize the system using the sidebar to start asking questions.
            </p>
            <p style="font-size: 0.9rem;">
                <strong>Sample questions to try:</strong><br>
                What is machine learning?<br>
                How do neural networks learn?<br>
                What are the benefits of RAG systems?
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>',
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">{message["content"]}</div>',
                       unsafe_allow_html=True)

            if message.get("sources"):
                with st.expander(f"View Sources ({len(message['sources'])} documents)"):
                    for i, source in enumerate(message["sources"], 1):
                        filename = source.metadata.get('filename', 'Unknown')
                        score = source.score
                        content = source.content[:250] + "..." if len(source.content) > 250 else source.content

                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-title">
                                Source {i}: {filename}
                                <span class="relevance-badge">{score:.0%} relevant</span>
                            </div>
                            <div class="source-content">{content}</div>
                        </div>
                        """, unsafe_allow_html=True)

            if message.get("response_time"):
                st.markdown(f'<span class="response-time">Generated in {message["response_time"]:.1f}s</span>',
                           unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask a question about the documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)

        with st.spinner("Searching documents and generating response..."):
            start_time = time.time()
            response = st.session_state.pipeline.query(
                prompt, top_k=top_k, show_sources=False
            )
            elapsed = time.time() - start_time

        st.markdown(f'<div class="assistant-message">{response.answer}</div>',
                   unsafe_allow_html=True)

        if response.sources:
            with st.expander(f"View Sources ({len(response.sources)} documents)"):
                for i, source in enumerate(response.sources, 1):
                    filename = source.metadata.get('filename', 'Unknown')
                    score = source.score
                    content = source.content[:250] + "..." if len(source.content) > 250 else source.content

                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">
                            Source {i}: {filename}
                            <span class="relevance-badge">{score:.0%} relevant</span>
                        </div>
                        <div class="source-content">{content}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown(f'<span class="response-time">Generated in {elapsed:.1f}s</span>',
                   unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
            "response_time": elapsed,
        })


def main():
    """Main application."""
    render_hero()
    top_k = render_sidebar()
    render_chat_interface(top_k)


if __name__ == "__main__":
    main()
