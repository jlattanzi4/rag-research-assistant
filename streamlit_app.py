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

# Advanced CSS - Pushing Streamlit to its limits
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* CSS Variables for theming */
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --accent: #22d3ee;
        --bg-dark: #0f0f1a;
        --bg-card: rgba(255,255,255,0.03);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --glass-bg: rgba(255,255,255,0.05);
        --glass-border: rgba(255,255,255,0.1);
    }

    /* Global dark theme */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #0f0f1a 100%);
        background-attachment: fixed;
    }

    /* Animated background gradient */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background:
            radial-gradient(ellipse at 20% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(34, 211, 238, 0.1) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(139, 92, 246, 0.05) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
        animation: gradientShift 15s ease-in-out infinite;
    }

    @keyframes gradientShift {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.02);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.3);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.5);
    }

    /* Glassmorphism Hero */
    .hero-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(34, 211, 238, 0.05) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 3rem 2.5rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.8s ease-out;
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    }

    .hero-container::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.3; }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Gradient text for title */
    .hero-title {
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #6366f1 50%, #22d3ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        font-size: 1.3rem;
        color: var(--text-secondary);
        font-weight: 400;
        margin-bottom: 1.5rem;
        position: relative;
        z-index: 1;
    }

    .hero-description {
        font-size: 1rem;
        color: #cbd5e1;
        line-height: 1.8;
        max-width: 650px;
        position: relative;
        z-index: 1;
    }

    /* Animated tech badges */
    .tech-stack {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 2rem;
        position: relative;
        z-index: 1;
    }

    .tech-badge {
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        border: 1px solid rgba(99, 102, 241, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: default;
    }

    .tech-badge:hover {
        background: rgba(99, 102, 241, 0.25);
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.2);
    }

    /* Glass chat container */
    .chat-container {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid var(--glass-border);
    }

    /* Message bubbles with animations */
    .user-message {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        padding: 1.1rem 1.4rem;
        border-radius: 20px 20px 4px 20px;
        margin: 1rem 0;
        max-width: 75%;
        margin-left: auto;
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.3);
        animation: slideInRight 0.4s ease-out;
        position: relative;
        overflow: hidden;
    }

    .user-message::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, transparent 50%);
        pointer-events: none;
    }

    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    .assistant-message {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        color: var(--text-primary);
        padding: 1.4rem 1.6rem;
        border-radius: 20px 20px 20px 4px;
        margin: 1rem 0;
        max-width: 85%;
        border: 1px solid rgba(255,255,255,0.08);
        line-height: 1.8;
        animation: slideInLeft 0.4s ease-out;
    }

    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* Source cards with hover effects */
    .source-card {
        background: rgba(99, 102, 241, 0.08);
        border-left: 3px solid #6366f1;
        padding: 1.2rem;
        margin: 0.75rem 0;
        border-radius: 0 12px 12px 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: default;
    }

    .source-card:hover {
        background: rgba(99, 102, 241, 0.12);
        transform: translateX(8px);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.15);
    }

    .source-title {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        flex-wrap: wrap;
    }

    .source-content {
        color: var(--text-secondary);
        font-size: 0.88rem;
        line-height: 1.7;
    }

    .relevance-badge {
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.2) 0%, rgba(99, 102, 241, 0.2) 100%);
        color: #22d3ee;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(34, 211, 238, 0.3);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 15, 26, 0.98) 0%, rgba(26, 26, 46, 0.98) 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--text-secondary);
    }

    .sidebar-section {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(10px);
        padding: 1.25rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.05);
        transition: all 0.3s ease;
    }

    .sidebar-section:hover {
        background: rgba(255,255,255,0.05);
        border-color: rgba(99, 102, 241, 0.2);
    }

    .sidebar-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
    }

    /* Animated stat box */
    .stat-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(34, 211, 238, 0.1) 100%);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        border: 1px solid rgba(99, 102, 241, 0.2);
        position: relative;
        overflow: hidden;
    }

    .stat-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.03), transparent);
        animation: shimmer 3s infinite;
    }

    @keyframes shimmer {
        0% { transform: translateX(-100%) rotate(45deg); }
        100% { transform: translateX(100%) rotate(45deg); }
    }

    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #22d3ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
        z-index: 1;
    }

    .stat-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }

    /* Button styling with glow */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border: none;
        padding: 0.85rem 1.75rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        position: relative;
        overflow: hidden;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.4);
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    .stButton > button:active {
        transform: translateY(-1px);
    }

    /* Input styling */
    .stTextInput > div > div > input,
    .stChatInput > div > div > textarea {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 14px !important;
        border: 2px solid rgba(255,255,255,0.1) !important;
        padding: 0.9rem 1.2rem !important;
        font-size: 1rem !important;
        color: var(--text-primary) !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput > div > div > input:focus,
    .stChatInput > div > div > textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15) !important;
        background: rgba(255,255,255,0.08) !important;
    }

    .stTextInput > div > div > input::placeholder,
    .stChatInput > div > div > textarea::placeholder {
        color: var(--text-secondary) !important;
    }

    /* Chat input container */
    [data-testid="stChatInput"] {
        background: transparent !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: var(--text-primary) !important;
        background: rgba(255,255,255,0.03) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease;
    }

    .streamlit-expanderHeader:hover {
        background: rgba(255,255,255,0.06) !important;
    }

    .streamlit-expanderContent {
        background: rgba(0,0,0,0.2) !important;
        border-radius: 0 0 12px 12px !important;
    }

    /* Response time badge */
    .response-time {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(34, 211, 238, 0.1);
        color: #22d3ee;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.75rem;
        border: 1px solid rgba(34, 211, 238, 0.2);
        animation: fadeIn 0.5s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* Loading animation */
    .loading-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--text-secondary);
        padding: 1rem;
    }

    .loading-dots {
        display: flex;
        gap: 4px;
    }

    .loading-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #6366f1;
        animation: loadingBounce 1.4s ease-in-out infinite;
    }

    .loading-dot:nth-child(1) { animation-delay: 0s; }
    .loading-dot:nth-child(2) { animation-delay: 0.2s; }
    .loading-dot:nth-child(3) { animation-delay: 0.4s; }

    @keyframes loadingBounce {
        0%, 80%, 100% { transform: scale(0.6); opacity: 0.5; }
        40% { transform: scale(1); opacity: 1; }
    }

    /* Link styling */
    a {
        color: #6366f1;
        text-decoration: none;
        transition: all 0.2s ease;
    }

    a:hover {
        color: #818cf8;
        text-decoration: none;
    }

    /* Status indicators */
    .status-online {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: #22c55e;
        font-weight: 500;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        animation: statusPulse 2s ease-in-out infinite;
    }

    @keyframes statusPulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.4); }
        50% { box-shadow: 0 0 0 8px rgba(34, 197, 94, 0); }
    }

    /* Welcome card */
    .welcome-card {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        animation: fadeInUp 0.6s ease-out;
    }

    .welcome-icon {
        font-size: 3rem;
        margin-bottom: 1.5rem;
        opacity: 0.8;
    }

    .welcome-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
    }

    .welcome-text {
        color: var(--text-secondary);
        line-height: 1.7;
        max-width: 500px;
        margin: 0 auto;
    }

    .sample-questions {
        margin-top: 2rem;
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 0.75rem;
    }

    .sample-question {
        background: rgba(99, 102, 241, 0.1);
        color: #a5b4fc;
        padding: 0.6rem 1.2rem;
        border-radius: 20px;
        font-size: 0.85rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        cursor: default;
        transition: all 0.3s ease;
    }

    .sample-question:hover {
        background: rgba(99, 102, 241, 0.2);
        transform: translateY(-2px);
    }

    /* Slider styling */
    .stSlider > div > div > div {
        background: rgba(99, 102, 241, 0.3) !important;
    }

    .stSlider > div > div > div > div {
        background: #6366f1 !important;
    }

    /* Success message */
    .stSuccess {
        background: rgba(34, 197, 94, 0.1) !important;
        border: 1px solid rgba(34, 197, 94, 0.2) !important;
        color: #22c55e !important;
    }

    /* Error message */
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.2) !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-color: #6366f1 !important;
    }

    /* Architecture cards */
    .arch-card {
        background: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.1);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        transition: all 0.3s ease;
    }

    .arch-card:hover {
        background: rgba(99, 102, 241, 0.1);
        border-color: rgba(99, 102, 241, 0.2);
    }

    .arch-title {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }

    .arch-desc {
        color: var(--text-secondary);
        font-size: 0.8rem;
        line-height: 1.5;
    }

    /* Footer link */
    .footer-link {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--text-secondary);
        font-size: 0.85rem;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.05);
        transition: all 0.3s ease;
        text-decoration: none;
    }

    .footer-link:hover {
        background: rgba(99, 102, 241, 0.1);
        border-color: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
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
        <div class="hero-subtitle">Intelligent Document Analysis Powered by AI</div>
        <div class="hero-description">
            A production-grade Retrieval-Augmented Generation system that transforms how you interact with documents.
            Semantic search meets large language models for accurate, cited answers.
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
            st.markdown("""
            <div class="status-online">
                <span class="status-dot"></span>
                System Online
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

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
        <div class="arch-card">
            <div class="arch-title">Semantic Retrieval</div>
            <div class="arch-desc">Sentence embeddings find contextually relevant document chunks</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="arch-card">
            <div class="arch-title">LLM Generation</div>
            <div class="arch-desc">Llama 3.1 8B via Groq for fast, accurate responses</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="arch-card">
            <div class="arch-title">Vector Storage</div>
            <div class="arch-desc">ChromaDB for persistent, efficient similarity search</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">About</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 1rem;">
            Built by <strong style="color: var(--text-primary);">Joseph Lattanzi</strong>
        </div>
        <a href="https://github.com/jlattanzi4/rag-research-assistant" target="_blank" class="footer-link">
            View Source Code
        </a>
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
        <div class="welcome-card">
            <div class="welcome-icon">&#9889;</div>
            <div class="welcome-title">Ready to Explore Your Documents</div>
            <div class="welcome-text">
                Initialize the system using the sidebar to start asking questions.
                The AI will search through your documents and provide accurate answers with citations.
            </div>
            <div class="sample-questions">
                <span class="sample-question">What is machine learning?</span>
                <span class="sample-question">How do neural networks work?</span>
                <span class="sample-question">Benefits of RAG systems?</span>
            </div>
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
