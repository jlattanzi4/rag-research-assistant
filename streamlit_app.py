"""Streamlit Cloud app for RAG Research Assistant."""

import os
import time

import streamlit as st

# Set page config first
st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📚",
    layout="wide",
)

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
    # Set environment variables from Streamlit secrets
    if hasattr(st, "secrets"):
        os.environ["LLM_PROVIDER"] = st.secrets.get("LLM_PROVIDER", "groq")
        os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")
        os.environ["GROQ_MODEL"] = st.secrets.get("GROQ_MODEL", "llama-3.1-8b-instant")
        os.environ["EMBEDDING_MODEL"] = st.secrets.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    from rag_assistant import create_pipeline
    return create_pipeline()


def main():
    st.title("📚 RAG Research Assistant")
    st.markdown("Ask questions about documents and get AI-powered answers with citations.")

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown(
            """
            This is a **Retrieval-Augmented Generation (RAG)** system that:
            1. Searches through indexed documents
            2. Finds relevant passages
            3. Generates accurate answers with citations

            **Tech Stack:**
            - LangChain + Groq (Llama 3.1)
            - ChromaDB vector store
            - HuggingFace embeddings
            - Streamlit frontend
            """
        )

        st.divider()
        st.header("Settings")
        top_k = st.slider("Number of sources", 1, 10, 5)

        st.divider()

        # Initialize button
        if not st.session_state.initialized:
            if st.button("🚀 Initialize System", type="primary"):
                with st.spinner("Loading models and documents..."):
                    try:
                        st.session_state.pipeline = init_pipeline()

                        # Ingest sample documents if empty
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
                        st.error(f"Failed to initialize: {e}")
        else:
            st.success("✅ System ready!")
            stats = st.session_state.pipeline.vector_store.get_stats()
            st.metric("Documents indexed", stats.get("total_chunks", 0))

            if st.button("🔄 Reset"):
                st.session_state.initialized = False
                st.session_state.messages = []
                st.cache_resource.clear()
                st.rerun()

    # Main content
    if not st.session_state.initialized:
        st.info("👈 Click **Initialize System** in the sidebar to start.")

        st.markdown("### Sample Questions")
        st.markdown(
            """
            Once initialized, try asking:
            - "What is machine learning?"
            - "How do neural networks work?"
            - "What are the benefits of RAG?"
            - "What is backpropagation?"
            """
        )
        return

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**[Source {i}]** {source.metadata.get('filename', 'Unknown')}")
                        st.caption(source.content[:300] + "..." if len(source.content) > 300 else source.content)

    # Chat input
    if prompt := st.chat_input("Ask a question about the documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                response = st.session_state.pipeline.query(
                    prompt, top_k=top_k, show_sources=False
                )
                elapsed = time.time() - start_time

            st.markdown(response.answer)

            if response.sources:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(response.sources, 1):
                        st.markdown(f"**[Source {i}]** {source.metadata.get('filename', 'Unknown')} (relevance: {source.score:.0%})")
                        st.caption(source.content[:300] + "..." if len(source.content) > 300 else source.content)

            st.caption(f"⏱️ Response time: {elapsed:.1f}s")

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
        })


if __name__ == "__main__":
    main()
