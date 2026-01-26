"""LLM service for response generation using Ollama (local) or Groq (cloud)."""

from dataclasses import dataclass

from rag_assistant.config import settings
from rag_assistant.retrieval.vector_store import SearchResult


@dataclass
class RAGResponse:
    """Represents a response from the RAG system."""

    answer: str
    sources: list[SearchResult]
    query: str


# System prompt for RAG responses
RAG_SYSTEM_PROMPT = """You are a helpful research assistant that answers questions based on the provided context.

Your task is to:
1. Answer the user's question using ONLY the information from the provided context
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Always cite your sources by referencing the source document
4. Be concise but thorough in your responses

Important guidelines:
- Do NOT make up information that isn't in the context
- If you're unsure, express that uncertainty
- Use direct quotes when appropriate, with proper attribution
- Structure your response clearly for readability"""

RAG_USER_PROMPT_TEMPLATE = """Context from relevant documents:
{context}

---

Question: {question}

Please provide a comprehensive answer based on the context above. Include citations to the source documents."""


def create_llm(provider: str | None = None):
    """Create an LLM instance based on the provider.

    Args:
        provider: LLM provider ('ollama' or 'groq'). Defaults to settings.

    Returns:
        LLM instance.
    """
    provider = provider or settings.llm_provider

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    else:  # Default to Ollama
        from langchain_ollama import OllamaLLM
        return OllamaLLM(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens,
        )


class LLMService:
    """Service for LLM-based response generation."""

    def __init__(
        self,
        provider: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize the LLM service.

        Args:
            provider: LLM provider ('ollama' or 'groq'). Defaults to settings.
            temperature: Temperature for generation. Defaults to settings.
            max_tokens: Maximum tokens for response. Defaults to settings.
        """
        self.provider = provider or settings.llm_provider
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens

        # Initialize the LLM
        self.llm = create_llm(self.provider)

    def _format_context(self, search_results: list[SearchResult]) -> str:
        """Format search results into context string.

        Args:
            search_results: List of search results from vector store.

        Returns:
            Formatted context string.
        """
        context_parts = []

        for i, result in enumerate(search_results, 1):
            source = result.metadata.get("filename", "Unknown source")
            chunk_index = result.metadata.get("chunk_index", 0)
            score = result.score

            context_parts.append(
                f"[Source {i}: {source} (chunk {chunk_index}, relevance: {score:.2f})]\n"
                f"{result.content}\n"
            )

        return "\n---\n".join(context_parts)

    def generate(
        self, query: str, search_results: list[SearchResult] | None = None, context: str | None = None
    ) -> RAGResponse:
        """Generate a response using the LLM.

        Args:
            query: User's question.
            search_results: Retrieved context from vector store.
            context: Pre-formatted context string (alternative to search_results).

        Returns:
            RAGResponse with answer and sources.
        """
        if not search_results and not context:
            return RAGResponse(
                answer="I couldn't find any relevant information in the documents to answer your question.",
                sources=[],
                query=query,
            )

        # Format context from search results if not provided directly
        if context is None:
            context = self._format_context(search_results)

        # Build the full prompt
        full_prompt = f"""{RAG_SYSTEM_PROMPT}

{RAG_USER_PROMPT_TEMPLATE.format(context=context, question=query)}"""

        # Generate response
        if self.provider == "groq":
            # Groq uses chat interface
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=full_prompt)])
            answer = response.content
        else:
            # Ollama uses simple string interface
            answer = self.llm.invoke(full_prompt)

        return RAGResponse(
            answer=answer,
            sources=search_results or [],
            query=query,
        )

    def generate_simple(self, prompt: str) -> str:
        """Generate a simple response without RAG context.

        Args:
            prompt: User prompt.

        Returns:
            Generated response text.
        """
        if self.provider == "groq":
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        else:
            return self.llm.invoke(prompt)
