"""LLM service for response generation using Ollama (free, local)."""

from dataclasses import dataclass

from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

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


class LLMService:
    """Service for LLM-based response generation using Ollama."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize the LLM service.

        Args:
            model: Model name. Defaults to settings.ollama_model.
            temperature: Temperature for generation. Defaults to settings.
            max_tokens: Maximum tokens for response. Defaults to settings.
            base_url: Ollama server URL. Defaults to settings.ollama_base_url.
        """
        self.model = model or settings.ollama_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.base_url = base_url or settings.ollama_base_url

        # Initialize Ollama client (free, local)
        self.llm = OllamaLLM(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            num_predict=self.max_tokens,
        )

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_USER_PROMPT_TEMPLATE),
        ])

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

        # Build the full prompt (Ollama works best with a single string prompt)
        full_prompt = f"""{RAG_SYSTEM_PROMPT}

{RAG_USER_PROMPT_TEMPLATE.format(context=context, question=query)}"""

        # Generate response
        response = self.llm.invoke(full_prompt)

        return RAGResponse(
            answer=response,
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
        response = self.llm.invoke(prompt)
        return response
