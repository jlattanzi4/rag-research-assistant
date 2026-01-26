"""Generation module for LLM response generation."""

from .context import BuiltContext, ContextBuilder, ContextSource, DiversityAwareContextBuilder
from .llm_service import LLMService, RAGResponse
from .prompts import (
    PROMPT_TEMPLATES,
    PromptStyle,
    PromptTemplate,
    get_prompt_template,
)
from .quality import (
    EdgeCaseHandler,
    QualityAssessment,
    QueryAnalyzer,
    QueryType,
    ResponseQuality,
    ResponseQualityChecker,
)

__all__ = [
    # LLM Service
    "LLMService",
    "RAGResponse",
    # Prompts
    "PromptStyle",
    "PromptTemplate",
    "PROMPT_TEMPLATES",
    "get_prompt_template",
    # Context
    "ContextBuilder",
    "DiversityAwareContextBuilder",
    "BuiltContext",
    "ContextSource",
    # Quality
    "ResponseQuality",
    "QueryType",
    "QualityAssessment",
    "QueryAnalyzer",
    "ResponseQualityChecker",
    "EdgeCaseHandler",
]
