"""Response quality assessment and edge case handling."""

import re
from dataclasses import dataclass
from enum import Enum

from rag_assistant.retrieval.vector_store import SearchResult


class ResponseQuality(Enum):
    """Quality assessment levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class QueryType(Enum):
    """Types of queries for specialized handling."""

    FACTUAL = "factual"  # What, when, who questions
    EXPLANATORY = "explanatory"  # How, why questions
    COMPARATIVE = "comparative"  # Compare X and Y
    DEFINITIONAL = "definitional"  # What is X
    PROCEDURAL = "procedural"  # How to do X
    OPINION = "opinion"  # What do you think
    AMBIGUOUS = "ambiguous"  # Unclear intent


@dataclass
class QualityAssessment:
    """Assessment of response quality."""

    quality: ResponseQuality
    confidence: float  # 0-1
    issues: list[str]
    suggestions: list[str]
    has_citations: bool
    citation_count: int
    estimated_hallucination_risk: float  # 0-1


class QueryAnalyzer:
    """Analyzes queries to determine type and potential issues."""

    FACTUAL_PATTERNS = [
        r"^what (is|are|was|were)\b",
        r"^who (is|are|was|were)\b",
        r"^when (did|does|is|was)\b",
        r"^where (is|are|was|were)\b",
        r"^which\b",
    ]

    EXPLANATORY_PATTERNS = [
        r"^why\b",
        r"^how (does|do|did|is|are)\b",
        r"\bexplain\b",
        r"\bdescribe\b",
    ]

    COMPARATIVE_PATTERNS = [
        r"\bcompare\b",
        r"\bdifference between\b",
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bbetter than\b",
    ]

    DEFINITIONAL_PATTERNS = [
        r"^what is (a |an |the )?[\w\s]+\??$",
        r"^define\b",
        r"\bdefinition of\b",
    ]

    PROCEDURAL_PATTERNS = [
        r"^how (to|do I|can I)\b",
        r"\bsteps to\b",
        r"\bprocess of\b",
    ]

    def analyze(self, query: str) -> QueryType:
        """Analyze query to determine its type.

        Args:
            query: User query.

        Returns:
            QueryType classification.
        """
        query_lower = query.lower().strip()

        # Check patterns in order of specificity
        for pattern in self.DEFINITIONAL_PATTERNS:
            if re.search(pattern, query_lower):
                return QueryType.DEFINITIONAL

        for pattern in self.COMPARATIVE_PATTERNS:
            if re.search(pattern, query_lower):
                return QueryType.COMPARATIVE

        for pattern in self.PROCEDURAL_PATTERNS:
            if re.search(pattern, query_lower):
                return QueryType.PROCEDURAL

        for pattern in self.EXPLANATORY_PATTERNS:
            if re.search(pattern, query_lower):
                return QueryType.EXPLANATORY

        for pattern in self.FACTUAL_PATTERNS:
            if re.search(pattern, query_lower):
                return QueryType.FACTUAL

        # Check for opinion-seeking
        if any(phrase in query_lower for phrase in [
            "do you think", "your opinion", "should i", "would you recommend"
        ]):
            return QueryType.OPINION

        # Default to ambiguous if no clear pattern
        if len(query.split()) < 3:
            return QueryType.AMBIGUOUS

        return QueryType.FACTUAL


class ResponseQualityChecker:
    """Checks response quality and detects potential issues."""

    def __init__(
        self,
        min_sources_for_high_quality: int = 2,
        min_response_length: int = 50,
        hallucination_phrases: list[str] | None = None,
    ) -> None:
        """Initialize the quality checker.

        Args:
            min_sources_for_high_quality: Minimum sources for high quality.
            min_response_length: Minimum response length in chars.
            hallucination_phrases: Phrases that may indicate hallucination.
        """
        self.min_sources_for_high_quality = min_sources_for_high_quality
        self.min_response_length = min_response_length
        self.hallucination_phrases = hallucination_phrases or [
            "I believe",
            "I think",
            "In my opinion",
            "It's widely known",
            "Everyone knows",
            "Obviously",
            "Clearly",
            "As we all know",
        ]

    def check_citations(self, response: str) -> tuple[bool, int]:
        """Check for citations in response.

        Args:
            response: Generated response text.

        Returns:
            Tuple of (has_citations, citation_count).
        """
        # Look for [Source N] patterns
        citation_pattern = r"\[Source\s*\d+\]"
        citations = re.findall(citation_pattern, response, re.IGNORECASE)
        return len(citations) > 0, len(set(citations))

    def estimate_hallucination_risk(
        self,
        response: str,
        sources: list[SearchResult],
    ) -> float:
        """Estimate risk of hallucination in response.

        Args:
            response: Generated response.
            sources: Source documents used.

        Returns:
            Risk score 0-1 (higher = more risk).
        """
        risk = 0.0

        # Check for hallucination indicator phrases
        response_lower = response.lower()
        phrase_count = sum(
            1 for phrase in self.hallucination_phrases
            if phrase.lower() in response_lower
        )
        risk += min(0.3, phrase_count * 0.1)

        # Check if response contains claims not traceable to sources
        has_citations, _ = self.check_citations(response)
        if not has_citations and len(response) > 200:
            risk += 0.3

        # Low source relevance increases risk
        if sources:
            avg_score = sum(s.score for s in sources) / len(sources)
            if avg_score < 0.5:
                risk += 0.2
        else:
            risk += 0.4

        return min(1.0, risk)

    def assess(
        self,
        response: str,
        sources: list[SearchResult],
        query: str,
    ) -> QualityAssessment:
        """Assess overall response quality.

        Args:
            response: Generated response.
            sources: Source documents used.
            query: Original query.

        Returns:
            QualityAssessment with detailed analysis.
        """
        issues = []
        suggestions = []

        # Check citations
        has_citations, citation_count = self.check_citations(response)
        if not has_citations:
            issues.append("Response lacks citations")
            suggestions.append("Add [Source N] citations to support claims")

        # Check response length
        if len(response) < self.min_response_length:
            issues.append("Response may be too brief")
            suggestions.append("Consider providing more detail")

        # Check source quality
        if not sources:
            issues.append("No source documents available")
            suggestions.append("Expand search or rephrase query")
        elif len(sources) < self.min_sources_for_high_quality:
            issues.append("Limited source documents")

        # Estimate hallucination risk
        hallucination_risk = self.estimate_hallucination_risk(response, sources)
        if hallucination_risk > 0.5:
            issues.append("Higher risk of unsupported claims")
            suggestions.append("Verify claims against source documents")

        # Calculate confidence
        confidence = 1.0
        if not sources:
            confidence -= 0.5
        else:
            avg_relevance = sum(s.score for s in sources) / len(sources)
            confidence *= avg_relevance

        if not has_citations:
            confidence -= 0.2

        confidence = max(0.0, min(1.0, confidence))

        # Determine quality level
        if not sources or confidence < 0.3:
            quality = ResponseQuality.INSUFFICIENT
        elif confidence < 0.5 or len(issues) > 2:
            quality = ResponseQuality.LOW
        elif confidence < 0.7 or len(issues) > 0:
            quality = ResponseQuality.MEDIUM
        else:
            quality = ResponseQuality.HIGH

        return QualityAssessment(
            quality=quality,
            confidence=confidence,
            issues=issues,
            suggestions=suggestions,
            has_citations=has_citations,
            citation_count=citation_count,
            estimated_hallucination_risk=hallucination_risk,
        )


class EdgeCaseHandler:
    """Handles edge cases in RAG responses."""

    NO_RESULTS_RESPONSE = (
        "I couldn't find any relevant information in the available documents "
        "to answer your question. This could be because:\n"
        "- The topic isn't covered in the indexed documents\n"
        "- The query might need to be rephrased\n"
        "- More specific search terms might help"
    )

    LOW_RELEVANCE_RESPONSE = (
        "I found some potentially relevant information, but the match "
        "confidence is low. The response below should be verified:\n\n"
    )

    AMBIGUOUS_QUERY_RESPONSE = (
        "Your question seems ambiguous. Could you please clarify what "
        "specifically you'd like to know about? For example:\n"
    )

    def __init__(
        self,
        min_relevance_threshold: float = 0.3,
        min_results_for_answer: int = 1,
    ) -> None:
        """Initialize the edge case handler.

        Args:
            min_relevance_threshold: Minimum relevance score to use results.
            min_results_for_answer: Minimum results needed to generate answer.
        """
        self.min_relevance_threshold = min_relevance_threshold
        self.min_results_for_answer = min_results_for_answer
        self.query_analyzer = QueryAnalyzer()

    def should_answer(self, sources: list[SearchResult]) -> tuple[bool, str]:
        """Determine if we should generate an answer.

        Args:
            sources: Retrieved source documents.

        Returns:
            Tuple of (should_answer, reason_if_not).
        """
        if not sources:
            return False, "no_results"

        if len(sources) < self.min_results_for_answer:
            return False, "insufficient_results"

        # Check if any source meets relevance threshold
        max_relevance = max(s.score for s in sources)
        if max_relevance < self.min_relevance_threshold:
            return False, "low_relevance"

        return True, ""

    def handle_no_results(self, query: str) -> str:
        """Generate response when no results found.

        Args:
            query: Original query.

        Returns:
            Appropriate response message.
        """
        return self.NO_RESULTS_RESPONSE

    def handle_low_relevance(
        self, query: str, sources: list[SearchResult]
    ) -> str:
        """Generate prefix for low-relevance responses.

        Args:
            query: Original query.
            sources: Retrieved sources.

        Returns:
            Warning prefix to prepend to response.
        """
        max_score = max(s.score for s in sources) if sources else 0
        return (
            f"{self.LOW_RELEVANCE_RESPONSE}"
            f"(Highest relevance score: {max_score:.0%})\n\n"
        )

    def handle_ambiguous_query(self, query: str) -> str:
        """Generate clarification request for ambiguous queries.

        Args:
            query: Ambiguous query.

        Returns:
            Clarification request.
        """
        query_type = self.query_analyzer.analyze(query)

        clarifications = []
        if query_type == QueryType.AMBIGUOUS:
            # Generate potential clarifications
            words = query.lower().split()
            if len(words) >= 1:
                clarifications.append(f"- What is {words[0]}?")
                clarifications.append(f"- How does {words[0]} work?")
                clarifications.append(f"- Examples of {words[0]}?")

        if clarifications:
            return self.AMBIGUOUS_QUERY_RESPONSE + "\n".join(clarifications)
        return self.AMBIGUOUS_QUERY_RESPONSE

    def process(
        self, query: str, sources: list[SearchResult]
    ) -> tuple[bool, str | None]:
        """Process query and sources for edge cases.

        Args:
            query: User query.
            sources: Retrieved sources.

        Returns:
            Tuple of (proceed_with_generation, edge_case_response).
            If proceed is False, use the edge_case_response instead.
        """
        # Check query type
        query_type = self.query_analyzer.analyze(query)

        # Handle opinion queries
        if query_type == QueryType.OPINION:
            return True, None  # Proceed but LLM should note it can't give opinions

        # Handle ambiguous queries
        if query_type == QueryType.AMBIGUOUS and len(query.split()) < 3:
            return False, self.handle_ambiguous_query(query)

        # Check if we should answer based on sources
        should_answer, reason = self.should_answer(sources)

        if not should_answer:
            if reason == "no_results":
                return False, self.handle_no_results(query)
            elif reason == "low_relevance":
                # Proceed but with warning
                return True, self.handle_low_relevance(query, sources)
            else:
                return False, self.handle_no_results(query)

        return True, None
