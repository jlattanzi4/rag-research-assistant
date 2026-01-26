"""Retrieval and generation quality metrics."""

import math
from dataclasses import dataclass, field
from typing import Callable


def calculate_precision_at_k(
    retrieved: list[str], relevant: set[str], k: int
) -> float:
    """Calculate Precision@K.

    Precision@K = (# of relevant items in top-k) / k

    Args:
        retrieved: List of retrieved document IDs in order.
        relevant: Set of relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Precision@K score (0-1).
    """
    if k <= 0:
        return 0.0

    top_k = retrieved[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant)
    return relevant_in_top_k / k


def calculate_recall_at_k(
    retrieved: list[str], relevant: set[str], k: int
) -> float:
    """Calculate Recall@K.

    Recall@K = (# of relevant items in top-k) / (total # of relevant items)

    Args:
        retrieved: List of retrieved document IDs in order.
        relevant: Set of relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Recall@K score (0-1).
    """
    if not relevant:
        return 0.0

    top_k = retrieved[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant)
    return relevant_in_top_k / len(relevant)


def calculate_mrr(
    retrieved: list[str], relevant: set[str]
) -> float:
    """Calculate Mean Reciprocal Rank (MRR).

    MRR = 1 / rank of first relevant result

    Args:
        retrieved: List of retrieved document IDs in order.
        relevant: Set of relevant document IDs.

    Returns:
        MRR score (0-1).
    """
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def calculate_ndcg(
    retrieved: list[str],
    relevance_scores: dict[str, float],
    k: int,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K).

    Args:
        retrieved: List of retrieved document IDs in order.
        relevance_scores: Mapping of document ID to relevance score.
        k: Number of top results to consider.

    Returns:
        NDCG@K score (0-1).
    """
    if k <= 0 or not relevance_scores:
        return 0.0

    # Calculate DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], 1):
        rel = relevance_scores.get(doc_id, 0.0)
        dcg += (2 ** rel - 1) / math.log2(i + 1)

    # Calculate ideal DCG
    sorted_relevance = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(sorted_relevance, 1):
        idcg += (2 ** rel - 1) / math.log2(i + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_hit_rate(
    retrieved: list[str], relevant: set[str], k: int
) -> float:
    """Calculate Hit Rate@K (whether any relevant doc is in top-k).

    Args:
        retrieved: List of retrieved document IDs in order.
        relevant: Set of relevant document IDs.
        k: Number of top results to consider.

    Returns:
        1.0 if any relevant doc in top-k, else 0.0.
    """
    top_k = set(retrieved[:k])
    return 1.0 if top_k & relevant else 0.0


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""

    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    hit_rate_at_1: float = 0.0
    hit_rate_at_5: float = 0.0
    num_queries: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "precision@1": self.precision_at_1,
            "precision@3": self.precision_at_3,
            "precision@5": self.precision_at_5,
            "precision@10": self.precision_at_10,
            "recall@1": self.recall_at_1,
            "recall@3": self.recall_at_3,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "mrr": self.mrr,
            "ndcg@5": self.ndcg_at_5,
            "ndcg@10": self.ndcg_at_10,
            "hit_rate@1": self.hit_rate_at_1,
            "hit_rate@5": self.hit_rate_at_5,
            "num_queries": self.num_queries,
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"RetrievalMetrics(MRR={self.mrr:.3f}, "
            f"P@5={self.precision_at_5:.3f}, "
            f"R@5={self.recall_at_5:.3f}, "
            f"NDCG@5={self.ndcg_at_5:.3f})"
        )


class RetrievalEvaluator:
    """Evaluator for retrieval system performance."""

    def __init__(self, retrieval_fn: Callable[[str], list[str]]) -> None:
        """Initialize the evaluator.

        Args:
            retrieval_fn: Function that takes a query and returns list of doc IDs.
        """
        self.retrieval_fn = retrieval_fn

    def evaluate_single(
        self,
        query: str,
        relevant_docs: set[str],
        relevance_scores: dict[str, float] | None = None,
    ) -> RetrievalMetrics:
        """Evaluate a single query.

        Args:
            query: Search query.
            relevant_docs: Set of relevant document IDs.
            relevance_scores: Optional graded relevance scores.

        Returns:
            Metrics for this query.
        """
        retrieved = self.retrieval_fn(query)

        # Use binary relevance if no scores provided
        if relevance_scores is None:
            relevance_scores = {doc_id: 1.0 for doc_id in relevant_docs}

        return RetrievalMetrics(
            precision_at_1=calculate_precision_at_k(retrieved, relevant_docs, 1),
            precision_at_3=calculate_precision_at_k(retrieved, relevant_docs, 3),
            precision_at_5=calculate_precision_at_k(retrieved, relevant_docs, 5),
            precision_at_10=calculate_precision_at_k(retrieved, relevant_docs, 10),
            recall_at_1=calculate_recall_at_k(retrieved, relevant_docs, 1),
            recall_at_3=calculate_recall_at_k(retrieved, relevant_docs, 3),
            recall_at_5=calculate_recall_at_k(retrieved, relevant_docs, 5),
            recall_at_10=calculate_recall_at_k(retrieved, relevant_docs, 10),
            mrr=calculate_mrr(retrieved, relevant_docs),
            ndcg_at_5=calculate_ndcg(retrieved, relevance_scores, 5),
            ndcg_at_10=calculate_ndcg(retrieved, relevance_scores, 10),
            hit_rate_at_1=calculate_hit_rate(retrieved, relevant_docs, 1),
            hit_rate_at_5=calculate_hit_rate(retrieved, relevant_docs, 5),
            num_queries=1,
        )

    def evaluate_dataset(
        self,
        queries: list[str],
        relevant_docs_list: list[set[str]],
        relevance_scores_list: list[dict[str, float]] | None = None,
    ) -> RetrievalMetrics:
        """Evaluate on a dataset of queries.

        Args:
            queries: List of search queries.
            relevant_docs_list: List of relevant doc sets for each query.
            relevance_scores_list: Optional list of relevance score dicts.

        Returns:
            Aggregated metrics over all queries.
        """
        if relevance_scores_list is None:
            relevance_scores_list = [None] * len(queries)

        # Accumulate metrics
        total = RetrievalMetrics(num_queries=len(queries))

        for query, relevant_docs, relevance_scores in zip(
            queries, relevant_docs_list, relevance_scores_list
        ):
            metrics = self.evaluate_single(query, relevant_docs, relevance_scores)

            total.precision_at_1 += metrics.precision_at_1
            total.precision_at_3 += metrics.precision_at_3
            total.precision_at_5 += metrics.precision_at_5
            total.precision_at_10 += metrics.precision_at_10
            total.recall_at_1 += metrics.recall_at_1
            total.recall_at_3 += metrics.recall_at_3
            total.recall_at_5 += metrics.recall_at_5
            total.recall_at_10 += metrics.recall_at_10
            total.mrr += metrics.mrr
            total.ndcg_at_5 += metrics.ndcg_at_5
            total.ndcg_at_10 += metrics.ndcg_at_10
            total.hit_rate_at_1 += metrics.hit_rate_at_1
            total.hit_rate_at_5 += metrics.hit_rate_at_5

        # Average
        n = len(queries)
        if n > 0:
            total.precision_at_1 /= n
            total.precision_at_3 /= n
            total.precision_at_5 /= n
            total.precision_at_10 /= n
            total.recall_at_1 /= n
            total.recall_at_3 /= n
            total.recall_at_5 /= n
            total.recall_at_10 /= n
            total.mrr /= n
            total.ndcg_at_5 /= n
            total.ndcg_at_10 /= n
            total.hit_rate_at_1 /= n
            total.hit_rate_at_5 /= n

        return total
