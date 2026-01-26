"""Evaluation dataset management for RAG systems."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class EvaluationExample:
    """Single evaluation example with query and expected results."""

    query: str
    relevant_doc_ids: list[str]
    expected_answer: str = ""
    relevance_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "relevant_doc_ids": self.relevant_doc_ids,
            "expected_answer": self.expected_answer,
            "relevance_scores": self.relevance_scores,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationExample":
        """Create from dictionary."""
        return cls(
            query=data["query"],
            relevant_doc_ids=data.get("relevant_doc_ids", []),
            expected_answer=data.get("expected_answer", ""),
            relevance_scores=data.get("relevance_scores", {}),
            metadata=data.get("metadata", {}),
        )


class EvaluationDataset:
    """Dataset of evaluation examples for RAG system evaluation."""

    def __init__(self, name: str = "default") -> None:
        """Initialize the evaluation dataset.

        Args:
            name: Name of the dataset.
        """
        self.name = name
        self.examples: list[EvaluationExample] = []

    def add_example(
        self,
        query: str,
        relevant_doc_ids: list[str],
        expected_answer: str = "",
        relevance_scores: dict[str, float] | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Add an evaluation example.

        Args:
            query: Search query.
            relevant_doc_ids: List of relevant document IDs.
            expected_answer: Expected/ground truth answer.
            relevance_scores: Optional graded relevance scores.
            metadata: Optional metadata.
        """
        self.examples.append(
            EvaluationExample(
                query=query,
                relevant_doc_ids=relevant_doc_ids,
                expected_answer=expected_answer,
                relevance_scores=relevance_scores or {},
                metadata=metadata or {},
            )
        )

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)

    def __iter__(self) -> Iterator[EvaluationExample]:
        """Iterate over examples."""
        return iter(self.examples)

    def __getitem__(self, idx: int) -> EvaluationExample:
        """Get example by index."""
        return self.examples[idx]

    def get_queries(self) -> list[str]:
        """Get all queries."""
        return [ex.query for ex in self.examples]

    def get_relevant_docs(self) -> list[set[str]]:
        """Get all relevant doc sets."""
        return [set(ex.relevant_doc_ids) for ex in self.examples]

    def get_relevance_scores(self) -> list[dict[str, float]]:
        """Get all relevance score dicts."""
        return [ex.relevance_scores for ex in self.examples]

    def save(self, path: Path) -> None:
        """Save dataset to JSON file.

        Args:
            path: Path to save file.
        """
        path = Path(path)
        data = {
            "name": self.name,
            "examples": [ex.to_dict() for ex in self.examples],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "EvaluationDataset":
        """Load dataset from JSON file.

        Args:
            path: Path to load from.

        Returns:
            Loaded dataset.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        dataset = cls(name=data.get("name", "loaded"))
        for ex_data in data.get("examples", []):
            dataset.examples.append(EvaluationExample.from_dict(ex_data))

        return dataset

    @classmethod
    def create_sample_dataset(cls) -> "EvaluationDataset":
        """Create a sample evaluation dataset for testing.

        Returns:
            Sample dataset with ML-related queries.
        """
        dataset = cls(name="sample_ml_dataset")

        # Sample evaluation examples based on the sample documents
        dataset.add_example(
            query="What is machine learning?",
            relevant_doc_ids=["machine_learning_basics_0"],
            expected_answer="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            relevance_scores={"machine_learning_basics_0": 1.0},
            metadata={"category": "definition"},
        )

        dataset.add_example(
            query="What are the types of machine learning?",
            relevant_doc_ids=["machine_learning_basics_0", "machine_learning_basics_1"],
            expected_answer="The main types are supervised learning, unsupervised learning, and reinforcement learning.",
            relevance_scores={
                "machine_learning_basics_0": 1.0,
                "machine_learning_basics_1": 0.8,
            },
            metadata={"category": "types"},
        )

        dataset.add_example(
            query="What is a neural network?",
            relevant_doc_ids=["neural_networks_0"],
            expected_answer="Neural networks are computing systems inspired by biological neural networks in the human brain.",
            relevance_scores={"neural_networks_0": 1.0},
            metadata={"category": "definition"},
        )

        dataset.add_example(
            query="What is RAG?",
            relevant_doc_ids=["rag_systems_0"],
            expected_answer="RAG (Retrieval-Augmented Generation) is an AI framework that combines information retrieval with text generation.",
            relevance_scores={"rag_systems_0": 1.0},
            metadata={"category": "definition"},
        )

        dataset.add_example(
            query="How does backpropagation work?",
            relevant_doc_ids=["neural_networks_1"],
            expected_answer="Backpropagation calculates gradients of the loss function with respect to network weights using the chain rule of calculus.",
            relevance_scores={"neural_networks_1": 1.0},
            metadata={"category": "algorithm"},
        )

        dataset.add_example(
            query="What are the benefits of RAG over fine-tuning?",
            relevant_doc_ids=["rag_systems_0", "rag_systems_1"],
            expected_answer="RAG offers reduced hallucination, up-to-date information, source attribution, and domain adaptation without model fine-tuning.",
            relevance_scores={
                "rag_systems_0": 1.0,
                "rag_systems_1": 0.7,
            },
            metadata={"category": "comparison"},
        )

        dataset.add_example(
            query="What is overfitting?",
            relevant_doc_ids=["machine_learning_basics_0"],
            expected_answer="Overfitting occurs when a model learns the training data too well, including its noise, leading to poor generalization.",
            relevance_scores={"machine_learning_basics_0": 1.0},
            metadata={"category": "concept"},
        )

        dataset.add_example(
            query="What activation functions are used in neural networks?",
            relevant_doc_ids=["neural_networks_0"],
            expected_answer="Common activation functions include ReLU, sigmoid, tanh, and softmax.",
            relevance_scores={"neural_networks_0": 1.0},
            metadata={"category": "technique"},
        )

        return dataset
