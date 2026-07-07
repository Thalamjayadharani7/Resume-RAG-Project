from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Raised when evaluation input data cannot be processed."""


class ResumeRAGEvaluator:
    """Evaluate a resume retrieval augmented generation pipeline."""

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        column_mapping: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize the evaluator with a dataset path and optional column names."""
        self.dataset_path = Path(dataset_path or "ground_truth/evaluation_dataset.csv")
        self.column_mapping = dict(column_mapping or {})

    def evaluate(self, dataset_path: str | Path | None = None) -> dict[str, Any]:
        """Evaluate the dataset and return a summary report."""
        path = Path(dataset_path or self.dataset_path)
        rows = self._load_dataset(path)

        if not rows:
            logger.warning("No evaluation rows were loaded from %s", path)
            return self._empty_report(path)

        retrieval_scores: list[float] = []
        faithfulness_scores: list[float] = []
        context_precision_scores: list[float] = []
        context_recall_scores: list[float] = []
        answer_correctness_scores: list[float] = []

        for row in rows:
            retrieved_context = self._get_value(row, "retrieved_context", "retrieved")
            gold_context = self._get_value(row, "gold_context", "context", "expected_context")
            user_question = self._get_value(row, "question", "user_question")
            answer = self._get_value(row, "answer", "generated_answer")
            gold_answer = self._get_value(row, "gold_answer", "expected_answer", "ground_truth")

            retrieval_scores.append(self._retrieval_accuracy(retrieved_context, gold_context))
            faithfulness_scores.append(self._faithfulness(retrieved_context, answer))
            context_precision_scores.append(self._context_precision(retrieved_context, gold_context))
            context_recall_scores.append(self._context_recall(retrieved_context, gold_context))
            answer_correctness_scores.append(self._answer_correctness(answer, gold_answer, user_question))

        return {
            "dataset_path": str(path),
            "rows_evaluated": len(rows),
            "retrieval_accuracy": self._round_metric(sum(retrieval_scores) / len(retrieval_scores)),
            "faithfulness": self._round_metric(sum(faithfulness_scores) / len(faithfulness_scores)),
            "context_precision": self._round_metric(sum(context_precision_scores) / len(context_precision_scores)),
            "context_recall": self._round_metric(sum(context_recall_scores) / len(context_recall_scores)),
            "answer_correctness": self._round_metric(sum(answer_correctness_scores) / len(answer_correctness_scores)),
        }

    def _load_dataset(self, path: Path) -> list[dict[str, str]]:
        """Load rows from a CSV file and return them as dictionaries."""
        if not path.exists():
            logger.warning("Evaluation dataset was not found at %s", path)
            return []

        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    logger.warning("Evaluation dataset %s is empty or missing headers.", path)
                    return []
                return [
                    {key: (value or "") for key, value in row.items() if key is not None}
                    for row in reader
                ]
        except (FileNotFoundError, PermissionError, csv.Error, UnicodeDecodeError) as exc:
            logger.exception("Failed to read evaluation dataset %s", path)
            raise EvaluationError(f"Unable to read evaluation dataset: {exc}") from exc

    def _empty_report(self, path: Path) -> dict[str, Any]:
        """Return a zero-filled report when no valid dataset can be used."""
        return {
            "dataset_path": str(path),
            "rows_evaluated": 0,
            "retrieval_accuracy": 0.0,
            "faithfulness": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "answer_correctness": 0.0,
        }

    def _get_value(self, row: Mapping[str, str], *candidates: str) -> str:
        """Retrieve a field value using configured column names or defaults."""
        for candidate in candidates:
            mapped_name = self.column_mapping.get(candidate, candidate)
            if mapped_name in row and row[mapped_name]:
                return str(row[mapped_name]).strip()
        return ""

    def _normalize_text(self, value: str) -> str:
        """Normalize whitespace and casing for text comparison."""
        return re.sub(r"\s+", " ", value).strip().lower()

    def _tokenize(self, value: str) -> list[str]:
        """Split text into normalized tokens."""
        normalized = self._normalize_text(value)
        return [token for token in re.split(r"\W+", normalized) if token]

    def _f1_score(self, candidate: str, reference: str) -> float:
        """Compute a token-based F1 score between two texts."""
        if not candidate and not reference:
            return 1.0
        if not candidate or not reference:
            return 0.0

        candidate_tokens = self._tokenize(candidate)
        reference_tokens = self._tokenize(reference)
        if not candidate_tokens or not reference_tokens:
            return 0.0

        overlap = len(set(candidate_tokens) & set(reference_tokens))
        precision = overlap / len(candidate_tokens)
        recall = overlap / len(reference_tokens)
        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    def _retrieval_accuracy(self, retrieved_context: str, gold_context: str) -> float:
        """Measure how well retrieved context matches the gold context."""
        return self._f1_score(retrieved_context, gold_context)

    def _faithfulness(self, retrieved_context: str, answer: str) -> float:
        """Measure whether the answer is supported by the retrieved context."""
        if not answer:
            return 0.0
        if not retrieved_context:
            return 0.0

        answer_tokens = set(self._tokenize(answer))
        context_tokens = set(self._tokenize(retrieved_context))
        if not answer_tokens:
            return 0.0
        return len(answer_tokens & context_tokens) / len(answer_tokens)

    def _context_precision(self, retrieved_context: str, gold_context: str) -> float:
        """Measure precision of the retrieved context against the gold context."""
        if not retrieved_context:
            return 0.0
        if not gold_context:
            return 0.0

        retrieved_tokens = set(self._tokenize(retrieved_context))
        gold_tokens = set(self._tokenize(gold_context))
        if not retrieved_tokens:
            return 0.0
        return len(retrieved_tokens & gold_tokens) / len(retrieved_tokens)

    def _context_recall(self, retrieved_context: str, gold_context: str) -> float:
        """Measure recall of the retrieved context against the gold context."""
        if not gold_context:
            return 0.0
        if not retrieved_context:
            return 0.0

        retrieved_tokens = set(self._tokenize(retrieved_context))
        gold_tokens = set(self._tokenize(gold_context))
        if not gold_tokens:
            return 0.0
        return len(retrieved_tokens & gold_tokens) / len(gold_tokens)

    def _answer_correctness(self, answer: str, gold_answer: str, question: str) -> float:
        """Measure answer correctness using token overlap and question context."""
        if not answer and not gold_answer:
            return 1.0
        if not answer or not gold_answer:
            return 0.0

        answer_score = self._f1_score(answer, gold_answer)
        if not question:
            return answer_score
        question_terms = set(self._tokenize(question))
        answer_terms = set(self._tokenize(answer))
        if not question_terms or not answer_terms:
            return answer_score
        return (answer_score + (len(answer_terms & question_terms) / len(question_terms))) / 2

    def _round_metric(self, value: float) -> float:
        """Round metric values for stable reporting."""
        return round(float(value), 4)


def evaluate_resume_rag(
    dataset_path: str | Path | None = None,
    column_mapping: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Convenience function for evaluating a resume RAG dataset."""
    evaluator = ResumeRAGEvaluator(dataset_path=dataset_path, column_mapping=column_mapping)
    return evaluator.evaluate()
