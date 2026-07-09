from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Optional, Sequence

from src.data_processing.chunking import TextChunk
from src.data_processing.embedding import EmbeddingGenerator
from src.data_processing.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Embed queries and retrieve relevant context from the vector store."""

    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_store: Optional[VectorStore] = None,
        top_k: int = 5,
    ) -> None:
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.vector_store = vector_store or VectorStore()
        self.top_k = max(1, top_k)

    def index_documents(self, chunks: Sequence[TextChunk]) -> list[str]:
        """Generate embeddings for chunks and store them in the vector database."""
        if not chunks:
            return []

        embedded_chunks = self.embedding_generator.generate_embeddings(chunks)
        return self.vector_store.add_documents(embedded_chunks)

    def _embed_query(self, question: str) -> list[float]:
        """Generate an embedding for a user question."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        model = self.embedding_generator.load_model()
        embeddings = model.encode(
            [question],
            convert_to_numpy=False,
            normalize_embeddings=True,
        )
        return [float(value) for value in embeddings[0]]

    def _normalize_name(self, value: Optional[str]) -> str:
        if not value:
            return ""
        return re.sub(r"[^a-z0-9]+", "", str(value).lower())

    def _name_score(self, candidate_value: Optional[str], expected_name: Optional[str]) -> float:
        if not candidate_value or not expected_name:
            return 0.0

        left = self._normalize_name(candidate_value)
        right = self._normalize_name(expected_name)
        if not left or not right:
            return 0.0

        if left == right:
            return 1.0
        if left.startswith(right) or right.startswith(left):
            return 0.95
        if right in left or left in right:
            return 0.9

        sequence_score = SequenceMatcher(None, left, right).ratio()
        left_tokens = set(left)
        right_tokens = set(right)
        if left_tokens and right_tokens:
            overlap = len(left_tokens & right_tokens)
            token_score = overlap / max(1, len(right_tokens))
        else:
            token_score = 0.0

        return max(token_score, sequence_score)

    def _matches_resume_name(self, candidate_value: Optional[str], expected_name: Optional[str]) -> bool:
        return self._name_score(candidate_value, expected_name) >= 0.55

    def _extract_question_terms(self, question: str) -> list[str]:
        stop_words = {
            "what",
            "which",
            "where",
            "when",
            "why",
            "how",
            "tell",
            "me",
            "about",
            "the",
            "a",
            "an",
            "are",
            "is",
            "was",
            "were",
            "do",
            "did",
            "does",
            "can",
            "could",
            "skills",
            "skill",
            "project",
            "projects",
            "work",
            "worked",
            "working",
            "resume",
            "candidate",
            "cv",
            "experience",
            "education",
            "contact",
            "details",
            "information",
            "give",
            "show",
            "list",
            "for",
            "with",
            "and",
            "of",
            "on",
            "in",
            "my",
            "your",
            "his",
            "her",
            "their",
            "this",
            "that",
            "these",
            "those",
        }
        return [
            term.lower()
            for term in re.findall(r"[A-Za-z]+", question)
            if len(term) > 2 and term.lower() not in stop_words
        ]

    def _find_best_metadata_match(self, query_value: str, metadata_items: Sequence[dict[str, object]]) -> Optional[str]:
        best_name: Optional[str] = None
        best_score = 0.0
        for metadata in metadata_items:
            for candidate_value in (metadata.get("candidate_name"), metadata.get("filename")):
                if not candidate_value:
                    continue
                score = self._name_score(query_value.lower(), str(candidate_value).lower())
                if score > best_score:
                    best_score = score
                    best_name = str(candidate_value)
        if best_score >= 0.55:
            return best_name
        return None

    def _infer_resume_name(self, question: str, resume_name: Optional[str]) -> Optional[str]:
        metadata_items = list(self.vector_store.get_all_metadata())
        if resume_name:
            resolved_name = self._find_best_metadata_match(str(resume_name), metadata_items)
            if resolved_name:
                return resolved_name
            return str(resume_name)

        question_terms = self._extract_question_terms(question)
        for term in question_terms:
            resolved_name = self._find_best_metadata_match(term, metadata_items)
            if resolved_name:
                return resolved_name
        return None
     
    def retrieve_context(
        self,
        question: str,
        top_k: Optional[int] = None,
        resume_name: Optional[str] = None,
    ) -> str:
        """Retrieve the most relevant chunks."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        query_embedding = self._embed_query(question)

        filter_name = self._infer_resume_name(question, resume_name)

        where_clause = None
        if filter_name:
            where_clause = {"candidate_name": filter_name}

        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=20,
            where=where_clause,
        )

        if filter_name:
            results = [
                r for r in results
                if self._matches_resume_name(
                    (r.get("metadata") or {}).get("candidate_name"),
                    filter_name,
                )
                or self._matches_resume_name(
                    (r.get("metadata") or {}).get("filename"),
                    filter_name,
                )
            ]

        # Sort chunks in resume order
        results.sort(
            key=lambda r: (
                (r.get("metadata") or {}).get("page_number", 1),
                (r.get("metadata") or {}).get("chunk_id", ""),
            )
        )

        seen = set()
        context_parts = []

        for result in results:
            text = result.get("document", "").strip()
            if text and text not in seen:
                context_parts.append(text)
                seen.add(text)

        logger.info(
            "Retrieved %d chunk(s) for question '%s'",
            len(context_parts),
            question,
        )

        return "\n\n".join(context_parts)

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        resume_name: Optional[str] = None,
    ) -> str:
        return self.retrieve_context(
            question=question,
            top_k=top_k,
            resume_name=resume_name,
        )


def retrieve_relevant_text(question: str, documents: Sequence[str]) -> str:
    """
    Compatibility helper for uploaded-document workflow.
    """

    if not documents:
        return ""

    question_terms = [
        word
        for word in re.findall(r"\w+", question.lower())
        if len(word) > 2
    ]

    best_document = documents[0]
    best_score = -1

    for document in documents:
        score = 0
        lower = document.lower()

        for term in question_terms:
            score += lower.count(term)

        if score > best_score:
            best_score = score
            best_document = document

    if best_score <= 0:
        return best_document

    lines = [
        line.strip()
        for line in best_document.splitlines()
        if line.strip()
    ]

    relevant_lines = [
        line
        for line in lines
        if any(term in line.lower() for term in question_terms)
    ]

    if relevant_lines:
        return "\n".join(relevant_lines[:5])

    return best_document