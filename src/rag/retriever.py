from __future__ import annotations

import logging
import re
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

    def _matches_resume_name(self, candidate_value: Optional[str], expected_name: Optional[str]) -> bool:
        if not candidate_value or not expected_name:
            return False

        left = self._normalize_name(candidate_value)
        right = self._normalize_name(expected_name)
        if not left or not right:
            return False

        return (
            left == right
            or left.startswith(right)
            or right.startswith(left)
            or right in left
            or left in right
        )

    def _infer_resume_name(self, question: str, resume_name: Optional[str]) -> Optional[str]:
        if resume_name:
            return resume_name

        question_terms = [term for term in re.findall(r"[A-Za-z]+", question) if len(term) > 2]
        for metadata in self.vector_store.get_all_metadata():
            candidate_name = metadata.get("candidate_name")
            filename = metadata.get("filename")
            if self._matches_resume_name(candidate_name, question):
                return candidate_name
            if self._matches_resume_name(filename, question):
                return filename
            for term in question_terms:
                if self._matches_resume_name(candidate_name, term):
                    return candidate_name
                if self._matches_resume_name(filename, term):
                    return filename
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
            where_clause = {
                "$or": [
                    {"candidate_name": {"$eq": filter_name}},
                    {"filename": {"$eq": filter_name}},
                ]
            }

        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=max(top_k or self.top_k, 20),
            where=where_clause,
        )

        context_parts = []
        seen = set()

        if filter_name:
            filtered_results = [
                result
                for result in results
                if self._matches_resume_name((result.get("metadata") or {}).get("candidate_name"), filter_name)
                or self._matches_resume_name((result.get("metadata") or {}).get("filename"), filter_name)
            ]
        else:
            filtered_results = results

        for result in filtered_results[: top_k or self.top_k]:
            text = result.get("document")
            if isinstance(text, str):
                text = text.strip()
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