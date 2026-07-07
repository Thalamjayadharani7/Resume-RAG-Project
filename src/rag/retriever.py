from __future__ import annotations

import logging
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
        """Generate an embedding for a user question using the existing embedding model."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        model = self.embedding_generator.load_model()
        embeddings = model.encode([question], convert_to_numpy=False, normalize_embeddings=True)
        return [float(value) for value in embeddings[0]]

    def retrieve_context(self, question: str, top_k: Optional[int] = None, resume_name: Optional[str] = None) -> str:
        """Retrieve the most relevant chunks and join them into prompt-ready context."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        query_embedding = self._embed_query(question)
        where_clause = None
        if resume_name:
            where_clause = {"filename": resume_name}

        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k or self.top_k,
            where=where_clause,
        )

        context_parts: list[str] = []
        seen_context: set[str] = set()
        for result in results:
            document_text = result.get("document")
            if isinstance(document_text, str):
                cleaned = document_text.strip()
                if cleaned and cleaned not in seen_context:
                    context_parts.append(cleaned)
                    seen_context.add(cleaned)

        logger.info("Retrieved %d context chunk(s) for question '%s'", len(context_parts), question)
        return "\n\n".join(context_parts)

    def retrieve(self, question: str, top_k: Optional[int] = None, resume_name: Optional[str] = None) -> str:
        """Compatibility helper returning prompt-ready context for the question."""
        return self.retrieve_context(question=question, top_k=top_k, resume_name=resume_name)
