from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from .chunking import TextChunk

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - handled at runtime
    SentenceTransformer = None


@dataclass
class EmbeddedChunk:
    """Represents a text chunk with its generated embedding vector."""

    filename: str
    chunk_id: str
    chunk_text: str
    embedding_vector: list[float]


class EmbeddingGenerator:
    """Generate sentence embeddings for text chunks."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = None
        logger.info("Initialized EmbeddingGenerator with model: %s", self.model_name)

    def load_model(self):
        """Load the sentence-transformers model lazily."""
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required to generate embeddings.")

        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
            logger.info("Loaded embedding model: %s", self.model_name)

        return self.model

    def generate_embeddings(self, chunks: Sequence[TextChunk]) -> list[EmbeddedChunk]:
        """Generate embeddings for each provided text chunk."""
        if not chunks:
            return []

        model = self.load_model()
        chunk_texts = [chunk.chunk_text for chunk in chunks]

        try:
            embeddings = model.encode(chunk_texts, convert_to_numpy=False)
        except Exception as exc:  # pragma: no cover - logging branch
            logger.exception("Embedding generation failed: %s", exc)
            raise

        embedded_chunks: list[EmbeddedChunk] = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunks.append(
                EmbeddedChunk(
                    filename=chunk.filename,
                    chunk_id=chunk.chunk_id,
                    chunk_text=chunk.chunk_text,
                    embedding_vector=[float(value) for value in embedding],
                )
            )

        return embedded_chunks
