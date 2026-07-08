from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .chunking import TextChunk

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - handled at runtime
    SentenceTransformer = None


class EmbeddingGenerationError(RuntimeError):
    """Raised when embedding generation fails."""


@dataclass
class EmbeddedChunk:
    """Represents a text chunk with its generated embedding vector."""

    filename: str
    chunk_id: str
    chunk_text: str
    embedding_vector: list[float]
    candidate_name: str | None = None
    page_number: int | None = None


class EmbeddingGenerator:
    """Generate sentence embeddings for text chunks."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 16) -> None:
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self.model = None
        self.cache_dir = Path(os.getenv("HF_HOME", "./.cache/huggingface")).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initialized EmbeddingGenerator with model: %s", self.model_name)

    def load_model(self):
        """Load the sentence-transformers model lazily and reuse it."""
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required to generate embeddings.")

        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name, cache_folder=str(self.cache_dir))
            except Exception as exc:  # pragma: no cover - runtime path
                logger.exception("Failed to load embedding model '%s'", self.model_name)
                raise EmbeddingGenerationError(f"Unable to load embedding model: {exc}") from exc
            logger.info("Loaded embedding model: %s", self.model_name)

        return self.model

    def generate_embeddings(self, chunks: Sequence[TextChunk]) -> list[EmbeddedChunk]:
        """Generate embeddings for each provided text chunk in batches."""
        if not chunks:
            return []

        model = self.load_model()
        chunk_texts = [chunk.chunk_text for chunk in chunks]
        embedded_chunks: list[EmbeddedChunk] = []

        try:
            for start in range(0, len(chunk_texts), self.batch_size):
                batch_texts = chunk_texts[start : start + self.batch_size]
                batch_chunks = chunks[start : start + self.batch_size]
                embeddings = model.encode(batch_texts, convert_to_numpy=False, normalize_embeddings=True)
                for chunk, embedding in zip(batch_chunks, embeddings):
                    embedded_chunks.append(
                        EmbeddedChunk(
                            filename=chunk.filename,
                            chunk_id=chunk.chunk_id,
                            chunk_text=chunk.chunk_text,
                            embedding_vector=[float(value) for value in embedding],
                            candidate_name=chunk.candidate_name,
                            page_number=chunk.page_number,
                        )
                    )
        except Exception as exc:  # pragma: no cover - logging branch
            logger.exception("Embedding generation failed: %s", exc)
            raise EmbeddingGenerationError(f"Embedding generation failed: {exc}") from exc

        return embedded_chunks
