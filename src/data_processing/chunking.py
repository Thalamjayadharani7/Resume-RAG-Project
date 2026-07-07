from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

from .pdf_loader import PDFDocument

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a chunk of text extracted from a PDF document."""

    filename: str
    chunk_id: str
    chunk_text: str


class TextChunker:
    """Split document text into smaller, overlapping chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(
            "Initialized TextChunker with chunk_size=%s and chunk_overlap=%s",
            self.chunk_size,
            self.chunk_overlap,
        )

    def chunk_documents(self, documents: Sequence[PDFDocument]) -> list[TextChunk]:
        """Split a sequence of PDF documents into chunk objects."""
        if not documents:
            return []

        chunks: list[TextChunk] = []
        for document in documents:
            if not document.text.strip():
                logger.warning("Skipping empty document: %s", document.filename)
                continue

            for index, chunk_text in enumerate(self._deduplicate_chunks(self._split_text(document.text))):
                chunk_id = f"{document.filename}__{index}"
                chunks.append(
                    TextChunk(
                        filename=document.filename,
                        chunk_id=chunk_id,
                        chunk_text=chunk_text,
                    )
                )

        return chunks

    def _split_text(self, text: str) -> list[str]:
        normalized_text = re.sub(r"\s+", " ", text).strip()
        if not normalized_text:
            return []

        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", normalized_text) if sentence.strip()]
        if not sentences:
            sentences = [normalized_text]

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            if current_chunk and current_length + 1 + sentence_length > self.chunk_size:
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk).strip())

        final_chunks: list[str] = []
        for chunk in chunks:
            final_chunks.extend(self._split_large_chunk(chunk))

        return final_chunks

    def _split_large_chunk(self, chunk: str) -> list[str]:
        if len(chunk) <= self.chunk_size:
            return [chunk]

        split_chunks: list[str] = []
        start = 0
        while start < len(chunk):
            end = min(start + self.chunk_size, len(chunk))
            if end < len(chunk):
                last_space = chunk.rfind(" ", start, end)
                if last_space != -1:
                    end = last_space

            piece = chunk[start:end].strip()
            if not piece:
                break

            split_chunks.append(piece)
            if end >= len(chunk):
                break

            start = max(start + self.chunk_size - self.chunk_overlap, end)

        return split_chunks

    def _deduplicate_chunks(self, chunks: Sequence[str]) -> list[str]:
        """Remove repeated chunks while preserving order."""
        unique_chunks: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            normalized_chunk = chunk.strip()
            if not normalized_chunk:
                continue
            if normalized_chunk in seen:
                continue
            seen.add(normalized_chunk)
            unique_chunks.append(normalized_chunk)
        return unique_chunks
