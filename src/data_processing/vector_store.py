from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

logger = logging.getLogger(__name__)

try:
    import chromadb
except ImportError:  # pragma: no cover - handled at runtime
    chromadb = None

from .embedding import EmbeddedChunk


class VectorStore:
    """Store and query chunk embeddings in ChromaDB."""

    def __init__(
        self,
        collection_name: str = "resume_chunks",
        persist_directory: Optional[Union[str, Path]] = None,
    ) -> None:
        default_dir = Path(__file__).resolve().parents[2] / ".chromadb"
        self.persist_directory = Path(persist_directory or default_dir).resolve()
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        logger.info(
            "Initialized VectorStore for collection '%s' at %s",
            self.collection_name,
            self.persist_directory,
        )

    def create_collection(self) -> Any:
        """Create or retrieve the ChromaDB collection."""
        if chromadb is None:
            raise ImportError("chromadb is required to store embeddings.")

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        try:
            if hasattr(chromadb, "PersistentClient"):
                self.client = chromadb.PersistentClient(path=str(self.persist_directory))
            else:
                self.client = chromadb.Client()
        except Exception as exc:  # pragma: no cover - logging branch
            logger.exception("Failed to initialize ChromaDB client: %s", exc)
            raise

        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:  # pragma: no cover - logging branch
            logger.exception("Failed to create collection '%s': %s", self.collection_name, exc)
            raise

        return self.collection

    def add_documents(self, documents: Sequence[EmbeddedChunk]) -> list[str]:
        """Add embedded chunks to the ChromaDB collection."""
        if not documents:
            return []

        if self.collection is None:
            self.create_collection()

        ids = [f"{document.filename}:{document.chunk_id}" for document in documents]
        metadatas = [
            {"filename": document.filename, "chunk_id": document.chunk_id}
            for document in documents
        ]

        try:
            self.collection.add(
                ids=ids,
                documents=[document.chunk_text for document in documents],
                embeddings=[document.embedding_vector for document in documents],
                metadatas=metadatas,
            )
            logger.info("Added %d documents to collection '%s'", len(documents), self.collection_name)
        except Exception as exc:  # pragma: no cover - logging branch
            logger.exception("Failed to add documents to collection '%s': %s", self.collection_name, exc)
            raise

        return ids

    def similarity_search(self, query_embedding: Sequence[float], top_k: int = 5) -> list[dict[str, Any]]:
        """Find the most similar stored chunks for a query embedding."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if self.collection is None:
            self.create_collection()

        query_vector = [float(value) for value in query_embedding]
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # pragma: no cover - logging branch
            logger.exception("Similarity search failed: %s", exc)
            raise

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        return [
            {
                "id": id_value,
                "document": document,
                "metadata": metadata,
                "distance": distance,
            }
            for id_value, document, metadata, distance in zip(ids, documents, metadatas, distances)
        ]
