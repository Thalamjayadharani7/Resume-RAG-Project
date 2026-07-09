from __future__ import annotations

import logging
import re
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
        self.create_collection()

    def _initialize_client(self) -> None:
        """Create the ChromaDB client once for the lifetime of this VectorStore instance."""
        if self.client is not None:
            return

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

    def create_collection(self) -> Any:
        """Create or retrieve the ChromaDB collection without recreating the client."""
        if self.collection is not None:
            return self.collection

        self._initialize_client()

        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:  # pragma: no cover - logging branch
            logger.exception("Failed to create collection '%s': %s", self.collection_name, exc)
            raise

        return self.collection

    def reset_collection(self) -> Any:
        """Delete any existing collection state and recreate an empty collection."""
        self._initialize_client()

        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            logger.debug("Collection '%s' did not exist before reset", self.collection_name)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self.collection

    def get_all_metadata(self) -> list[dict[str, Any]]:
        """Return all collection metadata entries."""
        self.create_collection()

        result = self.collection.get(include=["metadatas"])
        metadatas = result.get("metadatas", [])
        return [metadata if isinstance(metadata, dict) else {} for metadata in metadatas]

    @staticmethod
    def _infer_candidate_name_from_filename(filename: str) -> str:
        """Infer a candidate name from the resume filename when metadata is missing."""
        stem = Path(filename).stem
        stem = re.sub(
            r"(?i)\b(resume|profile|cv|sample|b\.?tech|btech)\b",
            "",
            stem,
        )
        stem = re.sub(r"\(.*?\)", " ", stem)
        stem = re.sub(r"\d+", " ", stem)
        stem = re.sub(r"[_\-.]+", " ", stem)
        stem = re.sub(r"\s+", " ", stem).strip()

        words = [
            word.title()
            for word in stem.split()
            if len(word) > 1 and word.lower() not in {"resume", "profile", "sample", "b", "tech"}
        ]
        if 2 <= len(words) <= 4:
            return " ".join(words)
        return ""

    def add_documents(self, documents: Sequence[EmbeddedChunk]) -> list[str]:
        """Add embedded chunks to the ChromaDB collection."""
        if not documents:
            return []

        if self.collection is None:
            self.create_collection()

        ids = [f"{document.filename}:{document.chunk_id}" for document in documents]
        metadatas = []
        for document in documents:
            candidate_name = (document.candidate_name or "").strip()
            if not candidate_name:
                candidate_name = self._infer_candidate_name_from_filename(document.filename)
            metadatas.append(
                {
                    "filename": document.filename,
                    "chunk_id": document.chunk_id,
                    "candidate_name": candidate_name,
                    "page_number": document.page_number if document.page_number is not None else 1,
                }
            )

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

    def similarity_search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Find the most similar stored chunks for a query embedding."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        self.create_collection()

        query_vector = [float(value) for value in query_embedding]
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        try:
            results = self.collection.query(**query_kwargs)
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
