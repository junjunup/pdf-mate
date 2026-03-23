"""Vector storage module: manage document chunks in ChromaDB."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import chromadb
except ImportError:  # pragma: no cover
    chromadb = None  # type: ignore[assignment]

from .exceptions import StorageError

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk with metadata for retrieval."""

    text: str
    metadata: dict[str, Any]


class VectorStore:
    """ChromaDB-backed vector store for RAG retrieval.

    Examples:
        >>> store = VectorStore(collection_name="my_docs")
        >>> store.add([Chunk(text="hello", metadata={"source": "a.pdf"})])
        >>> results = store.query(query_text="hello", n_results=1)
    """

    def __init__(
        self,
        collection_name: str = "pdf_mate",
        persist_directory: str | None = None,
    ):
        """Initialize the vector store.

        Args:
            collection_name: ChromaDB collection name.
            persist_directory: Directory to persist the database (``None`` = in-memory).

        Raises:
            StorageError: If ChromaDB initialisation fails.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        if chromadb is None:
            raise StorageError(
                "chromadb is required for vector storage. "
                "Install it with: pip install pdf-mate[rag]"
            )

        try:
            if persist_directory:
                self._client = chromadb.PersistentClient(path=persist_directory)
            else:
                self._client = chromadb.EphemeralClient()
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise StorageError(f"Failed to initialise ChromaDB: {exc}") from exc

    @property
    def count(self) -> int:
        """Return the number of stored documents."""
        return self._collection.count()

    def add(
        self,
        chunks: list[Chunk],
        embeddings: list[np.ndarray] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """Add chunks to the vector store.

        Args:
            chunks: List of Chunk objects to store.
            embeddings: Pre-computed embeddings (optional, ChromaDB can embed).
            ids: Custom IDs for each chunk (auto-generated if ``None``).

        Raises:
            StorageError: If the add operation fails.
        """
        if not chunks:
            return

        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]

        if ids is None:
            start = self._collection.count()
            ids = [f"doc_{start + i}" for i in range(len(chunks))]

        embed_values = None
        if embeddings is not None:
            embed_values = [e.tolist() for e in embeddings]

        try:
            self._collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embed_values,
            )
        except Exception as exc:
            raise StorageError(f"Failed to add chunks: {exc}") from exc

    def query(
        self,
        query_embedding: np.ndarray | None = None,
        query_text: str | None = None,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Query the vector store for similar chunks.

        Args:
            query_embedding: Query vector.
            query_text: Query text (used if embedding not provided).
            n_results: Number of results to return.
            where: Metadata filter.

        Returns:
            List of (Chunk, distance_score) tuples, sorted by relevance.

        Raises:
            ValueError: If neither ``query_embedding`` nor ``query_text`` is provided.
            StorageError: If the query operation fails.
        """
        if query_embedding is None and query_text is None:
            raise ValueError("Either query_embedding or query_text must be provided")

        # Guard against querying an empty collection
        stored = self._collection.count()
        if stored == 0:
            return []

        # Clamp n_results to the number of stored documents
        effective_n = min(n_results, stored)

        kwargs: dict[str, Any] = {"n_results": effective_n}
        if where:
            kwargs["where"] = where
        if query_embedding is not None:
            kwargs["query_embeddings"] = [query_embedding.tolist()]
        else:
            kwargs["query_texts"] = [query_text]

        try:
            results = self._collection.query(**kwargs)
        except Exception as exc:
            raise StorageError(f"Query failed: {exc}") from exc

        chunks_with_scores: list[tuple[Chunk, float]] = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0
                chunks_with_scores.append(
                    (Chunk(text=doc, metadata=metadata), distance)
                )

        return chunks_with_scores

    def delete(self, source_name: str | None = None) -> None:
        """Delete all chunks from a specific source, or clear the entire collection.

        Args:
            source_name: Source document name to delete. If ``None``, clears all data.

        Raises:
            StorageError: If the delete operation fails.
        """
        try:
            if source_name:
                self._collection.delete(where={"source": source_name})
            else:
                self._client.delete_collection(self.collection_name)
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
        except Exception as exc:
            raise StorageError(f"Delete failed: {exc}") from exc

    def list_sources(self) -> list[str]:
        """List all unique document sources in the store.

        Returns:
            Sorted list of source names.
        """
        try:
            all_data = self._collection.get()
            sources: set[str] = set()
            if all_data and all_data["metadatas"]:
                for meta in all_data["metadatas"]:
                    if "source" in meta:
                        sources.add(meta["source"])
            return sorted(sources)
        except Exception:
            logger.warning("Failed to list sources", exc_info=True)
            return []
