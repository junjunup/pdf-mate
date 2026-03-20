"""Vector storage module: manage document chunks in ChromaDB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

import chromadb


@dataclass
class Chunk:
    """A text chunk with metadata for retrieval."""

    text: str
    metadata: dict[str, Any]


class VectorStore:
    """ChromaDB-backed vector store for RAG retrieval."""

    def __init__(
        self,
        collection_name: str = "pdf_mate",
        persist_directory: Optional[str] = None,
    ):
        """Initialize the vector store.

        Args:
            collection_name: ChromaDB collection name.
            persist_directory: Directory to persist the database (None = in-memory).
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        client_kwargs: dict[str, Any] = {}
        if persist_directory:
            client_kwargs["path"] = persist_directory

        self._client = chromadb.Client(client_kwargs)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        """Return the number of stored documents."""
        return self._collection.count()

    def add(
        self,
        chunks: list[Chunk],
        embeddings: Optional[list[np.ndarray]] = None,
        ids: Optional[list[str]] = None,
    ) -> None:
        """Add chunks to the vector store.

        Args:
            chunks: List of Chunk objects to store.
            embeddings: Pre-computed embeddings (optional, ChromaDB can embed).
            ids: Custom IDs for each chunk (auto-generated if None).
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

        self._collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embed_values,
        )

    def query(
        self,
        query_embedding: Optional[np.ndarray] = None,
        query_text: Optional[str] = None,
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> list[tuple[Chunk, float]]:
        """Query the vector store for similar chunks.

        Args:
            query_embedding: Query vector.
            query_text: Query text (used if embedding not provided).
            n_results: Number of results to return.
            where: Metadata filter.

        Returns:
            List of (Chunk, distance_score) tuples, sorted by relevance.
        """
        kwargs: dict[str, Any] = {"n_results": n_results}
        if where:
            kwargs["where"] = where
        if query_embedding is not None:
            kwargs["query_embeddings"] = [query_embedding.tolist()]
        elif query_text is not None:
            kwargs["query_texts"] = [query_text]
        else:
            raise ValueError("Either query_embedding or query_text must be provided")

        results = self._collection.query(**kwargs)

        chunks_with_scores: list[tuple[Chunk, float]] = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0
                chunks_with_scores.append(
                    (Chunk(text=doc, metadata=metadata), distance)
                )

        return chunks_with_scores

    def delete(self, source_name: Optional[str] = None) -> None:
        """Delete all chunks from a specific source.

        Args:
            source_name: Source document name to delete.
        """
        if source_name:
            try:
                self._collection.delete(where={"source": source_name})
            except Exception:
                pass
        else:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def list_sources(self) -> list[str]:
        """List all document sources in the store."""
        try:
            all_data = self._collection.get()
            sources = set()
            if all_data and all_data["metadatas"]:
                for meta in all_data["metadatas"]:
                    if "source" in meta:
                        sources.add(meta["source"])
            return sorted(sources)
        except Exception:
            return []
