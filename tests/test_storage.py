"""Tests for Storage module."""

from __future__ import annotations

import pytest

from pdf_mate.storage import Chunk, VectorStore


class TestChunk:
    def test_creation(self):
        chunk = Chunk(text="Hello world", metadata={"source": "test.pdf"})
        assert chunk.text == "Hello world"
        assert chunk.metadata["source"] == "test.pdf"


class TestVectorStore:
    def test_create_and_count(self):
        store = VectorStore(collection_name="test_store")
        assert store.count == 0

    def test_add_and_query(self):
        store = VectorStore(collection_name="test_store_query")
        chunks = [
            Chunk(text="Python is a programming language", metadata={"source": "doc1.pdf", "page": 1}),
            Chunk(text="Machine learning is a subset of AI", metadata={"source": "doc1.pdf", "page": 2}),
            Chunk(text="RAG combines retrieval and generation", metadata={"source": "doc2.pdf", "page": 1}),
        ]
        store.add(chunks=chunks)
        assert store.count == 3

    def test_list_sources(self):
        store = VectorStore(collection_name="test_store_sources")
        chunks = [
            Chunk(text="Content A", metadata={"source": "file_a.pdf"}),
            Chunk(text="Content B", metadata={"source": "file_b.pdf"}),
        ]
        store.add(chunks=chunks)
        sources = store.list_sources()
        assert "file_a.pdf" in sources
        assert "file_b.pdf" in sources

    def test_delete_source(self):
        store = VectorStore(collection_name="test_store_delete")
        chunks = [
            Chunk(text="Content A", metadata={"source": "file_a.pdf"}),
            Chunk(text="Content B", metadata={"source": "file_b.pdf"}),
        ]
        store.add(chunks=chunks)
        assert store.count == 2

        store.delete("file_a.pdf")
        assert store.count == 1

    def test_query_by_text(self):
        store = VectorStore(collection_name="test_store_query_text")
        chunks = [
            Chunk(text="Python programming language fundamentals", metadata={"source": "doc.pdf"}),
            Chunk(text="Advanced machine learning techniques", metadata={"source": "doc.pdf"}),
        ]
        store.add(chunks=chunks)
        results = store.query(query_text="What is Python?", n_results=2)
        assert len(results) > 0
