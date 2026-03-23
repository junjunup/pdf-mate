"""Tests for Storage module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pdf_mate.storage import Chunk, VectorStore


class TestChunk:
    def test_creation(self):
        chunk = Chunk(text="Hello world", metadata={"source": "test.pdf"})
        assert chunk.text == "Hello world"
        assert chunk.metadata["source"] == "test.pdf"

    def test_empty_metadata(self):
        chunk = Chunk(text="text", metadata={})
        assert chunk.metadata == {}


class TestVectorStore:
    @patch("pdf_mate.storage.chromadb.EphemeralClient")
    def test_create_ephemeral(self, mock_client_class):
        """Test creating an ephemeral (in-memory) vector store."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        store = VectorStore(collection_name="test_store")
        assert store.count == 0
        mock_client_class.assert_called_once()

    @patch("pdf_mate.storage.chromadb.PersistentClient")
    def test_create_persistent(self, mock_client_class, temp_dir):
        """Test creating a persistent vector store."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        store = VectorStore(
            collection_name="test_store",
            persist_directory=str(temp_dir),
        )
        assert store.count == 0
        mock_client_class.assert_called_once_with(path=str(temp_dir))

    @patch("pdf_mate.storage.chromadb.EphemeralClient")
    def test_add_chunks(self, mock_client_class):
        """Test adding chunks to store."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        store = VectorStore(collection_name="test_store")
        chunks = [
            Chunk(text="Python is a programming language", metadata={"source": "doc1.pdf"}),
            Chunk(text="Machine learning is a subset of AI", metadata={"source": "doc1.pdf"}),
        ]
        store.add(chunks=chunks)
        mock_collection.add.assert_called_once()

    @patch("pdf_mate.storage.chromadb.EphemeralClient")
    def test_add_empty_chunks(self, mock_client_class):
        """Test that adding empty chunks is a no-op."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        store = VectorStore(collection_name="test_store")
        store.add(chunks=[])
        mock_collection.add.assert_not_called()

    @patch("pdf_mate.storage.chromadb.EphemeralClient")
    def test_add_with_embeddings(self, mock_client_class):
        """Test adding chunks with pre-computed embeddings."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        store = VectorStore(collection_name="test_store")
        chunks = [Chunk(text="Hello", metadata={"source": "a.pdf"})]
        embeddings = [np.array([0.1, 0.2, 0.3])]
        store.add(chunks=chunks, embeddings=embeddings)
        mock_collection.add.assert_called_once()

    @patch("pdf_mate.storage.chromadb.EphemeralClient")
    def test_query_empty_collection(self, mock_client_class):
        """Test querying an empty collection returns empty list."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        store = VectorStore(collection_name="test_store")
        results = store.query(query_text="What?", n_results=5)
        assert results == []
        mock_collection.query.assert_not_called()

    @patch("pdf_mate.storage.chromadb.EphemeralClient")
    def test_query_by_text(self, mock_client_class):
        """Test querying by text."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["Python programming", "ML techniques"]],
            "metadatas": [[{"source": "doc.pdf"}, {"source": "doc.pdf"}]],
            "distances": [[0.1, 0.2]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        store = VectorStore(collection_name="test_store")
        results = store.query(query_text="What is Python?", n_results=2)
        assert len(results) == 2
        assert results[0][0].text == "Python programming"
        assert results[0][1] == 0.1

    @patch("pdf_mate.storage.chromadb.EphemeralClient")
    def test_query_clamps_n_results(self, mock_client_class):
        """Test that n_results is clamped to stored count."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["text"]],
            "metadatas": [[{"source": "a.pdf"}]],
            "distances": [[0.1]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        store = VectorStore(collection_name="test_store")
        store.query(query_text="test", n_results=100)
        call_kwargs = mock_collection.query.call_args
        assert call_kwargs[1]["n_results"] == 2  # clamped to stored count

    @patch("pdf_mate.storage.chromadb.EphemeralClient")
    def test_query_no_args_raises(self, mock_client_class):
        """Test that query raises ValueError if no query provided."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        store = VectorStore(collection_name="test_store")
        with pytest.raises(ValueError, match="Either query_embedding or query_text"):
            store.query()

    @patch("pdf_mate.storage.chromadb.EphemeralClient")
    def test_list_sources(self, mock_client_class):
        """Test listing sources."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "metadatas": [
                {"source": "file_b.pdf"},
                {"source": "file_a.pdf"},
                {"source": "file_a.pdf"},  # duplicate
            ]
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        store = VectorStore(collection_name="test_store")
        sources = store.list_sources()
        assert sources == ["file_a.pdf", "file_b.pdf"]  # sorted, deduplicated

    @patch("pdf_mate.storage.chromadb.EphemeralClient")
    def test_delete_source(self, mock_client_class):
        """Test deleting a source."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        store = VectorStore(collection_name="test_store")
        store.delete("file_a.pdf")
        mock_collection.delete.assert_called_once_with(where={"source": "file_a.pdf"})

    @patch("pdf_mate.storage.chromadb.EphemeralClient")
    def test_delete_all(self, mock_client_class):
        """Test clearing the entire collection."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        store = VectorStore(collection_name="test_store")
        store.delete()
        mock_client.delete_collection.assert_called_once_with("test_store")
