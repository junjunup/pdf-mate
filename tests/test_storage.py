"""Tests for Storage module."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from pdf_mate.storage import Chunk, VectorStore


class TestChunk:
    def test_creation(self):
        chunk = Chunk(text="Hello world", metadata={"source": "test.pdf"})
        assert chunk.text == "Hello world"
        assert chunk.metadata["source"] == "test.pdf"


class TestVectorStore:
    @patch("pdf_mate.storage.Chroma")
    @patch("pdf_mate.storage.SentenceTransformer")
    def test_create_and_count(self, mock_st, mock_chroma):
        """Test creating a vector store."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        store = VectorStore(collection_name="test_store")
        assert store.count == 0

    @patch("pdf_mate.storage.Chroma")
    @patch("pdf_mate.storage.SentenceTransformer")
    def test_add_chunks(self, mock_st, mock_chroma):
        """Test adding chunks to store."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 3
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        # Mock embedding model
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 384] * 3
        mock_st.return_value = mock_model
        
        store = VectorStore(collection_name="test_store")
        chunks = [
            Chunk(text="Python is a programming language", metadata={"source": "doc1.pdf", "page": 1}),
            Chunk(text="Machine learning is a subset of AI", metadata={"source": "doc1.pdf", "page": 2}),
            Chunk(text="RAG combines retrieval and generation", metadata={"source": "doc2.pdf", "page": 1}),
        ]
        store.add(chunks=chunks)
        assert store.count == 3

    @patch("pdf_mate.storage.Chroma")
    @patch("pdf_mate.storage.SentenceTransformer")
    def test_list_sources(self, mock_st, mock_chroma):
        """Test listing sources."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.get.return_value = {
            "metadatas": [
                {"source": "file_a.pdf"},
                {"source": "file_b.pdf"},
            ]
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        store = VectorStore(collection_name="test_store")
        sources = store.list_sources()
        assert "file_a.pdf" in sources
        assert "file_b.pdf" in sources

    @patch("pdf_mate.storage.Chroma")
    @patch("pdf_mate.storage.SentenceTransformer")
    def test_delete_source(self, mock_st, mock_chroma):
        """Test deleting a source."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.side_effect = [2, 1]  # Before and after delete
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        store = VectorStore(collection_name="test_store")
        assert store.count == 2
        
        store.delete("file_a.pdf")
        assert store.count == 1

    @patch("pdf_mate.storage.Chroma")
    @patch("pdf_mate.storage.SentenceTransformer")
    def test_query_by_text(self, mock_st, mock_chroma):
        """Test querying by text."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["Python programming", "ML techniques"]],
            "metadatas": [[{"source": "doc.pdf"}, {"source": "doc.pdf"}]],
            "distances": [[0.1, 0.2]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        # Mock embedding model
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 384]
        mock_st.return_value = mock_model
        
        store = VectorStore(collection_name="test_store")
        results = store.query(query_text="What is Python?", n_results=2)
        assert len(results) == 2
