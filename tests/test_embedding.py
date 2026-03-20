"""Tests for Embedding module."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from pdf_mate.embedding import (
    SentenceTransformerBackend,
    OpenAIEmbeddingBackend,
    create_embedding_backend,
)


class TestCreateEmbeddingBackend:
    @patch("pdf_mate.embedding.SentenceTransformer")
    def test_local_backend(self, mock_st):
        """Test creating local embedding backend."""
        mock_st.return_value = MagicMock()
        backend = create_embedding_backend(provider="local")
        assert isinstance(backend, SentenceTransformerBackend)

    @patch("pdf_mate.embedding.OpenAI")
    def test_openai_backend(self, mock_openai):
        """Test creating OpenAI embedding backend."""
        mock_openai.return_value = MagicMock()
        backend = create_embedding_backend(provider="openai", model_name="text-embedding-3-small")
        assert isinstance(backend, OpenAIEmbeddingBackend)

    @patch("pdf_mate.embedding.SentenceTransformer")
    def test_default_backend(self, mock_st):
        """Test default backend is local."""
        mock_st.return_value = MagicMock()
        backend = create_embedding_backend()
        assert isinstance(backend, SentenceTransformerBackend)

    @patch("pdf_mate.embedding.SentenceTransformer")
    def test_custom_model(self, mock_st):
        """Test custom model name."""
        mock_st.return_value = MagicMock()
        backend = create_embedding_backend(
            provider="local", model_name="BAAI/bge-small-en"
        )
        assert backend.model_name == "BAAI/bge-small-en"
