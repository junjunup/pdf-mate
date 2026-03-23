"""Tests for Embedding module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pdf_mate.embedding import (
    EmbeddingBackend,
    OpenAIEmbeddingBackend,
    SentenceTransformerBackend,
    create_embedding_backend,
)
from pdf_mate.exceptions import ConfigError, EmbeddingError


class TestEmbeddingBackend:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            EmbeddingBackend()


class TestSentenceTransformerBackend:
    def test_init(self):
        backend = SentenceTransformerBackend(model_name="test-model")
        assert backend.model_name == "test-model"
        assert backend.batch_size == 32

    @patch("pdf_mate.embedding.SentenceTransformerBackend._get_model")
    def test_embed(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.encode.return_value = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        mock_get_model.return_value = mock_model

        backend = SentenceTransformerBackend()
        results = backend.embed(["hello", "world"])
        assert len(results) == 2
        assert isinstance(results[0], np.ndarray)

    @patch("pdf_mate.embedding.SentenceTransformerBackend._get_model")
    def test_embed_query(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.encode.return_value = [np.array([0.1, 0.2])]
        mock_get_model.return_value = mock_model

        backend = SentenceTransformerBackend()
        result = backend.embed_query("hello")
        assert isinstance(result, np.ndarray)


class TestOpenAIEmbeddingBackend:
    def test_init(self):
        backend = OpenAIEmbeddingBackend(model="text-embedding-3-small")
        assert backend.model == "text-embedding-3-small"
        assert backend.batch_size == 100

    @patch("pdf_mate.embedding.OpenAIEmbeddingBackend._get_client")
    def test_embed(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = [0.1, 0.2, 0.3]
        mock_response.data = [mock_item]
        mock_client.embeddings.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        backend = OpenAIEmbeddingBackend()
        results = backend.embed(["hello"])
        assert len(results) == 1
        assert isinstance(results[0], np.ndarray)


class TestCreateEmbeddingBackend:
    def test_local_backend(self):
        backend = create_embedding_backend(provider="local")
        assert isinstance(backend, SentenceTransformerBackend)

    def test_openai_backend(self):
        backend = create_embedding_backend(provider="openai", model_name="text-embedding-3-small")
        assert isinstance(backend, OpenAIEmbeddingBackend)

    def test_default_is_local(self):
        backend = create_embedding_backend()
        assert isinstance(backend, SentenceTransformerBackend)

    def test_custom_model_name(self):
        backend = create_embedding_backend(provider="local", model_name="BAAI/bge-small-en")
        assert backend.model_name == "BAAI/bge-small-en"

    def test_unknown_provider_raises(self):
        with pytest.raises(ConfigError, match="Unknown embedding provider"):
            create_embedding_backend(provider="invalid_provider")


class TestSentenceTransformerImportError:
    def test_import_error_raises_embedding_error(self):
        """Test that missing sentence-transformers raises EmbeddingError."""
        backend = SentenceTransformerBackend()
        backend._model = None  # ensure model is not cached

        with patch("pdf_mate.embedding.SentenceTransformerBackend._get_model") as mock_get:
            mock_get.side_effect = EmbeddingError("sentence-transformers is required")
            with pytest.raises(EmbeddingError, match="sentence-transformers"):
                backend.embed(["test"])

    @patch("pdf_mate.embedding.SentenceTransformerBackend._get_model")
    def test_embed_generic_error_wraps(self, mock_get_model):
        """Test that generic errors during embedding are wrapped."""
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("GPU OOM")
        mock_get_model.return_value = mock_model

        backend = SentenceTransformerBackend()
        with pytest.raises(EmbeddingError, match="Embedding failed"):
            backend.embed(["test"])


class TestOpenAIEmbeddingBatchProcessing:
    @patch("pdf_mate.embedding.OpenAIEmbeddingBackend._get_client")
    def test_batch_processing(self, mock_get_client):
        """Test that large input is processed in batches."""
        mock_client = MagicMock()

        def mock_create(input, model):
            """Return embeddings for each input text."""
            items = []
            for _ in input:
                item = MagicMock()
                item.embedding = [0.1, 0.2, 0.3]
                items.append(item)
            response = MagicMock()
            response.data = items
            return response

        mock_client.embeddings.create.side_effect = mock_create
        mock_get_client.return_value = mock_client

        backend = OpenAIEmbeddingBackend(batch_size=3)
        texts = [f"text {i}" for i in range(7)]
        results = backend.embed(texts)
        assert len(results) == 7
        # Should have made 3 calls: batch of 3, 3, 1
        assert mock_client.embeddings.create.call_count == 3

    @patch("pdf_mate.embedding.OpenAIEmbeddingBackend._get_client")
    def test_embed_api_error_wraps(self, mock_get_client):
        """Test that API errors are wrapped in EmbeddingError."""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = RuntimeError("API timeout")
        mock_get_client.return_value = mock_client

        backend = OpenAIEmbeddingBackend()
        with pytest.raises(EmbeddingError, match="OpenAI embedding failed"):
            backend.embed(["test"])

    def test_client_caching(self):
        """Test that _get_client caches the client."""
        backend = OpenAIEmbeddingBackend(api_key="sk-test", base_url="https://api.example.com")
        mock_client = MagicMock()
        backend._client = mock_client
        client1 = backend._get_client()
        client2 = backend._get_client()
        assert client1 is client2
        assert client1 is mock_client

    def test_client_import_error(self):
        """Test that missing openai package raises EmbeddingError."""
        backend = OpenAIEmbeddingBackend()
        backend._client = None

        with patch.dict("sys.modules", {"openai": None}):
            with patch("builtins.__import__", side_effect=ImportError("No openai")):
                with pytest.raises(EmbeddingError, match="openai is required"):
                    backend._get_client()
