"""Tests for Embedding module."""

from __future__ import annotations

import pytest

from pdf_mate.embedding import (
    SentenceTransformerBackend,
    OpenAIEmbeddingBackend,
    create_embedding_backend,
)


class TestCreateEmbeddingBackend:
    def test_local_backend(self):
        backend = create_embedding_backend(provider="local")
        assert isinstance(backend, SentenceTransformerBackend)

    def test_openai_backend(self):
        backend = create_embedding_backend(provider="openai", model_name="text-embedding-3-small")
        assert isinstance(backend, OpenAIEmbeddingBackend)

    def test_default_backend(self):
        backend = create_embedding_backend()
        assert isinstance(backend, SentenceTransformerBackend)

    def test_custom_model(self):
        backend = create_embedding_backend(
            provider="local", model_name="BAAI/bge-small-en"
        )
        assert backend.model_name == "BAAI/bge-small-en"
