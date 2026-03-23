"""Tests for RAG module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pdf_mate.rag import RAGAnswer, RAGConfig, RAGEngine, TextSplitter


class TestTextSplitter:
    def test_split_simple(self):
        splitter = TextSplitter(chunk_size=10, chunk_overlap=2)
        text = "word " * 30  # 30 words
        chunks = splitter.split_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) > 0

    def test_split_with_metadata(self):
        splitter = TextSplitter(chunk_size=20, chunk_overlap=5)
        text = "This is a test document with enough text to split into multiple chunks."
        chunks = splitter.split_text(text, metadata={"source": "test.pdf"})
        assert all(c.metadata["source"] == "test.pdf" for c in chunks)
        assert all("chunk_index" in c.metadata for c in chunks)

    def test_split_empty_text(self):
        splitter = TextSplitter()
        chunks = splitter.split_text("")
        assert chunks == []

    def test_split_whitespace_only(self):
        splitter = TextSplitter()
        chunks = splitter.split_text("   \n  \t  ")
        assert chunks == []

    def test_short_text_single_chunk(self):
        splitter = TextSplitter(chunk_size=1000, chunk_overlap=50)
        text = "Short text."
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."

    def test_chunk_indices_are_sequential(self):
        splitter = TextSplitter(chunk_size=5, chunk_overlap=1)
        text = "one two three four five six seven eight nine ten eleven twelve"
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i

    def test_start_char_correctness_word_fallback(self):
        """Verify that start_char metadata points to correct position when using word fallback."""
        splitter = TextSplitter(chunk_size=5, chunk_overlap=0)
        # Force word-level splitting by disabling tiktoken encoding
        splitter._encoding = None
        text = "alpha beta gamma delta epsilon zeta"
        chunks = splitter.split_text(text)
        for chunk in chunks:
            start = chunk.metadata["start_char"]
            first_word = chunk.text.split()[0]
            assert text[start:start + len(first_word)] == first_word

    def test_token_level_splitting_with_tiktoken(self):
        """When tiktoken is available, splitting should use token counts."""
        splitter = TextSplitter(chunk_size=10, chunk_overlap=2)
        if splitter._encoding is None:
            pytest.skip("tiktoken not available")

        text = "The quick brown fox jumps over the lazy dog. " * 20
        chunks = splitter.split_text(text)
        assert len(chunks) > 1

        # Verify each chunk has token_start/token_end metadata
        for chunk in chunks:
            assert "token_start" in chunk.metadata
            assert "token_end" in chunk.metadata

        # Verify token counts are roughly correct
        for chunk in chunks:
            token_count = len(splitter._encoding.encode(chunk.text))
            assert token_count <= splitter.chunk_size + 2  # allow small margin

    def test_token_overlap_between_chunks(self):
        """Verify overlap works correctly in token mode."""
        splitter = TextSplitter(chunk_size=20, chunk_overlap=5)
        if splitter._encoding is None:
            pytest.skip("tiktoken not available")

        text = "Natural language processing is a field of artificial intelligence. " * 10
        chunks = splitter.split_text(text)
        if len(chunks) >= 2:
            # Check that consecutive chunks have overlapping tokens
            t1_end = chunks[0].metadata["token_end"]
            t2_start = chunks[1].metadata["token_start"]
            overlap = t1_end - t2_start
            assert overlap == splitter.chunk_overlap

    def test_invalid_chunk_size_raises(self):
        """Verify that invalid chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be a positive"):
            TextSplitter(chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size must be a positive"):
            TextSplitter(chunk_size=-1)

    def test_invalid_chunk_overlap_raises(self):
        """Verify that invalid chunk_overlap raises ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            TextSplitter(chunk_size=10, chunk_overlap=-1)

    def test_overlap_gte_size_raises(self):
        """Verify overlap >= size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            TextSplitter(chunk_size=10, chunk_overlap=10)
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            TextSplitter(chunk_size=10, chunk_overlap=15)


class TestRAGConfig:
    def test_defaults(self):
        config = RAGConfig()
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.llm_provider == "openai"
        assert config.embedding_provider == "local"

    def test_custom_config(self):
        config = RAGConfig(
            chunk_size=256,
            llm_model="gpt-4o",
            embedding_model="BAAI/bge-small-en",
        )
        assert config.chunk_size == 256
        assert config.llm_model == "gpt-4o"


class TestRAGAnswer:
    def test_creation(self):
        answer = RAGAnswer(
            question="What is AI?",
            answer="AI stands for Artificial Intelligence.",
            sources=[("test.pdf", "AI stands for...")],
            score=0.85,
        )
        assert answer.question == "What is AI?"
        assert len(answer.sources) == 1

    def test_defaults(self):
        answer = RAGAnswer(question="Q", answer="A")
        assert answer.sources == []
        assert answer.score == 0.0


class TestRAGEngine:
    @patch("pdf_mate.rag.create_llm_backend")
    @patch("pdf_mate.rag.VectorStore")
    @patch("pdf_mate.rag.create_embedding_backend")
    def test_index_document(self, mock_embed_factory, mock_store_cls, mock_llm_factory):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [np.array([0.1, 0.2])] * 3
        mock_embed_factory.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        engine = RAGEngine()
        count = engine.index_document("Hello world. This is text.", source_name="doc.pdf")
        assert count >= 1
        mock_store.add.assert_called_once()

    @patch("pdf_mate.rag.create_llm_backend")
    @patch("pdf_mate.rag.VectorStore")
    @patch("pdf_mate.rag.create_embedding_backend")
    def test_index_empty_document(self, mock_embed_factory, mock_store_cls, mock_llm_factory):
        mock_embedder = MagicMock()
        mock_embed_factory.return_value = mock_embedder
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        engine = RAGEngine()
        count = engine.index_document("   ", source_name="empty.pdf")
        assert count == 0
        mock_store.add.assert_not_called()

    @patch("pdf_mate.rag.create_llm_backend")
    @patch("pdf_mate.rag.VectorStore")
    @patch("pdf_mate.rag.create_embedding_backend")
    def test_query_no_results(self, mock_embed_factory, mock_store_cls, mock_llm_factory):
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.array([0.1, 0.2])
        mock_embed_factory.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.query.return_value = []
        mock_store_cls.return_value = mock_store

        engine = RAGEngine()
        answer = engine.query("What is this?")
        assert "couldn't find" in answer.answer.lower()
        assert answer.sources == []
