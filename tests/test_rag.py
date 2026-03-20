"""Tests for RAG module."""

from __future__ import annotations

import pytest

from pdf_mate.rag import TextSplitter, Chunk, RAGConfig


class TestTextSplitter:
    def test_split_simple(self):
        splitter = TextSplitter(chunk_size=10, chunk_overlap=2)
        text = "word " * 30  # 60 words
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
        assert config.embedding_model == "BAAI/bge-small-en"


class TestRAGAnswer:
    def test_creation(self):
        from pdf_mate.rag import RAGAnswer

        answer = RAGAnswer(
            question="What is AI?",
            answer="AI stands for Artificial Intelligence.",
            sources=[("test.pdf", "AI stands for...")],
            score=0.85,
        )
        assert answer.question == "What is AI?"
        assert len(answer.sources) == 1
