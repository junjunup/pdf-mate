"""Shared test fixtures for pdf-mate."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from pdf_mate.llm import LLMBackend, LLMConfig, Message


class MockLLMBackend(LLMBackend):
    """A mock LLM backend for testing."""

    def __init__(self, responses: list[str] | None = None):
        super().__init__(LLMConfig())
        self._responses = responses or ["Mock response"]
        self._call_count = 0

    def chat(self, messages: list[Message], **kwargs) -> str:
        response = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return response

    async def chat_stream(self, messages: list[Message], **kwargs):
        text = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        for word in text.split():
            yield word + " "


@pytest.fixture
def temp_dir():
    """Provide a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text():
    """Provide sample document text."""
    return (
        "Machine learning is a branch of artificial intelligence. "
        "It focuses on the development of algorithms that can learn from data. "
        "Deep learning is a subset of machine learning using neural networks. "
        "Natural language processing enables computers to understand human language. "
        "Computer vision allows machines to interpret and make decisions from images."
    )


@pytest.fixture
def sample_chunks():
    """Provide sample text chunks."""
    from pdf_mate.storage import Chunk

    return [
        Chunk(
            text="Machine learning is a branch of AI.",
            metadata={"source": "doc.pdf", "page": 0},
        ),
        Chunk(
            text="Deep learning uses neural networks.",
            metadata={"source": "doc.pdf", "page": 1},
        ),
        Chunk(
            text="NLP enables language understanding.",
            metadata={"source": "doc.pdf", "page": 2},
        ),
    ]


@pytest.fixture
def mock_llm():
    """Provide a mock LLM backend."""
    return MockLLMBackend(responses=[
        "Mock Document Title",
        "This is a mock summary of the document.",
        "1. First key point\n2. Second key point\n3. Third key point",
    ])
