"""Tests for the summary module."""

from __future__ import annotations

import pytest

from pdf_mate.summary import DocumentSummarizer, DocumentSummary, SummaryConfig


class TestSummaryConfig:
    def test_defaults(self):
        config = SummaryConfig()
        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4o-mini"
        assert config.style == "concise"
        assert config.max_summary_length == 500

    def test_custom(self):
        config = SummaryConfig(llm_model="gpt-4o", style="detailed")
        assert config.llm_model == "gpt-4o"
        assert config.style == "detailed"


class TestDocumentSummary:
    def test_creation(self):
        summary = DocumentSummary(
            title="Test Title",
            summary="Test summary text.",
            key_points=["Point 1", "Point 2"],
            page_count=5,
            word_count=1000,
        )
        assert summary.title == "Test Title"
        assert len(summary.key_points) == 2
        assert summary.word_count == 1000


class TestDocumentSummarizer:
    def test_init_default(self):
        summarizer = DocumentSummarizer()
        assert summarizer.config.llm_provider == "openai"

    def test_init_custom(self):
        config = SummaryConfig(llm_model="gpt-4o")
        summarizer = DocumentSummarizer(config)
        assert summarizer.config.llm_model == "gpt-4o"

    def test_summarize_empty_text_raises(self):
        """Test that empty text raises ValueError."""
        summarizer = DocumentSummarizer()
        with pytest.raises(ValueError, match="Cannot summarize empty text"):
            summarizer.summarize(text="", filename="empty.pdf")

    def test_summarize_whitespace_only_raises(self):
        """Test that whitespace-only text raises ValueError."""
        summarizer = DocumentSummarizer()
        with pytest.raises(ValueError, match="Cannot summarize empty text"):
            summarizer.summarize(text="   \n  \t  ", filename="empty.pdf")

    def test_summarize_with_mock(self, mock_llm):
        """Test the summarize method using a mock LLM."""
        summarizer = DocumentSummarizer()
        summarizer._llm = mock_llm

        result = summarizer.summarize(
            text="This is a test document with enough content to summarize.",
            filename="test.pdf",
            page_count=3,
        )

        assert isinstance(result, DocumentSummary)
        assert result.title == "Mock Document Title"
        assert result.page_count == 3
        assert result.word_count > 0
        assert len(result.key_points) > 0

    def test_summarize_truncates_long_text(self, mock_llm):
        """Test that very long text gets truncated."""
        summarizer = DocumentSummarizer()
        summarizer._llm = mock_llm

        long_text = "word " * 50000  # ~250000 chars
        result = summarizer.summarize(text=long_text, filename="big.pdf")

        assert isinstance(result, DocumentSummary)

    def test_summarize_styles(self, mock_llm):
        """Test different summary styles."""
        for style in ("concise", "detailed", "bullets"):
            config = SummaryConfig(style=style)
            summarizer = DocumentSummarizer(config)
            summarizer._llm = mock_llm

            result = summarizer.summarize(text="Test content.", filename="test.pdf")
            assert isinstance(result, DocumentSummary)
