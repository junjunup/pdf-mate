"""Tests for the exceptions module."""

from __future__ import annotations

import pytest

from pdf_mate.exceptions import (
    ConfigError,
    EmbeddingError,
    LLMError,
    OCRError,
    ParseError,
    PDFMateError,
    RAGError,
    StorageError,
)


class TestExceptionHierarchy:
    """Verify the exception class hierarchy."""

    @pytest.mark.parametrize(
        "exc_cls",
        [ParseError, OCRError, EmbeddingError, StorageError, LLMError, RAGError, ConfigError],
    )
    def test_subclass_of_base(self, exc_cls):
        assert issubclass(exc_cls, PDFMateError)
        assert issubclass(exc_cls, Exception)

    def test_base_is_exception(self):
        assert issubclass(PDFMateError, Exception)

    @pytest.mark.parametrize(
        "exc_cls",
        [ParseError, OCRError, EmbeddingError, StorageError, LLMError, RAGError, ConfigError],
    )
    def test_catchable_by_base(self, exc_cls):
        with pytest.raises(PDFMateError):
            raise exc_cls("test error")

    def test_catchable_individually(self):
        with pytest.raises(ParseError):
            raise ParseError("parse failed")

    def test_message_preserved(self):
        exc = LLMError("connection refused")
        assert str(exc) == "connection refused"

    def test_raise_with_cause(self):
        cause = ValueError("inner")
        exc = StorageError("outer")
        exc.__cause__ = cause
        assert exc.__cause__ is cause
