"""Exceptions for pdf-mate.

Provides a hierarchy of domain-specific exceptions so callers can catch
errors at the right granularity::

    PDFMateError
    ├── ParseError
    ├── OCRError
    ├── EmbeddingError
    ├── StorageError      (re-exported from storage module for convenience)
    ├── LLMError
    ├── RAGError
    └── ConfigError
"""

from __future__ import annotations


class PDFMateError(Exception):
    """Base exception for all pdf-mate errors."""


class ParseError(PDFMateError):
    """Raised when PDF parsing fails."""


class OCRError(PDFMateError):
    """Raised when OCR processing fails."""


class EmbeddingError(PDFMateError):
    """Raised when text embedding fails."""


class StorageError(PDFMateError):
    """Raised when vector storage operations fail."""


class LLMError(PDFMateError):
    """Raised when LLM interaction fails."""


class RAGError(PDFMateError):
    """Raised when the RAG pipeline encounters an error."""


class ConfigError(PDFMateError):
    """Raised for invalid configuration."""
