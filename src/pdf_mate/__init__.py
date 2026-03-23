"""
pdf-mate: AI-powered PDF parsing, summarization, and conversational Q&A.

A lightweight Python tool that combines PDF extraction, OCR, RAG-based
question answering, and intelligent summarization with support for
multiple LLM backends.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Exceptions are always safe to import eagerly (pure Python, no heavy deps)
# Lazy imports for the rest — avoids pulling in heavy C-extension deps
# (fitz, chromadb, etc.) at import time. Users can do:
#   from pdf_mate import PDFParser     → triggers lazy load
#   from pdf_mate.parser import PDFParser  → direct import
import importlib as _importlib
from typing import TYPE_CHECKING

from .exceptions import (
    ConfigError,
    EmbeddingError,
    LLMError,
    OCRError,
    ParseError,
    PDFMateError,
    RAGError,
    StorageError,
)

if TYPE_CHECKING:
    # Static type checkers see the full API
    from .embedding import EmbeddingBackend, create_embedding_backend
    from .llm import LLMBackend, LLMConfig, Message, create_llm_backend
    from .ocr import OCREngine
    from .parser import ImageInfo, PDFContent, PDFParser, Table, TextBlock
    from .rag import RAGAnswer, RAGConfig, RAGEngine, TextSplitter
    from .storage import Chunk, VectorStore
    from .summary import DocumentSummarizer, DocumentSummary, SummaryConfig


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Parser
    "PDFParser": (".parser", "PDFParser"),
    "PDFContent": (".parser", "PDFContent"),
    "TextBlock": (".parser", "TextBlock"),
    "Table": (".parser", "Table"),
    "ImageInfo": (".parser", "ImageInfo"),
    # LLM
    "LLMBackend": (".llm", "LLMBackend"),
    "LLMConfig": (".llm", "LLMConfig"),
    "Message": (".llm", "Message"),
    "create_llm_backend": (".llm", "create_llm_backend"),
    # Embedding
    "EmbeddingBackend": (".embedding", "EmbeddingBackend"),
    "create_embedding_backend": (".embedding", "create_embedding_backend"),
    # Storage
    "Chunk": (".storage", "Chunk"),
    "VectorStore": (".storage", "VectorStore"),
    # RAG
    "RAGConfig": (".rag", "RAGConfig"),
    "RAGAnswer": (".rag", "RAGAnswer"),
    "RAGEngine": (".rag", "RAGEngine"),
    "TextSplitter": (".rag", "TextSplitter"),
    # Summary
    "SummaryConfig": (".summary", "SummaryConfig"),
    "DocumentSummary": (".summary", "DocumentSummary"),
    "DocumentSummarizer": (".summary", "DocumentSummarizer"),
    # OCR
    "OCREngine": (".ocr", "OCREngine"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Version
    "__version__",
    # Exceptions (eagerly loaded)
    "PDFMateError",
    "ParseError",
    "OCRError",
    "EmbeddingError",
    "StorageError",
    "LLMError",
    "RAGError",
    "ConfigError",
    # Parser
    "PDFParser",
    "PDFContent",
    "TextBlock",
    "Table",
    "ImageInfo",
    # LLM
    "LLMBackend",
    "LLMConfig",
    "Message",
    "create_llm_backend",
    # Embedding
    "EmbeddingBackend",
    "create_embedding_backend",
    # Storage
    "Chunk",
    "VectorStore",
    # RAG
    "RAGConfig",
    "RAGAnswer",
    "RAGEngine",
    "TextSplitter",
    # Summary
    "SummaryConfig",
    "DocumentSummary",
    "DocumentSummarizer",
    # OCR
    "OCREngine",
]
