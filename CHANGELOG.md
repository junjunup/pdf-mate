# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Unified exception hierarchy (`PDFMateError` base with `ParseError`, `OCRError`, `EmbeddingError`, `StorageError`, `LLMError`, `RAGError`, `ConfigError`)
- Comprehensive test suite achieving 89%+ code coverage
- Gradio-based Web UI with Parse, Summary, and Q&A tabs
- RAG pipeline with configurable chunking, embedding, and retrieval
- OCR support via Tesseract with automatic language detection
- CLI with `parse`, `summary`, `ocr`, `ask`, and `web` commands
- `py.typed` marker for PEP 561 typing support

### Fixed
- `parser.py` and `ocr.py` now raise domain-specific exceptions (`ParseError`, `OCRError`) instead of `FileNotFoundError`, maintaining the `PDFMateError` contract
- `embedding.py` no longer double-wraps `EmbeddingError` in the `embed()` method
- `summary.py` logs a warning when text is truncated beyond 120K characters
- CI workflow consolidated duplicate `pytest` runs into a single command

## [0.1.0] - 2025-03-23

### Added
- Initial release
- PDF parsing with text, table, and image extraction (pdfplumber + PyMuPDF)
- LLM-powered document summarization (OpenAI, Ollama backends)
- RAG-based document Q&A with vector storage (ChromaDB)
- Local and OpenAI embedding backends
- Command-line interface via Typer
- Gradio web interface

[Unreleased]: https://github.com/junjunup/pdf-mate/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/junjunup/pdf-mate/releases/tag/v0.1.0
