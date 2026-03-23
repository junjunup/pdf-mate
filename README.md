# 📄 pdf-mate

<p align="center">
  <strong>AI-powered PDF parsing, summarization, and conversational Q&A</strong>
</p>

<p align="center">
  <a href="https://github.com/junjunup/pdf-mate/stargazers"><img src="https://img.shields.io/github/stars/junjunup/pdf-mate?style=social" alt="Stars"></a>
  <a href="https://github.com/junjunup/pdf-mate/releases"><img src="https://img.shields.io/github/v/release/junjunup/pdf-mate" alt="Release"></a>
  <a href="https://github.com/junjunup/pdf-mate/blob/main/LICENSE"><img src="https://img.shields.io/github/license/junjunup/pdf-mate" alt="License"></a>
</p>

---

**pdf-mate** is a lightweight Python toolkit that combines PDF extraction, OCR, RAG-based question answering, and intelligent summarization. It supports multiple LLM backends (OpenAI, Ollama, etc.) and provides both a powerful CLI and a clean Gradio Web UI.

## ✨ Features

- **PDF Parsing** — Extract text, tables, and images from PDF files
- **OCR Support** — Extract text from scanned documents (powered by Tesseract)
- **RAG Q&A** — Ask questions about your documents with retrieval-augmented generation
- **AI Summarization** — Generate concise, detailed, or bullet-point summaries
- **Multi-LLM Support** — Works with OpenAI, Ollama, and any OpenAI-compatible API
- **Local Embedding** — Privacy-first with local sentence-transformers (no API needed)
- **CLI Tool** — Fast command-line interface for power users
- **Web UI** — Clean Gradio interface for interactive exploration

## 🚀 Quick Start

### Installation

The core package is lightweight (~100 MB). Heavy optional dependencies are split into extras:

```bash
# Core only (PDF parsing + CLI — lightweight, no API deps)
pip install pdf-mate

# With LLM support (OpenAI API + summarization)
pip install pdf-mate[llm]

# With RAG support (LLM + ChromaDB vector store + Q&A)
pip install pdf-mate[rag]

# With local embedding support (~2 GB, includes sentence-transformers)
pip install pdf-mate[local]

# With Web UI (includes Gradio)
pip install pdf-mate[web]

# With OCR support (requires Tesseract system package)
pip install pdf-mate[ocr]

# Everything
pip install pdf-mate[all]

# For development
pip install -e ".[dev]"
```

> **Note:** The core package only includes PDF parsing and CLI — no OpenAI or ChromaDB dependencies. Install `pdf-mate[rag]` for the full RAG Q&A experience, or `pdf-mate[llm]` for summarization only.

### CLI Usage

```bash
# Parse a PDF file
pdf-mate parse document.pdf

# Parse and save as markdown
pdf-mate parse document.pdf -o output.md --format markdown

# Generate an AI summary
pdf-mate summary document.pdf

# Interactive Q&A mode
pdf-mate ask document.pdf

# Ask a single question
pdf-mate ask document.pdf -q "What is the main topic?"

# OCR a scanned PDF
pdf-mate ocr scanned.pdf --lang eng+chi_sim

# Launch Web UI
pdf-mate web
```

### Web UI

```bash
pdf-mate web --port 7860
```

Then open `http://localhost:7860` in your browser. The Web UI provides:

- **Parse Tab** — Upload and extract content from PDFs
- **Summary Tab** — Generate AI-powered document summaries
- **Q&A Tab** — Index documents and ask questions interactively

## 📖 Usage Examples

### Python API

```python
from pdf_mate.parser import PDFParser

# Parse a PDF
parser = PDFParser()
content = parser.parse("document.pdf")

print(f"Pages: {content.page_count}")
print(f"Text: {content.full_text[:500]}")
print(f"Tables: {len(content.tables)}")
```

### RAG Question Answering

```python
from pdf_mate.parser import PDFParser
from pdf_mate.rag import RAGEngine, RAGConfig

# Setup RAG engine with local embeddings
config = RAGConfig(
    embedding_provider="local",  # requires: pip install pdf-mate[local]
    embedding_model="all-MiniLM-L6-v2",
    llm_model="gpt-4o-mini",
)
engine = RAGEngine(config)

# Index a document
parser = PDFParser()
content = parser.parse("research_paper.pdf")
engine.index_document(content.full_text, source_name="research_paper.pdf")

# Ask questions
answer = engine.query("What are the main findings?")
print(answer.answer)
print("Sources:", [s[0] for s in answer.sources])
```

### Document Summarization

```python
from pdf_mate.parser import PDFParser
from pdf_mate.summary import DocumentSummarizer, SummaryConfig

# Parse and summarize
parser = PDFParser()
content = parser.parse("report.pdf")

config = SummaryConfig(
    llm_model="gpt-4o-mini",
    style="bullets",  # "concise", "detailed", or "bullets"
)
summarizer = DocumentSummarizer(config)
summary = summarizer.summarize(
    text=content.full_text,
    filename=content.filename,
    page_count=content.page_count,
)

print(f"Title: {summary.title}")
print(f"Summary: {summary.summary}")
print("Key Points:")
for point in summary.key_points:
    print(f"  - {point}")
```

### OCR for Scanned Documents

```python
from pdf_mate.ocr import OCREngine

# requires: pip install pdf-mate[ocr]
engine = OCREngine(language="eng+chi_sim")
results = engine.extract_text_from_pdf("scanned.pdf")

for page_num, text in results:
    print(f"--- Page {page_num + 1} ---")
    print(text)
```

### Exception Handling

```python
from pdf_mate import PDFParser, PDFMateError, ParseError, LLMError

try:
    parser = PDFParser()
    content = parser.parse("document.pdf")
except ParseError as e:
    print(f"Parse failed: {e}")
except PDFMateError as e:
    print(f"pdf-mate error: {e}")
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM and embedding | - |
| `OPENAI_BASE_URL` | Custom API base URL | - |

### Using Ollama (Local LLM)

```bash
# Start Ollama server
ollama serve

# Pull a model
ollama pull llama3

# Use with pdf-mate
pdf-mate ask document.pdf --model llama3 --base-url http://localhost:11434/v1
```

## 🏗️ Architecture

```
pdf-mate/
├── src/pdf_mate/
│   ├── __init__.py     # Public API exports
│   ├── exceptions.py   # Custom exception hierarchy
│   ├── parser.py       # PDF text, table, image extraction
│   ├── ocr.py          # OCR for scanned documents
│   ├── llm.py          # Multi-backend LLM interface
│   ├── embedding.py    # Text vectorization
│   ├── storage.py      # ChromaDB vector store
│   ├── rag.py          # RAG pipeline (chunking + retrieval + generation)
│   ├── summary.py      # Document summarization
│   ├── cli.py          # CLI interface (Typer + Rich)
│   └── web.py          # Gradio Web UI
├── tests/              # pytest test suite
└── pyproject.toml      # Project configuration
```

## 🛠️ Development

```bash
# Clone the repository
git clone https://github.com/junjunup/pdf-mate.git
cd pdf-mate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=pdf_mate

# Lint
ruff check src/ tests/

# Type checking
mypy src/
```

## 📋 Requirements

- Python >= 3.10

**Core dependencies:**
- `pdfplumber` — PDF text and table extraction
- `PyMuPDF` — PDF image extraction and page rendering
- `numpy` — Numerical operations
- `typer` + `rich` — CLI framework

**Optional extras:**
- `openai` + `tiktoken` — LLM API client (`pip install pdf-mate[llm]`)
- `chromadb` — Vector database for RAG (`pip install pdf-mate[rag]`)
- `sentence-transformers` — Local embeddings (`pip install pdf-mate[local]`)
- `gradio` — Web UI (`pip install pdf-mate[web]`)
- `pytesseract` + `Pillow` — OCR support (`pip install pdf-mate[ocr]`)

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
