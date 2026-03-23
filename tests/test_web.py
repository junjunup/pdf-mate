"""Tests for the Web UI module.

Strategy: We mock the entire ``gradio`` package so that ``create_app()`` executes
all module-level and closure code without requiring an actual gradio installation.
We capture handler function references from ``Button.click()`` calls, then invoke
them directly with mocked dependencies to cover every handler path.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

# ── Ensure problematic native modules are mocked before any pdf_mate import ──
for _mod in ("fitz", "pymupdf", "pymupdf._extra", "pymupdf.extra", "pdfplumber"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest  # noqa: E402

# ── Build a comprehensive gradio mock ────────────────────────────────────────

_click_registry: list[dict] = []
"""Collects every ``button.click(fn=..., ...)`` call so we can extract handlers."""


def _make_component(**kwargs):
    """Return a MagicMock that records `.click()` calls."""
    comp = MagicMock()

    def _click(fn=None, inputs=None, outputs=None, **kw):
        _click_registry.append({"fn": fn, "inputs": inputs, "outputs": outputs})

    comp.click = _click
    return comp


def _build_gr_mock():
    """Build a fake ``gradio`` module with enough API surface for create_app()."""
    gr = MagicMock()

    # Context managers: Blocks, Tabs, Tab, Row, Column
    for attr in ("Blocks", "Tabs", "Tab", "Row", "Column"):
        ctx = MagicMock()
        ctx.__enter__ = lambda self: self
        ctx.__exit__ = lambda self, *a: None
        # Make calling it return a fresh context manager each time
        getattr(gr, attr).return_value = ctx

    # Blocks needs to be the demo object returned by create_app
    blocks_instance = MagicMock()
    blocks_instance.__enter__ = lambda self: self
    blocks_instance.__exit__ = lambda self, *a: None
    gr.Blocks.return_value = blocks_instance

    # themes.Soft()
    gr.themes.Soft.return_value = MagicMock()

    # State
    gr.State.return_value = MagicMock()

    # UI components that get assigned to variables
    gr.File.return_value = _make_component()
    gr.Checkbox.return_value = _make_component()
    gr.Button.return_value = _make_component()
    gr.Dropdown.return_value = _make_component()
    gr.Textbox.return_value = _make_component()
    gr.Markdown.return_value = _make_component()

    return gr


def _install_gr_mock():
    """Install fake gradio into sys.modules and return (gr_mock, handlers dict)."""
    global _click_registry
    _click_registry = []

    gr = _build_gr_mock()
    sys.modules["gradio"] = gr
    # Also mock gradio sub-packages that Gradio itself might reference
    sys.modules["gradio.themes"] = gr.themes

    # Ensure native modules are mocked (fitz / pymupdf may fail on DLL load)
    for _mod in ("fitz", "pymupdf", "pymupdf._extra", "pymupdf.extra", "pdfplumber"):
        if _mod not in sys.modules or not isinstance(sys.modules[_mod], MagicMock):
            sys.modules[_mod] = MagicMock()

    # Force reimport of web module so it picks up our mock
    if "pdf_mate.web" in sys.modules:
        del sys.modules["pdf_mate.web"]

    from pdf_mate.web import create_app

    app = create_app()

    # Build handler map from click registry:
    # The order in create_app is: parse_btn, sum_btn, index_btn, qa_ask_btn
    handlers = {}
    names = ["handle_parse", "handle_summary", "handle_index", "handle_question"]
    for i, name in enumerate(names):
        if i < len(_click_registry):
            handlers[name] = _click_registry[i]["fn"]

    return gr, app, handlers


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _cleanup_gradio():
    """Remove gradio mock from sys.modules after each test."""
    yield
    sys.modules.pop("gradio", None)
    sys.modules.pop("pdf_mate.web", None)


@pytest.fixture()
def web_handlers():
    """Set up mocked gradio and return (gr_mock, app, handlers)."""
    return _install_gr_mock()


# ── Test: create_app without gradio installed ────────────────────────────────

class TestCreateAppImportError:
    def test_raises_import_error_without_gradio(self):
        """create_app() raises ImportError when gradio is not installed."""
        # Ensure gradio is NOT in sys.modules
        saved = sys.modules.pop("gradio", None)
        sys.modules.pop("pdf_mate.web", None)
        try:
            # Patch builtins.__import__ to block gradio
            builtin_import = getattr(__builtins__, "__import__", __import__)

            def _blocked_import(name, *args, **kwargs):
                if name == "gradio":
                    raise ImportError("no gradio")
                return builtin_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=_blocked_import):
                from pdf_mate.web import create_app
                with pytest.raises(ImportError, match="Gradio is required"):
                    create_app()
        finally:
            if saved is not None:
                sys.modules["gradio"] = saved


class TestCreateApp:
    def test_returns_demo_object(self, web_handlers):
        """create_app() returns the Blocks demo object."""
        _gr, app, handlers = web_handlers
        assert app is not None

    def test_all_handlers_registered(self, web_handlers):
        """All 4 handler functions are registered via button.click()."""
        _gr, _app, handlers = web_handlers
        assert len(handlers) == 4
        for name in ("handle_parse", "handle_summary", "handle_index", "handle_question"):
            assert name in handlers
            assert callable(handlers[name])


# ── Test: handle_parse ───────────────────────────────────────────────────────

class TestHandleParse:
    def test_no_file_returns_prompt(self, web_handlers):
        """handle_parse returns a prompt when no file is uploaded."""
        _, _, h = web_handlers
        result = h["handle_parse"](None, False)
        assert "upload" in result[0].lower() or "Upload" in result[0]
        assert result[1] == ""
        assert result[2] == ""

    @patch("pdf_mate.parser.PDFParser")
    def test_success_returns_info(self, mock_cls, web_handlers):
        """handle_parse returns document info on success."""
        from pdf_mate.parser import PDFContent, TextBlock

        content = PDFContent(
            filename="test.pdf",
            page_count=3,
            text_blocks=[TextBlock(text="Hello world", page=0, bbox=(0, 0, 612, 792))],
            tables=[],
            images=[],
        )
        mock_cls.return_value.parse.return_value = content

        _, _, h = web_handlers
        info, text, md = h["handle_parse"]("/tmp/test.pdf", False)
        assert "test.pdf" in info
        assert "3" in info
        assert "Hello world" in text

    @patch("pdf_mate.parser.PDFParser")
    def test_pdf_mate_error_caught(self, mock_cls, web_handlers):
        """handle_parse catches PDFMateError and returns error message."""
        from pdf_mate.exceptions import ParseError

        mock_cls.return_value.parse.side_effect = ParseError("bad pdf")

        _, _, h = web_handlers
        info, text, md = h["handle_parse"]("/tmp/bad.pdf", True)
        assert "Error" in info
        assert "bad pdf" in info
        assert text == ""

    @patch("pdf_mate.parser.PDFParser")
    def test_unexpected_error_caught(self, mock_cls, web_handlers):
        """handle_parse catches unexpected exceptions."""
        mock_cls.return_value.parse.side_effect = RuntimeError("boom")

        _, _, h = web_handlers
        info, text, md = h["handle_parse"]("/tmp/err.pdf", False)
        assert "Error" in info
        assert "boom" in info

    @patch("pdf_mate.parser.PDFParser")
    def test_extract_images_flag(self, mock_cls, web_handlers):
        """handle_parse passes extract_images to PDFParser."""
        from pdf_mate.parser import PDFContent, TextBlock

        content = PDFContent(filename="img.pdf", page_count=1,
                             text_blocks=[TextBlock(text="t", page=0, bbox=(0, 0, 1, 1))])
        mock_cls.return_value.parse.return_value = content

        _, _, h = web_handlers
        h["handle_parse"]("/tmp/img.pdf", True)
        mock_cls.assert_called_with(extract_images=True)


# ── Test: handle_summary ─────────────────────────────────────────────────────

class TestHandleSummary:
    def test_no_file_returns_prompt(self, web_handlers):
        """handle_summary returns a prompt when no file is uploaded."""
        _, _, h = web_handlers
        result = h["handle_summary"](None, "gpt-4o-mini", "concise", "", "")
        assert "upload" in result[0].lower() or "Upload" in result[0]
        assert result[1] == ""

    @patch("pdf_mate.summary.DocumentSummarizer")
    @patch("pdf_mate.parser.PDFParser")
    def test_success_returns_summary(self, mock_parser_cls, mock_sum_cls, web_handlers):
        """handle_summary returns formatted title + summary."""
        from pdf_mate.parser import PDFContent, TextBlock
        from pdf_mate.summary import DocumentSummary

        content = PDFContent(
            filename="report.pdf", page_count=5,
            text_blocks=[TextBlock(text="Some important text.", page=0, bbox=(0, 0, 612, 792))],
        )
        mock_parser_cls.return_value.parse.return_value = content

        mock_sum_cls.return_value.summarize.return_value = DocumentSummary(
            title="Report Title",
            summary="This is a summary.",
            key_points=["Point A", "Point B"],
            page_count=5,
            word_count=100,
        )

        _, _, h = web_handlers
        title, body = h["handle_summary"]("/tmp/report.pdf", "gpt-4o", "detailed", "sk-xxx", "")
        assert "Report Title" in title
        assert "summary" in body.lower() or "Summary" in body
        assert "Point A" in body

    @patch("pdf_mate.parser.PDFParser")
    def test_empty_text_returns_no_text(self, mock_cls, web_handlers):
        """handle_summary returns 'No text' when PDF has no text."""
        from pdf_mate.parser import PDFContent

        content = PDFContent(filename="blank.pdf", page_count=1, text_blocks=[])
        mock_cls.return_value.parse.return_value = content

        _, _, h = web_handlers
        msg, body = h["handle_summary"]("/tmp/blank.pdf", "gpt-4o-mini", "concise", "", "")
        assert "No text" in msg

    @patch("pdf_mate.parser.PDFParser")
    def test_pdf_mate_error_caught(self, mock_cls, web_handlers):
        """handle_summary catches PDFMateError."""
        from pdf_mate.exceptions import LLMError

        mock_cls.return_value.parse.side_effect = LLMError("api down")

        _, _, h = web_handlers
        msg, body = h["handle_summary"]("/tmp/err.pdf", "gpt-4o-mini", "concise", "", "")
        assert "Error" in msg
        assert "api down" in msg

    @patch("pdf_mate.parser.PDFParser")
    def test_unexpected_error_caught(self, mock_cls, web_handlers):
        """handle_summary catches unexpected exceptions."""
        mock_cls.return_value.parse.side_effect = ValueError("unexpected")

        _, _, h = web_handlers
        msg, body = h["handle_summary"]("/tmp/err.pdf", "gpt-4o-mini", "concise", "", "")
        assert "Error" in msg


# ── Test: handle_index ───────────────────────────────────────────────────────

class TestHandleIndex:
    def test_no_file_returns_prompt(self, web_handlers):
        """handle_index returns a prompt when no file is uploaded."""
        _, _, h = web_handlers
        status, info, state = h["handle_index"](None, "gpt-4o-mini", "local", "", "", {})
        assert "upload" in status.lower() or "Upload" in status

    def test_already_indexed_returns_early(self, web_handlers):
        """handle_index short-circuits when file already indexed."""
        _, _, h = web_handlers
        rag_state = {"indexed": True, "_last_file": "/tmp/doc.pdf", "chunk_count": 10}
        status, info, state = h["handle_index"](
            "/tmp/doc.pdf", "gpt-4o-mini", "local", "", "", rag_state
        )
        assert "already indexed" in status.lower()
        assert "10" in info

    @patch("pdf_mate.rag.RAGEngine")
    @patch("pdf_mate.parser.PDFParser")
    def test_success_indexes_document(self, mock_parser_cls, mock_rag_cls, web_handlers):
        """handle_index parses and indexes the document."""
        from pdf_mate.parser import PDFContent, TextBlock

        content = PDFContent(
            filename="doc.pdf", page_count=2,
            text_blocks=[TextBlock(text="Content here.", page=0, bbox=(0, 0, 612, 792))],
        )
        mock_parser_cls.return_value.parse.return_value = content
        mock_rag_cls.return_value.index_document.return_value = 7

        _, _, h = web_handlers
        status, info, state = h["handle_index"](
            "/tmp/doc.pdf", "gpt-4o-mini", "local", "", "", {}
        )
        assert "success" in status.lower()
        assert "7" in info
        assert state["indexed"] is True
        assert state["chunk_count"] == 7

    @patch("pdf_mate.parser.PDFParser")
    def test_empty_text_returns_no_text(self, mock_cls, web_handlers):
        """handle_index returns 'No text' when document has no text."""
        from pdf_mate.parser import PDFContent

        content = PDFContent(filename="empty.pdf", page_count=1, text_blocks=[])
        mock_cls.return_value.parse.return_value = content

        _, _, h = web_handlers
        status, info, state = h["handle_index"](
            "/tmp/empty.pdf", "gpt-4o-mini", "local", "", "", {}
        )
        assert "No text" in status

    @patch("pdf_mate.parser.PDFParser")
    def test_pdf_mate_error_caught(self, mock_cls, web_handlers):
        """handle_index catches PDFMateError."""
        from pdf_mate.exceptions import RAGError

        mock_cls.return_value.parse.side_effect = RAGError("rag error")

        _, _, h = web_handlers
        status, info, state = h["handle_index"](
            "/tmp/err.pdf", "gpt-4o-mini", "local", "", "", {}
        )
        assert "Error" in status

    @patch("pdf_mate.parser.PDFParser")
    def test_unexpected_error_caught(self, mock_cls, web_handlers):
        """handle_index catches unexpected exceptions."""
        mock_cls.return_value.parse.side_effect = OSError("disk fail")

        _, _, h = web_handlers
        status, info, state = h["handle_index"](
            "/tmp/err.pdf", "gpt-4o-mini", "local", "", "", {}
        )
        assert "Error" in status
        assert "disk fail" in status


# ── Test: handle_question ────────────────────────────────────────────────────

class TestHandleQuestion:
    def test_empty_question_returns_prompt(self, web_handlers):
        """handle_question asks user to enter a question when input is empty."""
        _, _, h = web_handlers
        answer, state = h["handle_question"]("", {})
        assert "enter" in answer.lower() or "question" in answer.lower()

    def test_no_engine_returns_index_prompt(self, web_handlers):
        """handle_question tells user to index first when no engine is available."""
        _, _, h = web_handlers
        answer, state = h["handle_question"]("What is this?", {})
        assert "index" in answer.lower()

    def test_success_returns_answer(self, web_handlers):
        """handle_question returns formatted answer with sources."""
        from pdf_mate.rag import RAGAnswer

        mock_engine = MagicMock()
        mock_engine.query.return_value = RAGAnswer(
            question="What is ML?",
            answer="Machine learning is a branch of AI.",
            sources=[("doc.pdf", "ML is a branch..."), ("doc.pdf", "Another chunk")],
            score=0.95,
        )

        _, _, h = web_handlers
        rag_state = {"rag_engine": mock_engine}
        answer, state = h["handle_question"]("What is ML?", rag_state)
        assert "Machine learning" in answer
        assert "doc.pdf" in answer  # sources section

    def test_success_no_sources(self, web_handlers):
        """handle_question returns answer without sources section."""
        from pdf_mate.rag import RAGAnswer

        mock_engine = MagicMock()
        mock_engine.query.return_value = RAGAnswer(
            question="Q", answer="Just an answer.", sources=[], score=0.5,
        )

        _, _, h = web_handlers
        answer, state = h["handle_question"]("Q", {"rag_engine": mock_engine})
        assert "Just an answer." in answer
        assert "Sources" not in answer

    def test_pdf_mate_error_caught(self, web_handlers):
        """handle_question catches PDFMateError from engine.query()."""
        from pdf_mate.exceptions import RAGError

        mock_engine = MagicMock()
        mock_engine.query.side_effect = RAGError("search failed")

        _, _, h = web_handlers
        answer, state = h["handle_question"]("Q?", {"rag_engine": mock_engine})
        assert "Error" in answer

    def test_unexpected_error_caught(self, web_handlers):
        """handle_question catches unexpected exceptions."""
        mock_engine = MagicMock()
        mock_engine.query.side_effect = KeyError("missing key")

        _, _, h = web_handlers
        answer, state = h["handle_question"]("Q?", {"rag_engine": mock_engine})
        assert "Error" in answer


# ── Test: module-level attributes ────────────────────────────────────────────

class TestWebModule:
    def test_logger_defined(self):
        """web module has a logger."""
        # Ensure fresh import
        sys.modules.pop("pdf_mate.web", None)
        from pdf_mate import web
        assert hasattr(web, "logger")

    def test_pdfmateerror_imported(self):
        """web module imports PDFMateError."""
        sys.modules.pop("pdf_mate.web", None)
        from pdf_mate import web
        assert hasattr(web, "PDFMateError")

    def test_create_app_is_callable(self):
        """create_app is the public entry point."""
        sys.modules.pop("pdf_mate.web", None)
        from pdf_mate.web import create_app
        assert callable(create_app)
