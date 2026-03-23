"""Tests for the CLI module."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

# Mock fitz before any pdf_mate import
if "fitz" not in sys.modules:
    sys.modules["fitz"] = MagicMock()

from typer.testing import CliRunner

from pdf_mate.cli import app

runner = CliRunner()


class TestVersion:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "pdf-mate" in result.stdout

    def test_short_version_flag(self):
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert "pdf-mate" in result.stdout


class TestNoArgs:
    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        # no_args_is_help=True causes exit code 0 on some typer versions, 2 on others
        assert result.exit_code in (0, 2)
        # Should show some form of usage/help text
        out = result.stdout
        assert "pdf-mate" in out.lower() or "Usage" in out or "parse" in out


class TestParseCommand:
    def test_parse_nonexistent_file(self):
        result = runner.invoke(app, ["parse", "nonexistent_file.pdf"])
        assert result.exit_code != 0

    @patch("pdf_mate.parser.PDFParser", autospec=True)
    def test_parse_success(self, mock_parser_cls, tmp_path):
        """Test successful PDF parsing via CLI."""
        from pdf_mate.parser import PDFContent, TextBlock

        # Create a dummy file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        mock_content = PDFContent(
            filename="test.pdf",
            page_count=2,
            text_blocks=[TextBlock(text="Hello world", page=0, bbox=(0, 0, 612, 792))],
            tables=[],
            images=[],
        )
        mock_parser_cls.return_value.parse.return_value = mock_content

        result = runner.invoke(app, ["parse", str(pdf_file)])
        assert result.exit_code == 0
        assert "test.pdf" in result.stdout
        assert "2" in result.stdout  # page count

    @patch("pdf_mate.parser.PDFParser", autospec=True)
    def test_parse_markdown_format(self, mock_parser_cls, tmp_path):
        """Test markdown output format."""
        from pdf_mate.parser import PDFContent, TextBlock

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        mock_content = PDFContent(
            filename="test.pdf",
            page_count=1,
            text_blocks=[TextBlock(text="Content here", page=0, bbox=(0, 0, 612, 792))],
        )
        mock_parser_cls.return_value.parse.return_value = mock_content

        result = runner.invoke(app, ["parse", str(pdf_file), "--format", "markdown"])
        assert result.exit_code == 0

    @patch("pdf_mate.parser.PDFParser", autospec=True)
    def test_parse_save_output(self, mock_parser_cls, tmp_path):
        """Test saving output to file."""
        from pdf_mate.parser import PDFContent, TextBlock

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")
        out_file = tmp_path / "output.txt"

        mock_content = PDFContent(
            filename="test.pdf",
            page_count=1,
            text_blocks=[TextBlock(text="Saved text", page=0, bbox=(0, 0, 612, 792))],
        )
        mock_parser_cls.return_value.parse.return_value = mock_content

        result = runner.invoke(app, ["parse", str(pdf_file), "-o", str(out_file)])
        assert result.exit_code == 0
        assert out_file.exists()
        assert "Saved text" in out_file.read_text(encoding="utf-8")

    @patch("pdf_mate.parser.PDFParser", autospec=True)
    def test_parse_handles_error(self, mock_parser_cls, tmp_path):
        """Test that parse errors are handled gracefully."""
        from pdf_mate.exceptions import ParseError

        pdf_file = tmp_path / "broken.pdf"
        pdf_file.write_bytes(b"not a pdf")

        mock_parser_cls.return_value.parse.side_effect = ParseError("Corrupted PDF")

        result = runner.invoke(app, ["parse", str(pdf_file)])
        assert result.exit_code == 1
        assert "Error" in result.stdout


class TestSummaryCommand:
    @patch("pdf_mate.summary.DocumentSummarizer", autospec=True)
    @patch("pdf_mate.parser.PDFParser", autospec=True)
    def test_summary_no_text(self, mock_parser_cls, mock_summarizer_cls, tmp_path):
        """Test summary with empty PDF."""
        from pdf_mate.parser import PDFContent

        pdf_file = tmp_path / "empty.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        mock_content = PDFContent(filename="empty.pdf", page_count=1)
        mock_parser_cls.return_value.parse.return_value = mock_content

        result = runner.invoke(app, ["summary", str(pdf_file)])
        assert result.exit_code == 1
        assert "No text found" in result.stdout

    @patch("pdf_mate.summary.DocumentSummarizer", autospec=True)
    @patch("pdf_mate.parser.PDFParser", autospec=True)
    def test_summary_success(self, mock_parser_cls, mock_summarizer_cls, tmp_path):
        """Test summary with valid PDF content."""
        from pdf_mate.parser import PDFContent, TextBlock
        from pdf_mate.summary import DocumentSummary

        pdf_file = tmp_path / "report.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        mock_content = PDFContent(
            filename="report.pdf",
            page_count=5,
            text_blocks=[
                TextBlock(text="Machine learning is great.", page=0, bbox=(0, 0, 612, 792))
            ],
        )
        mock_parser_cls.return_value.parse.return_value = mock_content

        mock_summary = DocumentSummary(
            title="ML Report",
            summary="A report about ML.",
            key_points=["ML is great", "Deep learning works"],
            page_count=5,
            word_count=200,
        )
        mock_summarizer_cls.return_value.summarize.return_value = mock_summary

        result = runner.invoke(app, ["summary", str(pdf_file)])
        assert result.exit_code == 0
        assert "ML Report" in result.stdout

    @patch("pdf_mate.summary.DocumentSummarizer", autospec=True)
    @patch("pdf_mate.parser.PDFParser", autospec=True)
    def test_summary_handles_error(self, mock_parser_cls, mock_summarizer_cls, tmp_path):
        """Test summary gracefully handles LLMError."""
        from pdf_mate.exceptions import LLMError
        from pdf_mate.parser import PDFContent, TextBlock

        pdf_file = tmp_path / "report.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        mock_content = PDFContent(
            filename="report.pdf",
            page_count=1,
            text_blocks=[TextBlock(text="Text here.", page=0, bbox=(0, 0, 612, 792))],
        )
        mock_parser_cls.return_value.parse.return_value = mock_content
        mock_summarizer_cls.return_value.summarize.side_effect = LLMError("API down")

        result = runner.invoke(app, ["summary", str(pdf_file)])
        assert result.exit_code == 1
        assert "Error" in result.stdout


class TestOCRCommand:
    @patch("pdf_mate.ocr.OCREngine", autospec=True)
    def test_ocr_success(self, mock_engine_cls, tmp_path):
        """Test OCR command success path."""
        pdf_file = tmp_path / "scanned.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        mock_engine_cls.return_value.extract_text_from_pdf.return_value = [
            (0, "OCR extracted text from page 1"),
        ]

        result = runner.invoke(app, ["ocr", str(pdf_file)])
        assert result.exit_code == 0
        assert "OCR extracted text" in result.stdout

    @patch("pdf_mate.ocr.OCREngine", autospec=True)
    def test_ocr_save_output(self, mock_engine_cls, tmp_path):
        """Test OCR command saves output to file."""
        pdf_file = tmp_path / "scanned.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")
        out_file = tmp_path / "ocr_output.txt"

        mock_engine_cls.return_value.extract_text_from_pdf.return_value = [
            (0, "Page 1 OCR text"),
            (1, "Page 2 OCR text"),
        ]

        result = runner.invoke(app, ["ocr", str(pdf_file), "-o", str(out_file)])
        assert result.exit_code == 0
        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert "Page 1 OCR text" in content

    @patch("pdf_mate.ocr.OCREngine", autospec=True)
    def test_ocr_handles_error(self, mock_engine_cls, tmp_path):
        """Test OCR command handles OCRError."""
        from pdf_mate.exceptions import OCRError

        pdf_file = tmp_path / "bad.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        mock_engine_cls.return_value.extract_text_from_pdf.side_effect = OCRError(
            "Tesseract not found"
        )

        result = runner.invoke(app, ["ocr", str(pdf_file)])
        assert result.exit_code == 1
        assert "Error" in result.stdout


class TestAskCommand:
    @patch("pdf_mate.rag.RAGEngine", autospec=True)
    @patch("pdf_mate.parser.PDFParser", autospec=True)
    def test_ask_no_text(self, mock_parser_cls, mock_rag_cls, tmp_path):
        """Test ask with empty PDF."""
        from pdf_mate.parser import PDFContent

        pdf_file = tmp_path / "empty.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        mock_content = PDFContent(filename="empty.pdf", page_count=1)
        mock_parser_cls.return_value.parse.return_value = mock_content

        result = runner.invoke(app, ["ask", str(pdf_file), "-q", "What?"])
        assert result.exit_code == 1

    @patch("pdf_mate.rag.RAGEngine", autospec=True)
    @patch("pdf_mate.parser.PDFParser", autospec=True)
    def test_ask_success(self, mock_parser_cls, mock_rag_cls, tmp_path):
        """Test ask command with valid content and single question."""
        from pdf_mate.parser import PDFContent, TextBlock
        from pdf_mate.rag import RAGAnswer

        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        mock_content = PDFContent(
            filename="doc.pdf",
            page_count=1,
            text_blocks=[TextBlock(text="AI is the future.", page=0, bbox=(0, 0, 612, 792))],
        )
        mock_parser_cls.return_value.parse.return_value = mock_content
        mock_rag_cls.return_value.index_document.return_value = 3

        mock_answer = RAGAnswer(
            question="What is this about?",
            answer="This document is about AI.",
            sources=[("doc.pdf", "AI is the future.")],
            score=0.85,
        )
        mock_rag_cls.return_value.query.return_value = mock_answer

        # -q flag processes one question then enters loop; provide "quit" as stdin
        result = runner.invoke(
            app,
            ["ask", str(pdf_file), "-q", "What is this about?"],
            input="quit\n",
        )
        assert result.exit_code == 0
        assert "AI" in result.stdout


class TestWebCommand:
    def test_web_default_host(self):
        """Verify the default host is 127.0.0.1 (not 0.0.0.0)."""
        # Inspect the command's parameter defaults
        import inspect

        # Check the function definition source for safe defaults
        import pdf_mate.cli as cli_module

        source = inspect.getsource(cli_module.launch_web)
        assert "127.0.0.1" in source
        assert "0.0.0.0" not in source
