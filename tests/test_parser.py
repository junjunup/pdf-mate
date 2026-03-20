"""Tests for PDF parser module."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from pdf_mate.parser import PDFParser, PDFContent, TextBlock


class TestTextBlock:
    def test_creation(self):
        block = TextBlock(text="Hello", page=0, bbox=(0, 0, 612, 792))
        assert block.text == "Hello"
        assert block.page == 0
        assert block.bbox == (0, 0, 612, 792)


class TestTable:
    def test_creation(self):
        table = Table(
            rows=[["Name", "Age"], ["Alice", "30"]],
            page=0,
            bbox=(0, 0, 612, 792),
        )
        assert len(table.rows) == 2
        assert table.rows[0][0] == "Name"


class TestPDFContent:
    def test_empty_content(self):
        content = PDFContent(filename="test.pdf", page_count=0)
        assert content.full_text == ""
        assert content.markdown_text == "# test.pdf\n"

    def test_full_text(self):
        content = PDFContent(
            filename="test.pdf",
            page_count=2,
            text_blocks=[
                TextBlock(text="Page 1 content", page=0, bbox=(0, 0, 612, 792)),
                TextBlock(text="Page 2 content", page=1, bbox=(0, 0, 612, 792)),
            ],
        )
        assert "Page 1 content" in content.full_text
        assert "Page 2 content" in content.full_text

    def test_markdown_text(self):
        content = PDFContent(
            filename="test.pdf",
            page_count=1,
            text_blocks=[
                TextBlock(text="Some content here", page=0, bbox=(0, 0, 612, 792)),
            ],
            tables=[
                Table(
                    rows=[["Col1", "Col2"], ["a", "b"]],
                    page=0,
                    bbox=(0, 0, 612, 792),
                ),
            ],
        )
        md = content.markdown_text
        assert "# test.pdf" in md
        assert "Some content here" in md
        assert "| Col1 | Col2 |" in md


class TestPDFParser:
    def test_init_defaults(self):
        parser = PDFParser()
        assert parser.extract_images is False
        assert parser.dpi == 300

    def test_parse_file_not_found(self):
        parser = PDFParser()
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            parser.parse("nonexistent.pdf")

    @patch("pdf_mate.parser.pdfplumber")
    @patch("pdf_mate.parser.fitz")
    def test_parse_empty_pdf(self, mock_fitz, mock_pdfplumber):
        # Mock pdfplumber
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_page.extract_tables.return_value = []
        mock_page.width = 612
        mock_page.height = 792
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_pdfplumber.open.return_value = mock_pdf

        # Mock PyMuPDF
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = mock_doc

        parser = PDFParser()
        # Create a temp file to avoid file check
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_path = f.name

        try:
            content = parser.parse(temp_path)
            assert content.page_count == 1
            assert len(content.text_blocks) == 0
        finally:
            import os
            os.unlink(temp_path)
