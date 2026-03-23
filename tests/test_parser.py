"""Tests for PDF parser module."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

# Mock fitz (PyMuPDF) before importing the parser module
# This is necessary because fitz may not be available in the test environment
# (e.g., Python 3.14 where the C extension DLL cannot load).
if "fitz" not in sys.modules:
    sys.modules["fitz"] = MagicMock()

import pytest

from pdf_mate.exceptions import ParseError
from pdf_mate.parser import ImageInfo, PDFContent, PDFParser, Table, TextBlock


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


class TestImageInfo:
    def test_creation(self):
        img = ImageInfo(page=0, index=0, bbox=(0, 0, 100, 100), width=100, height=100)
        assert img.page == 0
        assert img.image_bytes == b""


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

    def test_markdown_text_with_tables(self):
        content = PDFContent(
            filename="test.pdf",
            page_count=1,
            text_blocks=[
                TextBlock(text="Some content", page=0, bbox=(0, 0, 612, 792)),
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
        assert "Some content" in md
        assert "| Col1 | Col2 |" in md
        assert "| --- | --- |" in md
        # Verify header separator is only after first row
        lines = md.split("\n")
        separator_count = sum(1 for line in lines if line.strip() == "| --- | --- |")
        assert separator_count == 1

    def test_markdown_text_duplicate_rows(self):
        """Verify that duplicate rows are rendered correctly (not confused with header)."""
        content = PDFContent(
            filename="test.pdf",
            page_count=1,
            tables=[
                Table(
                    rows=[["A", "B"], ["A", "B"], ["C", "D"]],
                    page=0,
                    bbox=(0, 0, 612, 792),
                ),
            ],
        )
        md = content.markdown_text
        # Should have separator only once (after first row)
        assert md.count("| --- | --- |") == 1
        # Should have "| A | B |" twice
        assert md.count("| A | B |") == 2


class TestPDFParser:
    def test_init_defaults(self):
        parser = PDFParser()
        assert parser.extract_images is False
        assert parser.dpi == 300
        assert parser.max_image_size == 10 * 1024 * 1024

    def test_parse_file_not_found(self):
        parser = PDFParser()
        with pytest.raises(ParseError, match="PDF file not found"):
            parser.parse("nonexistent.pdf")

    def test_extract_page_images_file_not_found(self):
        parser = PDFParser()
        with pytest.raises(ParseError, match="PDF file not found"):
            parser.extract_page_images("nonexistent.pdf", 0)

    @pytest.mark.skipif(
        isinstance(sys.modules.get("fitz"), MagicMock),
        reason="fitz is mocked; skipping integration-level parse test",
    )
    def test_parse_empty_pdf(self):
        """Integration test requiring real fitz - skipped when mocked."""
        pass

    def test_init_custom(self):
        parser = PDFParser(extract_images=True, dpi=150, max_image_size=5_000_000)
        assert parser.extract_images is True
        assert parser.dpi == 150
        assert parser.max_image_size == 5_000_000

    @patch("pdf_mate.parser.fitz")
    @patch("pdf_mate.parser.pdfplumber")
    def test_parse_success_with_mocks(self, mock_pdfplumber, mock_fitz, tmp_path):
        """Test parse() full success path with mock pdfplumber and fitz."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        # Mock pdfplumber
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Hello world from page 1"
        mock_page.extract_tables.return_value = [[["Col1", "Col2"], ["A", "B"]]]
        mock_page.width = 612
        mock_page.height = 792
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        # Mock fitz for page count
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = mock_doc

        parser = PDFParser()
        content = parser.parse(pdf_file)

        assert content.filename == "test.pdf"
        assert content.page_count == 1
        assert len(content.text_blocks) == 1
        assert content.text_blocks[0].text == "Hello world from page 1"
        assert len(content.tables) == 1
        assert content.tables[0].rows == [["Col1", "Col2"], ["A", "B"]]

    @patch("pdf_mate.parser.fitz")
    @patch("pdf_mate.parser.pdfplumber")
    def test_parse_with_images(self, mock_pdfplumber, mock_fitz, tmp_path):
        """Test parse() with image extraction enabled."""
        pdf_file = tmp_path / "with_images.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        # pdfplumber: no text, no tables
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Text"
        mock_page.extract_tables.return_value = []
        mock_page.width = 612
        mock_page.height = 792
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        # fitz: one image
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_fitz_page = MagicMock()
        mock_fitz_page.get_images.return_value = [(1, 0, 0, 0, 0, 0, 0)]
        mock_doc.__getitem__ = MagicMock(return_value=mock_fitz_page)
        mock_doc.extract_image.return_value = {
            "image": b"\x89PNG" + b"\x00" * 100,
            "width": 200,
            "height": 150,
        }
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = mock_doc

        parser = PDFParser(extract_images=True)
        content = parser.parse(pdf_file)

        assert len(content.images) == 1
        assert content.images[0].width == 200
        assert content.images[0].height == 150

    @patch("pdf_mate.parser.fitz")
    @patch("pdf_mate.parser.pdfplumber")
    def test_parse_handles_exception(self, mock_pdfplumber, mock_fitz, tmp_path):
        """Test that parse() wraps unexpected exceptions in ParseError."""
        from pdf_mate.exceptions import ParseError

        pdf_file = tmp_path / "bad.pdf"
        pdf_file.write_bytes(b"bad data")

        mock_pdfplumber.open.side_effect = RuntimeError("corrupt file")

        parser = PDFParser()
        with pytest.raises(ParseError, match="Failed to parse PDF"):
            parser.parse(pdf_file)

    @patch("pdf_mate.parser.fitz")
    @patch("pdf_mate.parser.pdfplumber")
    def test_extract_text_empty_page(self, mock_pdfplumber, mock_fitz, tmp_path):
        """Test parsing a page with no text."""
        pdf_file = tmp_path / "empty.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_page.extract_tables.return_value = []
        mock_page.width = 612
        mock_page.height = 792
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = mock_doc

        parser = PDFParser()
        content = parser.parse(pdf_file)
        assert content.text_blocks == []

    @patch("pdf_mate.parser.fitz")
    @patch("pdf_mate.parser.pdfplumber")
    def test_extract_images_skips_large_images(self, mock_pdfplumber, mock_fitz, tmp_path):
        """Test that images exceeding max_image_size are skipped."""
        pdf_file = tmp_path / "big_img.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Text"
        mock_page.extract_tables.return_value = []
        mock_page.width = 612
        mock_page.height = 792
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_fitz_page = MagicMock()
        mock_fitz_page.get_images.return_value = [(1, 0, 0, 0, 0, 0, 0)]
        mock_doc.__getitem__ = MagicMock(return_value=mock_fitz_page)
        # Image exceeds max_image_size (1 byte limit)
        mock_doc.extract_image.return_value = {
            "image": b"\x00" * 100,
            "width": 50,
            "height": 50,
        }
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = mock_doc

        parser = PDFParser(extract_images=True, max_image_size=10)  # very small limit
        content = parser.parse(pdf_file)
        assert len(content.images) == 0  # image was too large

    @patch("pdf_mate.parser.fitz")
    @patch("pdf_mate.parser.pdfplumber")
    def test_extract_images_handles_exception(self, mock_pdfplumber, mock_fitz, tmp_path):
        """Test that image extraction errors are logged and skipped."""
        pdf_file = tmp_path / "err_img.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Text"
        mock_page.extract_tables.return_value = []
        mock_page.width = 612
        mock_page.height = 792
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_fitz_page = MagicMock()
        mock_fitz_page.get_images.return_value = [(1, 0, 0, 0, 0, 0, 0)]
        mock_doc.__getitem__ = MagicMock(return_value=mock_fitz_page)
        mock_doc.extract_image.side_effect = RuntimeError("extraction failed")
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = mock_doc

        parser = PDFParser(extract_images=True)
        content = parser.parse(pdf_file)
        assert len(content.images) == 0  # error was caught and skipped

    @patch("pdf_mate.parser.fitz")
    def test_extract_page_images_success(self, mock_fitz, tmp_path):
        """Test extract_page_images returns PNG bytes."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"\x89PNG image bytes"
        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=3)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = mock_doc

        parser = PDFParser()
        result = parser.extract_page_images(pdf_file, 0)
        assert len(result) == 1
        assert result[0] == b"\x89PNG image bytes"

    @patch("pdf_mate.parser.fitz")
    def test_extract_page_images_out_of_range(self, mock_fitz, tmp_path):
        """Test extract_page_images raises IndexError for invalid page."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=3)
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = mock_doc

        parser = PDFParser()
        with pytest.raises(IndexError, match="out of range"):
            parser.extract_page_images(pdf_file, 10)

    @patch("pdf_mate.parser.fitz")
    @patch("pdf_mate.parser.pdfplumber")
    def test_table_cell_none_handling(self, mock_pdfplumber, mock_fitz, tmp_path):
        """Test that None table cells are converted to empty string."""
        pdf_file = tmp_path / "null_cells.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Text"
        mock_page.extract_tables.return_value = [[["Header", None], ["Val", ""]]]
        mock_page.width = 612
        mock_page.height = 792
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = mock_doc

        parser = PDFParser()
        content = parser.parse(pdf_file)
        assert content.tables[0].rows[0][1] == ""  # None converted to ""
