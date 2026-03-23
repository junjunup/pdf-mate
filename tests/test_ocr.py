"""Tests for the OCR module."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

# Mock fitz before any pdf_mate import
if "fitz" not in sys.modules:
    sys.modules["fitz"] = MagicMock()

import pytest

from pdf_mate.exceptions import OCRError
from pdf_mate.ocr import OCREngine


class TestOCREngine:
    def test_init_defaults(self):
        engine = OCREngine()
        assert engine.language == "eng+chi_sim"
        assert engine.dpi == 300
        assert engine.tessdata_path is None

    def test_init_custom(self):
        engine = OCREngine(language="chi_sim", dpi=150, tessdata_path="/usr/share/tessdata")
        assert engine.language == "chi_sim"
        assert engine.dpi == 150
        assert engine.tessdata_path == "/usr/share/tessdata"

    def test_tesseract_not_installed(self):
        """Test that OCRError is raised when pytesseract is not installed."""
        engine = OCREngine()
        # Force _tesseract to be None so _get_tesseract tries import
        engine._tesseract = None

        with patch.dict(sys.modules, {"pytesseract": None}):
            with patch("builtins.__import__", side_effect=ImportError("No pytesseract")):
                with pytest.raises(OCRError, match="pytesseract"):
                    engine._get_tesseract()

    def test_extract_text_file_not_found(self):
        engine = OCREngine()
        with pytest.raises(OCRError, match="PDF file not found"):
            engine.extract_text_from_pdf("nonexistent_file.pdf")

    def test_extract_image_file_not_found(self):
        engine = OCREngine()
        with pytest.raises(OCRError, match="Image file not found"):
            engine.extract_text_from_image("nonexistent_image.png")

    @patch("pdf_mate.ocr.fitz")
    def test_extract_text_from_pdf(self, mock_fitz):
        """Test OCR extraction from a PDF with mocked dependencies."""
        # Setup mock PDF document
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.width = 100
        mock_pix.height = 100
        mock_pix.samples = b"\x00" * (100 * 100 * 3)
        mock_page.get_pixmap.return_value = mock_pix
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = mock_doc

        # Setup mock pytesseract
        mock_tess = MagicMock()
        mock_tess.image_to_string.return_value = "Extracted text from OCR"
        mock_pil = MagicMock()
        mock_img = MagicMock()
        mock_pil.frombytes.return_value = mock_img

        engine = OCREngine()
        engine._tesseract = mock_tess
        engine._pil_image = mock_pil

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 dummy")
            f.flush()
            results = engine.extract_text_from_pdf(f.name)

        assert len(results) == 1
        assert results[0][0] == 0  # page number
        assert results[0][1] == "Extracted text from OCR"

    @patch("pdf_mate.ocr.fitz")
    def test_extract_text_specific_pages(self, mock_fitz):
        """Test OCR on specific pages only."""
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.width = 50
        mock_pix.height = 50
        mock_pix.samples = b"\x00" * (50 * 50 * 3)
        mock_page.get_pixmap.return_value = mock_pix
        mock_doc.__len__ = MagicMock(return_value=5)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = mock_doc

        mock_tess = MagicMock()
        mock_tess.image_to_string.return_value = "Page text"
        mock_pil = MagicMock()
        mock_pil.frombytes.return_value = MagicMock()

        engine = OCREngine()
        engine._tesseract = mock_tess
        engine._pil_image = mock_pil

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 dummy")
            f.flush()
            results = engine.extract_text_from_pdf(f.name, page_numbers=[1, 3])

        assert len(results) == 2

    def test_detect_language_cjk(self):
        """Test language detection for CJK text."""
        engine = OCREngine()

        with patch.object(engine, "extract_text_from_pdf") as mock_extract:
            # Simulate CJK-heavy text
            mock_extract.return_value = [(0, "这是一段中文测试文本")]
            lang = engine.detect_language("dummy.pdf")
            assert lang == "chi_sim"

    def test_detect_language_english(self):
        """Test language detection for English text."""
        engine = OCREngine()

        with patch.object(engine, "extract_text_from_pdf") as mock_extract:
            mock_extract.return_value = [(0, "This is English text for testing")]
            lang = engine.detect_language("dummy.pdf")
            assert lang == "eng"

    def test_detect_language_empty(self):
        """Test language detection with no text."""
        engine = OCREngine()

        with patch.object(engine, "extract_text_from_pdf") as mock_extract:
            mock_extract.return_value = []
            lang = engine.detect_language("dummy.pdf")
            assert lang == "eng"
