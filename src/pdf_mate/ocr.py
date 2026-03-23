"""OCR module: extract text from scanned PDF pages and images."""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


class OCREngine:
    """OCR engine for extracting text from scanned documents.

    Supports Tesseract OCR via pytesseract and layout-aware extraction.
    """

    def __init__(
        self,
        language: str = "eng+chi_sim",
        dpi: int = 300,
        tessdata_path: str | None = None,
    ):
        """Initialize the OCR engine.

        Args:
            language: Tesseract language code (e.g., 'eng', 'chi_sim', 'eng+chi_sim').
            dpi: DPI for rendering PDF pages to images.
            tessdata_path: Custom path to Tesseract tessdata directory.
        """
        self.language = language
        self.dpi = dpi
        self.tessdata_path = tessdata_path
        self._tesseract = None

    def _get_tesseract(self):
        """Lazy import and configure pytesseract."""
        if self._tesseract is None:
            try:
                import pytesseract
                from PIL import Image

                self._tesseract = pytesseract
                self._pil_image = Image

                if self.tessdata_path:
                    pytesseract.pytesseract.tessdata_dir_prefix = self.tessdata_path
            except ImportError:
                raise ImportError(
                    "pytesseract and Pillow are required for OCR. "
                    "Install them with: pip install pdf-mate[ocr]"
                )
        return self._tesseract, self._pil_image

    def extract_text_from_pdf(
        self, file_path: str | Path, page_numbers: list[int] | None = None
    ) -> list[tuple[int, str]]:
        """Extract text from a PDF using OCR.

        Args:
            file_path: Path to the PDF file.
            page_numbers: List of page numbers to process (None = all pages).

        Returns:
            List of (page_number, extracted_text) tuples.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        tess, PILImage = self._get_tesseract()
        results: list[tuple[int, str]] = []

        with fitz.open(str(file_path)) as doc:
            pages = range(len(doc)) if page_numbers is None else page_numbers
            for page_num in pages:
                page = doc[page_num]
                pix = page.get_pixmap(dpi=self.dpi)
                img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)

                text = tess.image_to_string(img, lang=self.language).strip()
                if text:
                    results.append((page_num, text))

        return results

    def extract_text_from_image(self, image_path: str | Path) -> str:
        """Extract text from an image file.

        Args:
            image_path: Path to the image file.

        Returns:
            Extracted text string.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        tess, PILImage = self._get_tesseract()
        img = PILImage.open(str(image_path))
        return tess.image_to_string(img, lang=self.language).strip()

    def detect_language(self, file_path: str | Path, page_num: int = 0) -> str:
        """Detect the primary language of a PDF page.

        Returns a suggested Tesseract language code.
        """
        text_results = self.extract_text_from_pdf(file_path, [page_num])
        if not text_results:
            return "eng"

        # Simple heuristic: check for CJK characters
        text = text_results[0][1]
        cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        total_alpha = sum(1 for c in text if c.isalpha())

        if total_alpha == 0:
            return "eng"

        if cjk_count / total_alpha > 0.3:
            return "chi_sim"
        return "eng"
