"""PDF parsing module: extract text, tables, and images from PDF files."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

from .exceptions import ParseError

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """A contiguous block of text extracted from a PDF page."""

    text: str
    page: int
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1


@dataclass
class Table:
    """A table extracted from a PDF page."""

    rows: list[list[str]]
    page: int
    bbox: tuple[float, float, float, float]


@dataclass
class ImageInfo:
    """Image metadata extracted from a PDF page."""

    page: int
    index: int
    bbox: tuple[float, float, float, float]
    width: int
    height: int
    image_bytes: bytes = b""


@dataclass
class PDFContent:
    """Complete parsed content of a PDF document."""

    filename: str
    page_count: int
    text_blocks: list[TextBlock] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    images: list[ImageInfo] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Return all text blocks concatenated with page separators."""
        parts: list[str] = []
        for block in self.text_blocks:
            parts.append(block.text)
        return "\n\n".join(parts)

    @property
    def markdown_text(self) -> str:
        """Return full text formatted with page headers."""
        lines: list[str] = []
        lines.append(f"# {self.filename}\n")
        for block in self.text_blocks:
            lines.append(f"## Page {block.page + 1}\n")
            lines.append(block.text)
        if self.tables:
            lines.append("\n## Tables\n")
            for table in self.tables:
                lines.append(f"### Table on page {table.page + 1}\n")
                for i, row in enumerate(table.rows):
                    lines.append("| " + " | ".join(row) + " |")
                    if i == 0:
                        lines.append("| " + " | ".join(["---"] * len(row)) + " |")
        return "\n".join(lines)


class PDFParser:
    """Parse PDF files to extract text, tables, and images."""

    def __init__(
        self,
        extract_images: bool = False,
        dpi: int = 300,
        max_image_size: int = 10 * 1024 * 1024,  # 10MB
    ):
        self.extract_images = extract_images
        self.dpi = dpi
        self.max_image_size = max_image_size

    def parse(self, file_path: str | Path) -> PDFContent:
        """Parse a PDF file and return all extracted content.

        Args:
            file_path: Path to the PDF file.

        Returns:
            PDFContent object with all extracted data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ParseError: If parsing fails for any other reason.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ParseError(f"PDF file not found: {file_path}")

        try:
            # Extract text, tables, and page count using pdfplumber
            text_blocks, tables, page_count = self._extract_text_and_tables(file_path)

            # Extract images using PyMuPDF (optional, graceful fallback)
            images: list[ImageInfo] = []
            if self.extract_images:
                try:
                    images = self._extract_images(file_path)
                except Exception:
                    logger.warning(
                        "Failed to extract images from '%s' via PyMuPDF, "
                        "skipping image extraction.",
                        file_path.name,
                        exc_info=True,
                    )
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError(f"Failed to parse PDF '{file_path.name}': {exc}") from exc

        return PDFContent(
            filename=file_path.name,
            page_count=page_count,
            text_blocks=text_blocks,
            tables=tables,
            images=images,
        )

    def _extract_text_and_tables(
        self, file_path: Path
    ) -> tuple[list[TextBlock], list[Table], int]:
        """Extract text blocks, tables, and page count using pdfplumber."""
        text_blocks: list[TextBlock] = []
        tables: list[Table] = []

        with pdfplumber.open(str(file_path)) as pdf:
            page_count = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages):
                # Extract text
                text = page.extract_text()
                if text and text.strip():
                    text_blocks.append(
                        TextBlock(
                            text=text.strip(),
                            page=page_num,
                            bbox=(0, 0, page.width, page.height),
                        )
                    )

                # Extract tables
                extracted_tables = page.extract_tables()
                if extracted_tables:
                    for table_data in extracted_tables:
                        # Clean table data
                        cleaned_rows = []
                        for row in table_data:
                            cleaned_rows.append([
                                cell.strip() if cell else ""
                                for cell in row
                            ])
                        if cleaned_rows:
                            tables.append(
                                Table(
                                    rows=cleaned_rows,
                                    page=page_num,
                                    bbox=(0, 0, page.width, page.height),
                                )
                            )

        return text_blocks, tables, page_count

    def _extract_images(self, file_path: Path) -> list[ImageInfo]:
        """Extract embedded images from PDF using PyMuPDF."""
        images: list[ImageInfo] = []

        with fitz.open(str(file_path)) as doc:
            img_index = 0
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)

                for img_info in image_list:
                    xref = img_info[0]
                    try:
                        base_image = doc.extract_image(xref)
                        if (
                            base_image
                            and len(base_image["image"]) <= self.max_image_size
                        ):
                            images.append(
                                ImageInfo(
                                    page=page_num,
                                    index=img_index,
                                    bbox=(0, 0, 0, 0),
                                    width=base_image.get("width", 0),
                                    height=base_image.get("height", 0),
                                    image_bytes=base_image["image"],
                                )
                            )
                            img_index += 1
                    except Exception:
                        logger.debug(
                            "Failed to extract image xref=%s on page %d",
                            xref, page_num, exc_info=True,
                        )
                        continue

        return images

    def extract_page_images(
        self, file_path: str | Path, page_num: int
    ) -> list[bytes]:
        """Render a specific page as image bytes (for OCR or preview).

        Args:
            file_path: Path to the PDF file.
            page_num: Zero-based page number.

        Returns:
            List of image bytes (PNG format).

        Raises:
            FileNotFoundError: If the file does not exist.
            IndexError: If ``page_num`` is out of range.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ParseError(f"PDF file not found: {file_path}")

        with fitz.open(str(file_path)) as doc:
            if page_num < 0 or page_num >= len(doc):
                raise IndexError(
                    f"Page {page_num} out of range (document has {len(doc)} pages)"
                )
            pix = doc[page_num].get_pixmap(dpi=self.dpi)
            return [pix.tobytes("png")]
