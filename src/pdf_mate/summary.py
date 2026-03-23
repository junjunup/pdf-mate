"""Summary module: generate intelligent document summaries."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .exceptions import LLMError
from .llm import LLMBackend, Message, create_llm_backend

logger = logging.getLogger(__name__)


@dataclass
class SummaryConfig:
    """Configuration for document summarization."""

    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    max_summary_length: int = 500
    style: str = "concise"  # "concise", "detailed", "bullets"


@dataclass
class DocumentSummary:
    """A generated document summary."""

    title: str
    summary: str
    key_points: list[str]
    page_count: int
    word_count: int


class DocumentSummarizer:
    """Generate AI-powered summaries for PDF documents."""

    def __init__(self, config: SummaryConfig | None = None):
        self.config = config or SummaryConfig()
        self._llm: LLMBackend | None = None

    def _get_llm(self) -> LLMBackend:
        if self._llm is None:
            self._llm = create_llm_backend(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
            )
        return self._llm

    def summarize(
        self,
        text: str,
        filename: str = "document.pdf",
        page_count: int = 0,
    ) -> DocumentSummary:
        """Generate a summary for a document.

        Args:
            text: Full document text.
            filename: Source filename (used for title generation).
            page_count: Number of pages in the document.

        Returns:
            DocumentSummary with summary and key points.

        Raises:
            ValueError: If *text* is empty or whitespace-only.
            LLMError: If any LLM call fails.
        """
        if not text.strip():
            raise ValueError("Cannot summarize empty text")

        try:
            llm = self._get_llm()

            # Truncate text if too long (roughly ~128k tokens for most models)
            max_chars = 120000
            if len(text) > max_chars:
                logger.warning(
                    "Text length (%d chars) exceeds max (%d chars), truncating. "
                    "Some content will not be included in the summary.",
                    len(text),
                    max_chars,
                )
            truncated_text = text[:max_chars] if len(text) > max_chars else text
            word_count = len(truncated_text.split())

            # Generate title
            title_messages = [
                Message(
                    role="system",
                    content="Generate a concise, descriptive title for this document. "
                    "Return ONLY the title, nothing else. Maximum 10 words.",
                ),
                Message(
                    role="user",
                    content=f"Document filename: {filename}\n\n"
                    f"First 2000 characters:\n{truncated_text[:2000]}",
                ),
            ]
            title = llm.chat(title_messages).strip().strip('"').strip("'")

            # Generate summary
            style_instruction = {
                "concise": "Provide a concise summary in 2-3 paragraphs.",
                "detailed": "Provide a detailed summary covering all major sections.",
                "bullets": "Provide a summary using bullet points for key information.",
            }.get(self.config.style, "Provide a concise summary in 2-3 paragraphs.")

            summary_messages = [
                Message(
                    role="system",
                    content=(
                        f"You are an expert document analyst. {style_instruction}\n"
                        f"Keep the summary under {self.config.max_summary_length} words. "
                        "Focus on the main arguments, findings, and conclusions."
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        "Please summarize the following document:\n\n"
                        f"{truncated_text}"
                    ),
                ),
            ]
            summary = llm.chat(summary_messages)

            # Extract key points
            key_points_messages = [
                Message(
                    role="system",
                    content=(
                        "Extract the 5-8 most important key points from this document. "
                        "Return them as a numbered list. Each point should be one sentence. "
                        "Return ONLY the numbered list, nothing else."
                    ),
                ),
                Message(
                    role="user",
                    content=f"Document:\n\n{truncated_text}",
                ),
            ]
            key_points_text = llm.chat(key_points_messages)
            key_points = [
                line.strip().lstrip("0123456789.-) ").strip()
                for line in key_points_text.strip().split("\n")
                if line.strip()
            ]

            return DocumentSummary(
                title=title,
                summary=summary,
                key_points=key_points,
                page_count=page_count,
                word_count=word_count,
            )
        except (LLMError, ValueError):
            raise
        except Exception as exc:
            raise LLMError(f"Summarization failed: {exc}") from exc
