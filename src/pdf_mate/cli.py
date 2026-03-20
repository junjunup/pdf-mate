"""CLI interface for pdf-mate."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import __version__

app = typer.Typer(
    name="pdf-mate",
    help="AI-powered PDF parsing, summarization, and conversational Q&A.",
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool):
    if value:
        console.print(f"pdf-mate v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, help="Show version."
    ),
):
    """pdf-mate: AI-powered PDF toolkit."""
    pass


# ─── Parse Command ────────────────────────────────────────────────────────────


@app.command(name="parse")
def parse_pdf(
    file_path: Path = typer.Argument(..., help="Path to the PDF file.", exists=True),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save extracted text to a file."
    ),
    format: str = typer.Option(
        "text", "--format", "-f", help="Output format: text, markdown."
    ),
    extract_images: bool = typer.Option(
        False, "--images", help="Extract embedded images."
    ),
):
    """Parse a PDF and extract text, tables, and images."""
    from .parser import PDFParser

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ):
        parser = PDFParser(extract_images=extract_images)
        content = parser.parse(file_path)

    # Display info
    info = Table(title="PDF Information")
    info.add_column("Property", style="cyan")
    info.add_column("Value", style="green")
    info.add_row("Filename", content.filename)
    info.add_row("Pages", str(content.page_count))
    info.add_row("Text blocks", str(len(content.text_blocks)))
    info.add_row("Tables", str(len(content.tables)))
    info.add_row("Images", str(len(content.images)))
    console.print(info)

    # Display content
    if format == "markdown":
        output_text = content.markdown_text
        console.print(Markdown(output_text))
    else:
        console.print(Panel(content.full_text[:5000], title="Extracted Text"))

    # Save to file
    if output:
        if format == "markdown":
            text = content.markdown_text
        else:
            text = content.full_text
        output.write_text(text, encoding="utf-8")
        console.print(f"\n[green]Text saved to {output}[/green]")


# ─── OCR Command ──────────────────────────────────────────────────────────────


@app.command(name="ocr")
def ocr_pdf(
    file_path: Path = typer.Argument(..., help="Path to the PDF file.", exists=True),
    language: str = typer.Option("eng+chi_sim", "--lang", "-l", help="OCR language."),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save OCR text to a file."
    ),
):
    """Extract text from scanned PDFs using OCR."""
    from .ocr import OCREngine

    console.print("[yellow]Starting OCR (this may take a while)...[/yellow]")

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ):
        engine = OCREngine(language=language)
        results = engine.extract_text_from_pdf(file_path)

    for page_num, text in results:
        console.print(Panel(text, title=f"Page {page_num + 1}"))

    if output:
        all_text = "\n\n".join(
            f"--- Page {num + 1} ---\n{text}" for num, text in results
        )
        output.write_text(all_text, encoding="utf-8")
        console.print(f"\n[green]OCR text saved to {output}[/green]")


# ─── Summary Command ──────────────────────────────────────────────────────────


@app.command(name="summary")
def summarize_pdf(
    file_path: Path = typer.Argument(..., help="Path to the PDF file.", exists=True),
    style: str = typer.Option(
        "concise", "--style", "-s", help="Summary style: concise, detailed, bullets."
    ),
    model: str = typer.Option("gpt-3.5-turbo", "--model", "-m", help="LLM model name."),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="LLM API key (or set OPENAI_API_KEY env var)."
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="Custom API base URL."
    ),
):
    """Generate an AI-powered summary of a PDF document."""
    import os

    from .parser import PDFParser
    from .rag import RAGConfig
    from .summary import SummaryConfig, DocumentSummarizer

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ):
        # Parse PDF
        console.print("[cyan]Parsing PDF...[/cyan]")
        parser = PDFParser()
        content = parser.parse(file_path)

        if not content.full_text.strip():
            console.print("[red]No text found in PDF. Try OCR mode for scanned documents.[/red]")
            raise typer.Exit(1)

        # Generate summary
        console.print("[cyan]Generating summary...[/cyan]")
        config = SummaryConfig(
            llm_model=model,
            llm_api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            llm_base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            style=style,
        )
        summarizer = DocumentSummarizer(config)
        doc_summary = summarizer.summarize(
            text=content.full_text,
            filename=content.filename,
            page_count=content.page_count,
        )

    # Display results
    console.print(Panel(doc_summary.title, title="Document Title", style="bold green"))
    console.print(Markdown(f"## Summary\n\n{doc_summary.summary}"))

    key_points_md = "\n".join(f"- {p}" for p in doc_summary.key_points)
    console.print(Markdown(f"## Key Points\n\n{key_points_md}"))

    stats = Table(title="Document Statistics")
    stats.add_column("Metric", style="cyan")
    stats.add_column("Value", style="green")
    stats.add_row("Pages", str(doc_summary.page_count))
    stats.add_row("Words", str(doc_summary.word_count))
    stats.add_row("Key Points", str(len(doc_summary.key_points)))
    console.print(stats)


# ─── Ask Command ──────────────────────────────────────────────────────────────


@app.command(name="ask")
def ask_question(
    file_path: Path = typer.Argument(..., help="Path to the PDF file.", exists=True),
    question: str = typer.Option(
        None, "--question", "-q", help="Question to ask (omit for interactive mode)."
    ),
    model: str = typer.Option("gpt-3.5-turbo", "--model", "-m", help="LLM model name."),
    embedding: str = typer.Option(
        "local", "--embedding", "-e", help="Embedding provider: local, openai."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="LLM API key (or set OPENAI_API_KEY env var)."
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="Custom API base URL."
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of context chunks."),
):
    """Ask questions about a PDF document using RAG."""
    import os

    from .parser import PDFParser
    from .rag import RAGConfig, RAGEngine

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ):
        # Parse PDF
        console.print("[cyan]Parsing and indexing PDF...[/cyan]")
        parser = PDFParser()
        content = parser.parse(file_path)

        if not content.full_text.strip():
            console.print("[red]No text found in PDF. Try OCR mode for scanned documents.[/red]")
            raise typer.Exit(1)

        # Setup RAG
        config = RAGConfig(
            llm_model=model,
            llm_api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            llm_base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            embedding_provider=embedding,
            retrieval_top_k=top_k,
        )
        engine = RAGEngine(config)
        engine.index_document(
            text=content.full_text,
            source_name=content.filename,
        )
        console.print(f"[green]Indexed {engine._store.count} chunks.[/green]")

    # Interactive Q&A
    console.print("\n[bold cyan]PDF Q&A Mode[/bold cyan]")
    console.print("Ask questions about the document. Type 'quit' to exit.\n")

    while True:
        q = question or typer.prompt("Ask a question")
        if question:
            question = None  # Only use once

        if q.lower() in ("quit", "exit", "q"):
            break

        if not q.strip():
            continue

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ):
            console.print("[cyan]Thinking...[/cyan]")
            answer = engine.query(q)

        console.print(Panel(answer.answer, title="Answer", border_style="green"))

        if answer.sources:
            console.print("[dim]Sources:[/dim]")
            for source, _ in answer.sources[:3]:
                console.print(f"  [dim]- {source}[/dim]")
            console.print()

    console.print("[green]Goodbye![/green]")


# ─── Web UI Command ───────────────────────────────────────────────────────────


@app.command(name="web")
def launch_web(
    host: str = typer.Option("0.0.0.0", "--host", help="Server host."),
    port: int = typer.Option(7860, "--port", "-p", help="Server port."),
    share: bool = typer.Option(False, "--share", help="Create a public share link."),
):
    """Launch the Gradio Web UI."""
    from .web import create_app

    app_ui = create_app()
    app_ui.launch(server_name=host, server_port=port, share=share)
