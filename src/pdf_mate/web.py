"""Web UI module: Gradio-based interface for pdf-mate."""

from __future__ import annotations

import os

import gradio as gr


def create_app():
    """Create and return the Gradio application."""

    # ─── State ────────────────────────────────────────────────────────────────

    state = {
        "parser": None,
        "content": None,
        "rag_engine": None,
        "indexed": False,
    }

    # ─── Parse Tab ────────────────────────────────────────────────────────────

    def handle_parse(file, extract_images: bool):
        from .parser import PDFParser

        if file is None:
            return "Please upload a PDF file.", "", ""

        parser = PDFParser(extract_images=extract_images)
        content = parser.parse(file.name)

        state["parser"] = parser
        state["content"] = content
        state["indexed"] = False

        info_lines = [
            f"**Filename:** {content.filename}",
            f"**Pages:** {content.page_count}",
            f"**Text blocks:** {len(content.text_blocks)}",
            f"**Tables:** {len(content.tables)}",
            f"**Images:** {len(content.images)}",
        ]
        return "\n".join(info_lines), content.full_text[:10000], content.markdown_text[:10000]

    # ─── Summary Tab ──────────────────────────────────────────────────────────

    def handle_summary(
        file,
        model: str,
        style: str,
        api_key: str,
        base_url: str,
    ):
        from .parser import PDFParser
        from .summary import SummaryConfig, DocumentSummarizer

        if file is None:
            return "Please upload a PDF file.", ""

        parser = PDFParser()
        content = parser.parse(file.name)

        if not content.full_text.strip():
            return "No text found in PDF.", ""

        config = SummaryConfig(
            llm_model=model,
            llm_api_key=api_key or os.environ.get("OPENAI_API_KEY") or None,
            llm_base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
            style=style,
        )
        summarizer = DocumentSummarizer(config)
        doc_summary = summarizer.summarize(
            text=content.full_text,
            filename=content.filename,
            page_count=content.page_count,
        )

        title_md = f"# {doc_summary.title}\n\n"
        summary_md = f"## Summary\n\n{doc_summary.summary}\n\n"
        points_md = "## Key Points\n\n" + "\n".join(
            f"- {p}" for p in doc_summary.key_points
        )
        stats_md = (
            f"\n\n---\n\n**Pages:** {doc_summary.page_count} | "
            f"**Words:** {doc_summary.word_count} | "
            f"**Key Points:** {len(doc_summary.key_points)}"
        )

        return title_md, summary_md + points_md + stats_md

    # ─── Q&A Tab ──────────────────────────────────────────────────────────────

    def handle_index(
        file,
        llm_model: str,
        embedding_provider: str,
        api_key: str,
        base_url: str,
    ):
        from .parser import PDFParser
        from .rag import RAGConfig, RAGEngine

        if file is None:
            return "Please upload a PDF file.", ""

        if state.get("indexed") and state.get("_last_file") == file.name:
            return "Document already indexed.", str(state["rag_engine"]._store.count) + " chunks"

        parser = PDFParser()
        content = parser.parse(file.name)

        if not content.full_text.strip():
            return "No text found in PDF.", ""

        config = RAGConfig(
            llm_model=llm_model,
            llm_api_key=api_key or os.environ.get("OPENAI_API_KEY") or None,
            llm_base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
            embedding_provider=embedding_provider,
        )
        engine = RAGEngine(config)
        chunk_count = engine.index_document(
            text=content.full_text,
            source_name=content.filename,
        )

        state["rag_engine"] = engine
        state["indexed"] = True
        state["_last_file"] = file.name

        return f"Indexed successfully!", f"{chunk_count} chunks indexed"

    def handle_question(question: str):
        if not question.strip():
            return "Please enter a question."

        engine = state.get("rag_engine")
        if engine is None:
            return "Please index a document first."

        answer = engine.query(question)

        sources_md = ""
        if answer.sources:
            sources_md = "\n\n---\n\n**Sources:**\n"
            for source, text in answer.sources[:3]:
                sources_md += f"- `{source}`\n"

        return answer.answer + sources_md

    # ─── Build Interface ──────────────────────────────────────────────────────

    with gr.Blocks(
        title="pdf-mate",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 📄 pdf-mate
            **AI-powered PDF parsing, summarization, and conversational Q&A.**

            Upload a PDF document and start exploring with AI assistance.
            """
        )

        with gr.Tabs():
            # ── Parse Tab ──
            with gr.Tab("Parse"):
                with gr.Row():
                    pdf_input = gr.File(
                        label="Upload PDF", file_types=[".pdf"], type="filepath"
                    )
                    with gr.Column():
                        img_check = gr.Checkbox(
                            label="Extract Images", value=False
                        )
                        parse_btn = gr.Button("Parse PDF", variant="primary")

                info_output = gr.Markdown(label="Document Info")
                with gr.Row():
                    text_output = gr.Textbox(
                        label="Extracted Text", lines=10, show_copy_button=True
                    )
                    md_output = gr.Markdown(label="Markdown Preview")

                parse_btn.click(
                    fn=handle_parse,
                    inputs=[pdf_input, img_check],
                    outputs=[info_output, text_output, md_output],
                )

            # ── Summary Tab ──
            with gr.Tab("Summary"):
                with gr.Row():
                    sum_pdf = gr.File(
                        label="Upload PDF", file_types=[".pdf"], type="filepath"
                    )
                    with gr.Column():
                        sum_model = gr.Dropdown(
                            label="LLM Model",
                            choices=["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"],
                            value="gpt-3.5-turbo",
                        )
                        sum_style = gr.Dropdown(
                            label="Summary Style",
                            choices=["concise", "detailed", "bullets"],
                            value="concise",
                        )
                        sum_api_key = gr.Textbox(
                            label="API Key (optional)", type="password"
                        )
                        sum_base_url = gr.Textbox(
                            label="Base URL (optional)", placeholder="Leave empty for default"
                        )
                        sum_btn = gr.Button("Generate Summary", variant="primary")

                sum_title = gr.Markdown(label="Title")
                sum_output = gr.Markdown(label="Summary")

                sum_btn.click(
                    fn=handle_summary,
                    inputs=[sum_pdf, sum_model, sum_style, sum_api_key, sum_base_url],
                    outputs=[sum_title, sum_output],
                )

            # ── Q&A Tab ──
            with gr.Tab("Q&A"):
                with gr.Row():
                    qa_pdf = gr.File(
                        label="Upload PDF", file_types=[".pdf"], type="filepath"
                    )
                    with gr.Column():
                        qa_model = gr.Dropdown(
                            label="LLM Model",
                            choices=["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"],
                            value="gpt-3.5-turbo",
                        )
                        qa_embedding = gr.Dropdown(
                            label="Embedding",
                            choices=["local", "openai"],
                            value="local",
                        )
                        qa_api_key = gr.Textbox(
                            label="API Key (optional)", type="password"
                        )
                        qa_base_url = gr.Textbox(
                            label="Base URL (optional)", placeholder="Leave empty for default"
                        )
                        index_btn = gr.Button("Index Document", variant="secondary")

                index_status = gr.Textbox(label="Index Status", interactive=False)

                qa_question = gr.Textbox(
                    label="Ask a question",
                    placeholder="e.g., What is the main topic of this document?",
                    lines=2,
                )
                qa_ask_btn = gr.Button("Ask", variant="primary")
                qa_answer = gr.Markdown(label="Answer")

                index_btn.click(
                    fn=handle_index,
                    inputs=[qa_pdf, qa_model, qa_embedding, qa_api_key, qa_base_url],
                    outputs=[index_status, index_status],
                )
                qa_ask_btn.click(
                    fn=handle_question,
                    inputs=[qa_question],
                    outputs=[qa_answer],
                )

    return demo
