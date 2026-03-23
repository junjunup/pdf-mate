"""RAG module: Retrieval-Augmented Generation for PDF Q&A."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None  # type: ignore[assignment]

from .embedding import create_embedding_backend
from .exceptions import RAGError
from .llm import LLMBackend, Message, create_llm_backend
from .storage import Chunk, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_chunks_per_query: int = 5

    # Retrieval
    retrieval_top_k: int = 5
    min_relevance_score: float = 0.5

    # LLM
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = None
    llm_base_url: str | None = None

    # Embedding
    embedding_provider: str = "local"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None

    # Storage
    persist_directory: str | None = None
    collection_name: str = "pdf_mate"


@dataclass
class RAGAnswer:
    """A RAG-generated answer with context."""

    question: str
    answer: str
    sources: list[tuple[str, str]] = field(default_factory=list)  # (source, chunk_text)
    score: float = 0.0


class TextSplitter:
    """Split text into overlapping chunks for RAG indexing.

    When tiktoken is available, chunk_size and chunk_overlap are measured
    in **tokens** (using the ``cl100k_base`` encoding).  When tiktoken is
    not available, they are measured in **whitespace-delimited words**.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self._encoding = tiktoken.get_encoding("cl100k_base") if tiktoken else None
        except Exception:
            self._encoding = None

    def split_text(
        self, text: str, metadata: dict | None = None
    ) -> list[Chunk]:
        """Split text into overlapping chunks.

        When tiktoken is available the text is first tokenized and then
        split by **token count**, so ``chunk_size=512`` means 512 tokens.
        Token boundaries are snapped to the nearest whitespace to avoid
        cutting words in half.

        Args:
            text: Text to split.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of Chunk objects.
        """
        if not text.strip():
            return []

        metadata = metadata or {}
        chunks: list[Chunk] = []

        if self._encoding:
            # ── Token-level splitting ──────────────────────────────
            tokens = self._encoding.encode(text)
            step = self.chunk_size - self.chunk_overlap
            i = 0
            while i < len(tokens):
                window = tokens[i : i + self.chunk_size]
                chunk_text = self._encoding.decode(window).strip()
                if chunk_text:
                    chunk_meta = {**metadata, "chunk_index": len(chunks)}
                    chunk_meta["token_start"] = i
                    chunk_meta["token_end"] = i + len(window)
                    chunks.append(Chunk(text=chunk_text, metadata=chunk_meta))
                i += step
        else:
            # ── Word-level fallback ────────────────────────────────
            words = text.split()
            step = self.chunk_size - self.chunk_overlap
            i = 0
            while i < len(words):
                window = words[i : i + self.chunk_size]
                chunk_text = " ".join(window)
                chunk_meta = {**metadata, "chunk_index": len(chunks)}

                # Compute character offset
                if i == 0:
                    start_char = text.find(window[0])
                else:
                    prefix = " ".join(words[:i])
                    start_char = len(prefix) + 1

                chunk_meta["start_char"] = start_char
                chunk_meta["end_char"] = start_char + len(chunk_text)
                chunks.append(Chunk(text=chunk_text, metadata=chunk_meta))
                i += step

        return chunks


class RAGEngine:
    """RAG engine for PDF question answering."""

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self._splitter = TextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self._embedder = create_embedding_backend(
            provider=self.config.embedding_provider,
            model_name=self.config.embedding_model,
            api_key=self.config.embedding_api_key,
            base_url=self.config.embedding_base_url,
        )
        self._store = VectorStore(
            collection_name=self.config.collection_name,
            persist_directory=self.config.persist_directory,
        )
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

    def index_document(
        self,
        text: str,
        source_name: str,
        metadata: dict | None = None,
    ) -> int:
        """Index a document's text for RAG retrieval.

        Args:
            text: Full document text.
            source_name: Name of the source document.
            metadata: Additional metadata.

        Returns:
            Number of chunks indexed.

        Raises:
            RAGError: If indexing fails.
        """
        try:
            meta = {**(metadata or {}), "source": source_name}
            chunks = self._splitter.split_text(text, meta)

            if chunks:
                embeddings = self._embedder.embed([c.text for c in chunks])
                self._store.add(chunks=chunks, embeddings=embeddings)

            return len(chunks)
        except RAGError:
            raise
        except Exception as exc:
            raise RAGError(f"Failed to index document '{source_name}': {exc}") from exc

    def query(self, question: str, n_results: int | None = None) -> RAGAnswer:
        """Ask a question and get a RAG-powered answer.

        Args:
            question: The question to answer.
            n_results: Number of context chunks to retrieve.

        Returns:
            RAGAnswer with the answer and sources.

        Raises:
            RAGError: If the query fails.
        """
        try:
            n = n_results or self.config.retrieval_top_k

            # Retrieve relevant chunks
            query_embedding = self._embedder.embed_query(question)
            results = self._store.query(
                query_embedding=query_embedding,
                n_results=n,
            )

            # Filter by relevance score
            relevant_chunks = [
                (chunk, score)
                for chunk, score in results
                if score <= (1 - self.config.min_relevance_score)  # ChromaDB returns distance
            ][:self.config.max_chunks_per_query]

            if not relevant_chunks:
                no_info = (
                    "I couldn't find relevant information in the "
                    "document to answer this question."
                )
                return RAGAnswer(
                    question=question,
                    answer=no_info,
                    sources=[],
                    score=0.0,
                )

            # Build context
            context_parts = []
            sources = []
            for chunk, _score in relevant_chunks:
                source = chunk.metadata.get("source", "unknown")
                context_parts.append(f"[Source: {source}]\n{chunk.text}")
                sources.append((source, chunk.text))

            context = "\n\n---\n\n".join(context_parts)

            # Generate answer using LLM
            llm = self._get_llm()
            messages = [
                Message(
                    role="system",
                    content=(
                        "You are a helpful assistant that answers questions based on "
                        "the provided document context. Always cite the source when "
                        "providing information. If the answer cannot be found in the "
                        "context, say so clearly. Be concise and accurate."
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        f"Context from documents:\n\n{context}\n\n"
                        f"Question: {question}\n\n"
                        "Please provide a clear, accurate answer based on the context above."
                    ),
                ),
            ]

            answer = llm.chat(messages)

            avg_score = sum(s for _, s in relevant_chunks) / len(relevant_chunks)

            return RAGAnswer(
                question=question,
                answer=answer,
                sources=sources,
                score=avg_score,
            )
        except RAGError:
            raise
        except Exception as exc:
            raise RAGError(f"RAG query failed: {exc}") from exc

    def list_indexed_sources(self) -> list[str]:
        """Return list of indexed document sources."""
        return self._store.list_sources()

    def delete_source(self, source_name: str) -> None:
        """Remove a document from the index."""
        self._store.delete(source_name=source_name)
