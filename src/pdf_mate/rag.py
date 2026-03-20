"""RAG module: Retrieval-Augmented Generation for PDF Q&A."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import tiktoken

from .embedding import EmbeddingBackend, create_embedding_backend
from .llm import LLMBackend, Message, create_llm_backend
from .storage import Chunk, VectorStore


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
    llm_model: str = "gpt-3.5-turbo"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None

    # Embedding
    embedding_provider: str = "local"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_api_key: Optional[str] = None
    embedding_base_url: Optional[str] = None

    # Storage
    persist_directory: Optional[str] = None
    collection_name: str = "pdf_mate"


@dataclass
class RAGAnswer:
    """A RAG-generated answer with context."""

    question: str
    answer: str
    sources: list[tuple[str, str]] = field(default_factory=list)  # (source, chunk_text)
    score: float = 0.0


class TextSplitter:
    """Split text into overlapping chunks for RAG indexing."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoding = None

    def _count_tokens(self, text: str) -> int:
        if self._encoding:
            return len(self._encoding.encode(text))
        return len(text)

    def split_text(
        self, text: str, metadata: Optional[dict] = None
    ) -> list[Chunk]:
        """Split text into overlapping chunks.

        Args:
            text: Text to split.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of Chunk objects.
        """
        if not text.strip():
            return []

        metadata = metadata or {}
        tokens = text.split() if not self._encoding else None
        chunks: list[Chunk] = []

        if tokens:
            # Word-level splitting with token count estimation
            i = 0
            while i < len(tokens):
                window = tokens[i : i + self.chunk_size]
                chunk_text = " ".join(window)
                chunk_meta = {**metadata, "chunk_index": len(chunks)}

                # Calculate character-based start/end for metadata
                start_char = text.find(window[0]) if window else 0
                chunk_meta["start_char"] = start_char
                chunk_meta["end_char"] = start_char + len(chunk_text)

                chunks.append(Chunk(text=chunk_text, metadata=chunk_meta))
                i += self.chunk_size - self.chunk_overlap
        else:
            # Simple character-based splitting
            i = 0
            while i < len(text):
                end = min(i + self.chunk_size, len(text))
                chunk_text = text[i:end]
                chunk_meta = {**metadata, "chunk_index": len(chunks)}
                chunk_meta["start_char"] = i
                chunk_meta["end_char"] = end
                chunks.append(Chunk(text=chunk_text, metadata=chunk_meta))
                i += self.chunk_size - self.chunk_overlap

        return chunks


class RAGEngine:
    """RAG engine for PDF question answering."""

    def __init__(self, config: Optional[RAGConfig] = None):
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
        self._llm: Optional[LLMBackend] = None

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
        metadata: Optional[dict] = None,
    ) -> int:
        """Index a document's text for RAG retrieval.

        Args:
            text: Full document text.
            source_name: Name of the source document.
            metadata: Additional metadata.

        Returns:
            Number of chunks indexed.
        """
        meta = {**(metadata or {}), "source": source_name}
        chunks = self._splitter.split_text(text, meta)

        if chunks:
            embeddings = self._embedder.embed([c.text for c in chunks])
            self._store.add(chunks=chunks, embeddings=embeddings)

        return len(chunks)

    def query(self, question: str, n_results: Optional[int] = None) -> RAGAnswer:
        """Ask a question and get a RAG-powered answer.

        Args:
            question: The question to answer.
            n_results: Number of context chunks to retrieve.

        Returns:
            RAGAnswer with the answer and sources.
        """
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
            return RAGAnswer(
                question=question,
                answer="I couldn't find relevant information in the document to answer this question.",
                sources=[],
                score=0.0,
            )

        # Build context
        context_parts = []
        sources = []
        for chunk, score in relevant_chunks:
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

    def list_indexed_sources(self) -> list[str]:
        """Return list of indexed document sources."""
        return self._store.list_sources()

    def delete_source(self, source_name: str) -> None:
        """Remove a document from the index."""
        self._store.delete(source_name=source_name)
