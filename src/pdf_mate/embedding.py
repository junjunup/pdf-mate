"""Embedding module: text vectorization for RAG retrieval."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from .exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for a list of texts.

        Args:
            texts: Input strings.

        Returns:
            One :class:`numpy.ndarray` per input string.
        """

    def embed_query(self, text: str) -> np.ndarray:
        """Generate embedding for a single query text.

        This is a convenience wrapper around :meth:`embed`.
        """
        results = self.embed([text])
        return results[0]


class SentenceTransformerBackend(EmbeddingBackend):
    """Local sentence-transformers embedding backend.

    Uses CPU/GPU-accelerated models from the ``sentence-transformers`` library.

    Note:
        Requires the ``sentence-transformers`` package (~2 GB).
        Install with ``pip install pdf-mate[local]``.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ):
        """Initialize the embedding backend.

        Args:
            model_name: HuggingFace model name or local path.
            device: Device to use (e.g., ``"cuda"``, ``"cpu"``, ``None`` for auto).
            batch_size: Batch size for encoding.
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise EmbeddingError(
                    "sentence-transformers is required for local embeddings. "
                    "Install it with: pip install pdf-mate[local]"
                ) from None

            kwargs = {"model_name_or_path": self.model_name}
            if self.device:
                kwargs["device"] = self.device
            self._model = SentenceTransformer(**kwargs)
        return self._model

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings using local sentence-transformers model.

        Raises:
            EmbeddingError: If model loading or encoding fails.
        """
        try:
            model = self._get_model()
            embeddings = model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return [np.array(e) for e in embeddings]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """OpenAI embedding backend using the API.

    Uses the OpenAI embeddings endpoint for text vectorization.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 100,
    ):
        """Initialize the OpenAI embedding backend.

        Args:
            model: Embedding model name.
            api_key: OpenAI API key.
            base_url: Custom API base URL.
            batch_size: Maximum batch size per request.
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.batch_size = batch_size
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise EmbeddingError(
                    "openai is required for OpenAI embeddings. "
                    "Install it with: pip install pdf-mate[llm]"
                ) from None

            kwargs: dict = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings via OpenAI API.

        Raises:
            EmbeddingError: If the API call fails.
        """
        try:
            client = self._get_client()
            all_embeddings: list[np.ndarray] = []

            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                response = client.embeddings.create(input=batch, model=self.model)
                for item in response.data:
                    all_embeddings.append(np.array(item.embedding))

            return all_embeddings
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"OpenAI embedding failed: {exc}") from exc


def create_embedding_backend(
    provider: str = "local",
    model_name: str = "all-MiniLM-L6-v2",
    api_key: str | None = None,
    base_url: str | None = None,
) -> EmbeddingBackend:
    """Factory function to create an embedding backend.

    Args:
        provider: ``"local"`` for sentence-transformers, ``"openai"`` for OpenAI API.
        model_name: Model name to use.
        api_key: API key for OpenAI backend.
        base_url: Custom API base URL.

    Returns:
        An initialized :class:`EmbeddingBackend` instance.

    Raises:
        ConfigError: If *provider* is not recognised.
    """
    if provider == "openai":
        return OpenAIEmbeddingBackend(
            model=model_name, api_key=api_key, base_url=base_url
        )
    if provider == "local":
        return SentenceTransformerBackend(model_name=model_name)

    from .exceptions import ConfigError

    raise ConfigError(
        f"Unknown embedding provider: {provider!r}. Choose 'local' or 'openai'."
    )
