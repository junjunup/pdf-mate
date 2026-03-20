"""LLM interface module: unified interface for multiple LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional


@dataclass
class Message:
    """A chat message."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMConfig:
    """Configuration for an LLM backend."""

    model: str = "gpt-3.5-turbo"
    temperature: float = 0.3
    max_tokens: int = 4096
    top_p: float = 1.0
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def chat(self, messages: list[Message], **kwargs: Any) -> str:
        """Send a chat completion request and return the response text."""

    @abstractmethod
    async def chat_stream(
        self, messages: list[Message], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream a chat completion response chunk by chunk."""


class OpenAIBackend(LLMBackend):
    """OpenAI-compatible LLM backend.

    Works with OpenAI, Azure OpenAI, and any OpenAI-compatible API
    (e.g., vLLM, LocalAI, Ollama's OpenAI-compatible endpoint).
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            client_kwargs: dict[str, Any] = {}
            if self.config.api_key:
                client_kwargs["api_key"] = self.config.api_key
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            self._client = OpenAI(**client_kwargs)
        return self._client

    def chat(self, messages: list[Message], **kwargs: Any) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            top_p=kwargs.get("top_p", self.config.top_p),
            **self.config.extra,
        )
        return response.choices[0].message.content or ""

    async def chat_stream(
        self, messages: list[Message], **kwargs: Any
    ) -> AsyncIterator[str]:
        from openai import AsyncOpenAI

        client_kwargs: dict[str, Any] = {}
        if self.config.api_key:
            client_kwargs["api_key"] = self.config.api_key
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url

        async_client = AsyncOpenAI(**client_kwargs)
        stream = await async_client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            top_p=kwargs.get("top_p", self.config.top_p),
            stream=True,
            **self.config.extra,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


class OllamaBackend(LLMBackend):
    """Ollama local LLM backend using OpenAI-compatible endpoint."""

    def __init__(self, config: LLMConfig):
        config_with_url = LLMConfig(
            model=config.model or "llama3",
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            base_url=config.base_url or "http://localhost:11434/v1",
            api_key=config.api_key or "ollama",
            extra=config.extra,
        )
        super().__init__(config_with_url)
        self._openai_backend = OpenAIBackend(config_with_url)

    def chat(self, messages: list[Message], **kwargs: Any) -> str:
        return self._openai_backend.chat(messages, **kwargs)

    async def chat_stream(
        self, messages: list[Message], **kwargs: Any
    ) -> AsyncIterator[str]:
        async for chunk in self._openai_backend.chat_stream(messages, **kwargs):
            yield chunk


def create_llm_backend(
    provider: str = "openai",
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> LLMBackend:
    """Factory function to create an LLM backend.

    Args:
        provider: Backend type - "openai" or "ollama".
        model: Model name to use.
        api_key: API key for the backend.
        base_url: Custom API base URL.
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.

    Returns:
        An initialized LLMBackend instance.
    """
    config = LLMConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
    )

    if provider == "ollama":
        return OllamaBackend(config)
    return OpenAIBackend(config)
