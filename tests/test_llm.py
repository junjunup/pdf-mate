"""Tests for LLM module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pdf_mate.exceptions import ConfigError
from pdf_mate.llm import (
    LLMConfig,
    Message,
    OllamaBackend,
    OpenAIBackend,
    create_llm_backend,
)


class TestMessage:
    def test_creation(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_roles(self):
        for role in ("system", "user", "assistant"):
            msg = Message(role=role, content="test")
            assert msg.role == role


class TestLLMConfig:
    def test_defaults(self):
        config = LLMConfig()
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.3
        assert config.max_tokens == 4096
        assert config.top_p == 1.0
        assert config.api_key is None
        assert config.base_url is None

    def test_custom(self):
        config = LLMConfig(model="gpt-4", temperature=0.7, api_key="sk-test")
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.api_key == "sk-test"


class TestOpenAIBackend:
    def test_init(self):
        config = LLMConfig(model="gpt-3.5-turbo")
        backend = OpenAIBackend(config)
        assert backend.config.model == "gpt-3.5-turbo"
        assert backend._client is None
        assert backend._async_client is None


class TestOllamaBackend:
    def test_url_configuration(self):
        backend = OllamaBackend(LLMConfig(model="llama3"))
        assert backend.config.base_url == "http://localhost:11434/v1"
        assert backend.config.api_key == "ollama"

    def test_custom_url(self):
        backend = OllamaBackend(
            LLMConfig(model="llama3", base_url="http://custom:11434/v1")
        )
        assert backend.config.base_url == "http://custom:11434/v1"

    def test_default_model(self):
        backend = OllamaBackend(LLMConfig(model=""))
        assert backend.config.model == "llama3"


class TestCreateLLMBackend:
    def test_openai_backend(self):
        backend = create_llm_backend(provider="openai", model="gpt-3.5-turbo")
        assert isinstance(backend, OpenAIBackend)

    def test_ollama_backend(self):
        backend = create_llm_backend(provider="ollama", model="llama3")
        assert isinstance(backend, OllamaBackend)

    def test_default_backend(self):
        backend = create_llm_backend()
        assert isinstance(backend, OpenAIBackend)

    def test_unknown_provider_raises(self):
        with pytest.raises(ConfigError, match="Unknown LLM provider"):
            create_llm_backend(provider="nonexistent")


class TestOpenAIBackendChat:
    def test_chat_success(self):
        """Test successful synchronous chat completion."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from GPT"
        mock_client.chat.completions.create.return_value = mock_response

        config = LLMConfig(model="gpt-4o-mini", api_key="sk-test")
        backend = OpenAIBackend(config)
        backend._client = mock_client  # inject mock client directly
        result = backend.chat([Message(role="user", content="Hi")])
        assert result == "Hello from GPT"

    def test_chat_empty_response(self):
        """Test chat completion with None content."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response

        config = LLMConfig(api_key="sk-test")
        backend = OpenAIBackend(config)
        backend._client = mock_client
        result = backend.chat([Message(role="user", content="Hi")])
        assert result == ""

    def test_chat_raises_llm_error(self):
        """Test that API errors are wrapped in LLMError."""
        from pdf_mate.exceptions import LLMError

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")

        config = LLMConfig(api_key="sk-test")
        backend = OpenAIBackend(config)
        backend._client = mock_client
        with pytest.raises(LLMError, match="Chat completion failed"):
            backend.chat([Message(role="user", content="Hi")])

    def test_client_caching(self):
        """Test that _get_client caches the OpenAI client."""
        config = LLMConfig(api_key="sk-test", base_url="https://example.com")
        backend = OpenAIBackend(config)
        # Inject a mock to verify caching
        mock_client = MagicMock()
        backend._client = mock_client
        client1 = backend._get_client()
        client2 = backend._get_client()
        assert client1 is client2
        assert client1 is mock_client

    def test_async_client_caching(self):
        """Test that _get_async_client caches the AsyncOpenAI client."""
        config = LLMConfig(api_key="sk-test")
        backend = OpenAIBackend(config)
        mock_async = MagicMock()
        backend._async_client = mock_async
        client1 = backend._get_async_client()
        client2 = backend._get_async_client()
        assert client1 is client2
        assert client1 is mock_async

    def test_chat_with_kwargs(self):
        """Test that kwargs override config defaults."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"
        mock_client.chat.completions.create.return_value = mock_response

        config = LLMConfig(model="gpt-4o-mini", temperature=0.3)
        backend = OpenAIBackend(config)
        backend._client = mock_client
        backend.chat(
            [Message(role="user", content="Hi")],
            model="gpt-4o",
            temperature=0.9,
        )
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs[1]["model"] == "gpt-4o"
        assert call_kwargs[1]["temperature"] == 0.9


class TestOpenAIBackendChatStream:
    @pytest.mark.asyncio
    async def test_chat_stream_success(self):
        """Test successful async chat streaming."""
        from unittest.mock import AsyncMock

        # Build mock async stream chunks
        mock_chunk_1 = MagicMock()
        mock_chunk_1.choices = [MagicMock()]
        mock_chunk_1.choices[0].delta.content = "Hello "

        mock_chunk_2 = MagicMock()
        mock_chunk_2.choices = [MagicMock()]
        mock_chunk_2.choices[0].delta.content = "World"

        mock_chunk_3 = MagicMock()
        mock_chunk_3.choices = [MagicMock()]
        mock_chunk_3.choices[0].delta.content = None  # empty delta

        # create() is awaited, so it must return an awaitable that resolves
        # to an async iterable
        class _FakeStream:
            def __init__(self, items):
                self._items = items
            def __aiter__(self):
                return self
            async def __anext__(self):
                if not self._items:
                    raise StopAsyncIteration
                return self._items.pop(0)

        mock_async_client = MagicMock()
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=_FakeStream([mock_chunk_1, mock_chunk_2, mock_chunk_3])
        )

        config = LLMConfig(api_key="sk-test")
        backend = OpenAIBackend(config)
        backend._async_client = mock_async_client

        chunks = []
        async for text in backend.chat_stream([Message(role="user", content="Hi")]):
            chunks.append(text)

        assert chunks == ["Hello ", "World"]

    @pytest.mark.asyncio
    async def test_chat_stream_raises_llm_error(self):
        """Test that stream errors are wrapped in LLMError."""
        from pdf_mate.exceptions import LLMError

        mock_async_client = MagicMock()
        mock_async_client.chat.completions.create = MagicMock(
            side_effect=RuntimeError("stream broken")
        )

        config = LLMConfig(api_key="sk-test")
        backend = OpenAIBackend(config)
        backend._async_client = mock_async_client

        with pytest.raises(LLMError, match="Chat stream failed"):
            async for _ in backend.chat_stream([Message(role="user", content="Hi")]):
                pass

    @pytest.mark.asyncio
    async def test_chat_stream_reraises_llm_error(self):
        """Test that LLMError from stream is re-raised directly."""
        from pdf_mate.exceptions import LLMError

        mock_async_client = MagicMock()
        mock_async_client.chat.completions.create = MagicMock(
            side_effect=LLMError("original error")
        )

        config = LLMConfig(api_key="sk-test")
        backend = OpenAIBackend(config)
        backend._async_client = mock_async_client

        with pytest.raises(LLMError, match="original error"):
            async for _ in backend.chat_stream([Message(role="user", content="Hi")]):
                pass


class TestOpenAIBackendChatLLMErrorPassthrough:
    def test_chat_reraises_llm_error(self):
        """Test that LLMError from client is re-raised without wrapping."""
        from pdf_mate.exceptions import LLMError

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = LLMError("direct error")

        config = LLMConfig(api_key="sk-test")
        backend = OpenAIBackend(config)
        backend._client = mock_client

        with pytest.raises(LLMError, match="direct error"):
            backend.chat([Message(role="user", content="Hi")])


class TestCreateLLMBackendParams:
    def test_all_params_passed(self):
        """Test that factory passes all params to config."""
        backend = create_llm_backend(
            provider="openai",
            model="gpt-4",
            api_key="sk-123",
            base_url="https://api.example.com",
            temperature=0.8,
            max_tokens=2048,
        )
        assert backend.config.model == "gpt-4"
        assert backend.config.api_key == "sk-123"
        assert backend.config.base_url == "https://api.example.com"
        assert backend.config.temperature == 0.8
        assert backend.config.max_tokens == 2048


class TestOllamaBackendChat:
    def test_ollama_chat_delegates(self):
        """Test that OllamaBackend delegates to OpenAIBackend."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Ollama response"
        mock_client.chat.completions.create.return_value = mock_response

        backend = OllamaBackend(LLMConfig(model="llama3"))
        backend._openai_backend._client = mock_client
        result = backend.chat([Message(role="user", content="Hello")])
        assert result == "Ollama response"

    @pytest.mark.asyncio
    async def test_ollama_chat_stream_delegates(self):
        """Test that OllamaBackend.chat_stream delegates to OpenAIBackend."""
        from unittest.mock import AsyncMock

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "streamed"

        class _FakeStream:
            def __init__(self, items):
                self._items = items
            def __aiter__(self):
                return self
            async def __anext__(self):
                if not self._items:
                    raise StopAsyncIteration
                return self._items.pop(0)

        mock_async_client = MagicMock()
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=_FakeStream([mock_chunk])
        )

        backend = OllamaBackend(LLMConfig(model="llama3"))
        backend._openai_backend._async_client = mock_async_client

        chunks = []
        async for text in backend.chat_stream([Message(role="user", content="Hi")]):
            chunks.append(text)

        assert chunks == ["streamed"]
