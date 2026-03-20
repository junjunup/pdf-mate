"""Tests for LLM module."""

from __future__ import annotations

import pytest

from pdf_mate.llm import (
    LLMConfig,
    Message,
    OpenAIBackend,
    OllamaBackend,
    create_llm_backend,
)


class TestMessage:
    def test_creation(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestLLMConfig:
    def test_defaults(self):
        config = LLMConfig()
        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.3
        assert config.max_tokens == 4096

    def test_custom(self):
        config = LLMConfig(model="gpt-4", temperature=0.7)
        assert config.model == "gpt-4"
        assert config.temperature == 0.7


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
