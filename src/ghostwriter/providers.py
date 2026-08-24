"""Adapters for the official OpenAI, Anthropic, and DeepSeek APIs."""

from __future__ import annotations

import os
from typing import Any, Sequence

from .models import Message


def _read_api_key(env_name: str) -> str:
    value = os.getenv(env_name)
    if not value:
        raise RuntimeError(f"Required environment variable {env_name} is not set")
    return value


class OpenAIChatModel:
    """Official OpenAI Chat Completions API adapter."""

    def __init__(
        self,
        model: str,
        *,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError('Install this provider with: pip install ".[openai]"') from exc

        self.model = model
        self.client = OpenAI(
            api_key=_read_api_key(api_key_env),
            timeout=timeout,
            max_retries=max_retries,
        )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        response = self.client.chat.completions.create(**request)
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Provider returned an empty response")
        return content


class AnthropicChatModel:
    """Anthropic Messages API adapter."""

    def __init__(
        self,
        model: str,
        *,
        api_key_env: str = "ANTHROPIC_API_KEY",
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError('Install this provider with: pip install ".[anthropic]"') from exc

        self.model = model
        self.client = Anthropic(
            api_key=_read_api_key(api_key_env),
            timeout=timeout,
            max_retries=max_retries,
        )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        system_parts = [message["content"] for message in messages if message["role"] == "system"]
        chat_messages = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
            if message["role"] != "system"
        ]
        response = self.client.messages.create(
            model=self.model,
            system="\n\n".join(system_parts),
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        if not text:
            raise RuntimeError("Provider returned an empty response")
        return text


DEEPSEEK_API_URL = "https://api.deepseek.com"


class DeepSeekChatModel:
    """Official DeepSeek Chat Completions API adapter."""

    def __init__(
        self,
        model: str,
        *,
        api_key_env: str = "DEEPSEEK_API_KEY",
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError('Install this provider with: pip install ".[deepseek]"') from exc

        self.model = model
        self.client = OpenAI(
            api_key=_read_api_key(api_key_env),
            base_url=DEEPSEEK_API_URL,
            timeout=timeout,
            max_retries=max_retries,
        )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Provider returned an empty response")
        return content


def create_chat_model(
    provider: str,
    model: str,
    *,
    api_key_env: str | None = None,
):
    """Create a supported provider adapter from CLI-friendly values."""
    if provider == "openai":
        return OpenAIChatModel(
            model,
            api_key_env=api_key_env or "OPENAI_API_KEY",
        )
    if provider == "anthropic":
        return AnthropicChatModel(
            model,
            api_key_env=api_key_env or "ANTHROPIC_API_KEY",
        )
    if provider == "deepseek":
        return DeepSeekChatModel(
            model,
            api_key_env=api_key_env or "DEEPSEEK_API_KEY",
        )
    raise ValueError(f"Unsupported provider: {provider}")
