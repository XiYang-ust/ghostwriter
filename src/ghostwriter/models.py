"""Small provider-independent interfaces used by the core algorithm."""

from __future__ import annotations

from typing import Protocol, Sequence, TypedDict


class Message(TypedDict):
    role: str
    content: str


class ChatModel(Protocol):
    """A minimal chat-completion interface.

    Custom local or hosted models can implement this protocol without pulling
    provider-specific dependencies into the core package.
    """

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Return the assistant's text for a sequence of chat messages."""
