"""
LLM provider abstraction.

Generation is kept behind this interface so retrieval/ingestion never talk to a
concrete backend (llama.cpp, an OpenAI-compatible API, ...). Providers consume
chat messages ([{"role", "content"}, ...]) and return the assistant text.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Common interface every concrete LLM backend must implement."""

    name: str

    @abstractmethod
    def generate(self, messages: list[dict], **kwargs) -> str:
        """Generate an assistant reply for the given chat messages."""
        raise NotImplementedError
