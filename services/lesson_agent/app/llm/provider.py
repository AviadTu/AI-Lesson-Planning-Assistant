"""
LLM provider factory for the lesson-agent.

Selects a concrete provider from ``settings.LLM_PROVIDER`` and caches it.
Supported values:
  * ``ollama``  — Ollama (local daemon + Ollama Cloud). DEFAULT.
                  ``ollama_cloud`` is accepted as an alias so the existing
                  repo-root .env (LLM_PROVIDER=ollama_cloud) keeps working.
  * ``openai``  — OpenAI Chat Completions API (implemented, opt-in only).

Mirrors the RAG service's factory (services/rag/app/llm/provider.py).
"""

from __future__ import annotations

import threading

from app.config import settings
from app.llm.base import LLMProvider

_provider: LLMProvider | None = None
_lock = threading.Lock()

_OLLAMA_ALIASES = {"ollama", "ollama_cloud", "ollama-cloud"}
_OPENAI_ALIASES = {"openai"}


class UnimplementedProvider(LLMProvider):
    """Placeholder for an unrecognised LLM_PROVIDER value."""

    def __init__(self, name: str) -> None:
        self.name = name

    def _fail(self):
        raise NotImplementedError(f"LLM provider '{self.name}' is not implemented.")

    def complete_fast(self, system: str, user: str) -> str:
        self._fail()

    def complete_strong(self, system, user, *, temperature=None, max_tokens=None) -> str:
        self._fail()


def get_llm_provider() -> LLMProvider:
    """Return a cached LLM provider instance for the configured backend."""
    global _provider
    if _provider is None:
        with _lock:
            if _provider is None:
                name = (settings.LLM_PROVIDER or "").strip().lower()
                if name in _OLLAMA_ALIASES:
                    from app.llm.ollama_provider import OllamaProvider

                    _provider = OllamaProvider()
                elif name in _OPENAI_ALIASES:
                    from app.llm.openai_provider import OpenAIProvider

                    _provider = OpenAIProvider()
                else:
                    _provider = UnimplementedProvider(name or "unknown")
    return _provider
