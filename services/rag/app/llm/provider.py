"""
LLM provider factory.

Selects a concrete provider based on ``settings.LLM_PROVIDER``. Supported:
  * ``llama_cpp``     — local GGUF via llama-cpp-python
  * ``ollama_cloud``  — cloud models via the local Ollama HTTP API
"""

from __future__ import annotations

import threading

from app.config import settings
from app.llm.base import LLMProvider

_provider: LLMProvider | None = None
_lock = threading.Lock()

_LLAMA_CPP_ALIASES = {"llama_cpp", "llamacpp", "llama-cpp"}
_OLLAMA_CLOUD_ALIASES = {"ollama_cloud", "ollama-cloud"}


class UnimplementedProvider(LLMProvider):
    """Placeholder for providers that are not wired yet."""

    def __init__(self, name: str) -> None:
        self.name = name

    def generate(self, messages: list[dict], **kwargs) -> str:
        raise NotImplementedError(
            f"LLM provider '{self.name}' is not implemented."
        )


def get_llm_provider() -> LLMProvider:
    """Return a cached LLM provider instance for the configured backend."""
    global _provider
    if _provider is None:
        with _lock:
            if _provider is None:
                name = (settings.LLM_PROVIDER or "").strip().lower()
                if name in _LLAMA_CPP_ALIASES:
                    from app.llm.llama_cpp_provider import LlamaCppProvider

                    _provider = LlamaCppProvider()
                elif name in _OLLAMA_CLOUD_ALIASES:
                    from app.llm.ollama_cloud_provider import OllamaCloudProvider

                    _provider = OllamaCloudProvider()
                else:
                    _provider = UnimplementedProvider(name or "unknown")
    return _provider
