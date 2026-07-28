"""
LLM provider abstraction for the lesson-agent.

Model access is kept behind this interface so the LangGraph flow (app/graph.py)
and the rest of the service never talk to a concrete backend (Ollama, an
OpenAI-compatible API, ...). Mirrors the RAG service's provider abstraction
(app/llm/ in the RAG service).

The lesson-agent uses two model *tiers* (an inherent part of its design, not a
provider detail):

  * fast   — cheap labelling: intent classification, missing-info detection,
             idea selection, and the short "assist" fallback reply.
  * strong — content generation: lesson ideas, the full lesson plan, revisions.

Every provider maps both tiers onto its own backend (for Ollama: the local
daemon for ``fast`` and Ollama Cloud for ``strong``; for OpenAI: a small model
for ``fast`` and a stronger model for ``strong``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Common interface every concrete LLM backend must implement."""

    name: str

    @abstractmethod
    def complete_fast(self, system: str, user: str) -> str:
        """One completion from the FAST tier. Returns the assistant text."""
        raise NotImplementedError

    @abstractmethod
    def complete_strong(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """One completion from the STRONG tier. Returns the assistant text."""
        raise NotImplementedError
