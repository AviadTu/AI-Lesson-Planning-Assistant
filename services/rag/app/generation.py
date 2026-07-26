"""
RAG answer generation orchestrator.

Flow for one turn (no LangGraph):
  1. Retrieve the top-K relevant chunks (existing retrieval layer / ChromaDB).
  2. If nothing relevant is found, return a clear "no relevant information"
     message — the LLM is never called and nothing is invented.
  3. Otherwise build the prompt, generate with the configured LLM provider,
     and attach citations derived ONLY from retrieval metadata.
"""

from __future__ import annotations

import logging
import re

from app import prompting, retrieval
from app.config import settings
from app.llm.provider import get_llm_provider

logger = logging.getLogger("rag.generation")

# Some GGUF chat templates (e.g. Qwen3) emit <think>...</think> reasoning; strip
# it so the user only sees the final answer.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _citations(chunks: list[dict]) -> list[dict]:
    """Build citations from retrieval metadata only, de-duplicated by doc_id."""
    seen: set = set()
    cites: list[dict] = []
    for ch in chunks:
        doc_id = ch.get("doc_id")
        key = doc_id or ch.get("original_filename")
        if key in seen:
            continue
        seen.add(key)
        cites.append(
            {
                "source": ch.get("original_filename") or doc_id or "",
                "doc_id": doc_id,
            }
        )
    return cites


def _generate_with_fallback(messages: list[dict]) -> str:
    """
    Generate with the configured provider, logging which one answered.

    If the primary provider is Ollama Cloud and it is unavailable (network /
    TLS / HTTP error, missing key, ...), gracefully fall back to the local
    llama.cpp provider so the demo keeps working offline. A llama.cpp primary
    has no cloud to fall back to, so its errors propagate unchanged.
    """
    provider = get_llm_provider()
    try:
        raw = provider.generate(messages)
        logger.info("Generation provider used: %s", provider.name)
        return raw
    except Exception as exc:  # noqa: BLE001 — decide fallback vs re-raise
        if provider.name != "ollama_cloud":
            raise
        logger.warning(
            "Ollama Cloud unavailable (%s: %s); falling back to local llama.cpp.",
            type(exc).__name__,
            exc,
        )
        from app.llm.llama_cpp_provider import LlamaCppProvider

        fallback = LlamaCppProvider()
        raw = fallback.generate(messages)
        logger.info(
            "Generation provider used: %s (fallback from ollama_cloud)", fallback.name
        )
        return raw


def answer(message: str, history: list[dict] | None = None) -> dict:
    """Return {"answer", "context"} for one conversational turn."""
    chunks = retrieval.retrieve(message, top_k=settings.RETRIEVAL_K)

    # No relevant context -> do not call the LLM, do not invent an answer.
    if not chunks:
        return {"answer": prompting.no_context_message(message), "context": []}

    messages = prompting.build_messages(message, chunks, history)
    raw = _generate_with_fallback(messages)

    text = _THINK_RE.sub("", raw).strip()
    if not text:
        # Model produced nothing usable — fall back rather than invent.
        text = prompting.no_context_message(message)
        return {"answer": text, "context": []}

    return {"answer": text, "context": _citations(chunks)}
