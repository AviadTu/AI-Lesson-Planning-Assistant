"""
Optional Langfuse tracing (feature-flagged).

A strict no-op unless ``LANGFUSE_ENABLED`` is set AND public/secret keys are
present AND the ``langfuse`` SDK is installed. Any error initializing or
emitting a trace is swallowed — observability must never block or break a
lesson-planning request. Wrapped defensively so it works across Langfuse SDK
versions (the emit call is best-effort).
"""

from __future__ import annotations

import logging

from app.config import settings

logger = logging.getLogger("lesson_agent.tracing")

_client = None
_initialized = False


def _get_client():
    global _client, _initialized
    if _initialized:
        return _client
    _initialized = True
    if not (
        settings.LANGFUSE_ENABLED
        and settings.LANGFUSE_PUBLIC_KEY
        and settings.LANGFUSE_SECRET_KEY
    ):
        return None
    try:
        from langfuse import Langfuse  # type: ignore

        _client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
        logger.info("Langfuse tracing enabled (host=%s)", settings.LANGFUSE_HOST)
    except Exception as exc:  # noqa: BLE001 — never let tracing break the app
        logger.warning("Langfuse unavailable (%s); tracing disabled.", exc)
        _client = None
    return _client


def enabled() -> bool:
    return _get_client() is not None


def log_generation(name: str, model: str, prompt: str, output: str) -> None:
    """Best-effort record of one LLM generation; no-op when disabled."""
    client = _get_client()
    if client is None:
        return
    try:
        # Prefer the modern context API; fall back to the classic method.
        if hasattr(client, "start_generation"):
            gen = client.start_generation(name=name, model=model, input=prompt)
            gen.update(output=output)
            gen.end()
        elif hasattr(client, "generation"):
            client.generation(name=name, model=model, input=prompt, output=output)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Langfuse log failed (%s); ignoring.", exc)
