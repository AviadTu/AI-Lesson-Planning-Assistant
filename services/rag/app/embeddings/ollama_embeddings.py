"""
Embeddings via a local Ollama instance (HTTP API).

Uses a dedicated embedding model (``settings.EMBEDDING_MODEL``, default
``nomic-embed-text``) — deliberately separate from the generation LLM. Ingestion
and (future) retrieval call this module; they never touch the LLM provider.
"""

from __future__ import annotations

import httpx

from app.config import settings


def _url(path: str) -> str:
    return settings.OLLAMA_EMBEDDINGS_BASE_URL.rstrip("/") + path


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Return one embedding vector per input text.

    Raises on any transport/HTTP error or malformed response so the caller can
    mark the ingestion job as failed.
    """
    vectors: list[list[float]] = []
    if not texts:
        return vectors

    with httpx.Client(timeout=settings.EMBEDDING_TIMEOUT) as client:
        for text in texts:
            resp = client.post(
                _url("/api/embeddings"),
                json={"model": settings.EMBEDDING_MODEL, "prompt": text},
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding")
            if not embedding:
                raise RuntimeError(
                    f"Ollama returned no embedding for model "
                    f"'{settings.EMBEDDING_MODEL}'."
                )
            vectors.append(embedding)
    return vectors


def embed_query(text: str) -> list[float]:
    """Embed a single query string (for future retrieval)."""
    return embed_texts([text])[0]


def health() -> dict:
    """Report Ollama connectivity and whether the embedding model is present."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(_url("/api/tags"))
            resp.raise_for_status()
            models = resp.json().get("models", []) or []
            names = [m.get("name", "") for m in models]
            wanted = settings.EMBEDDING_MODEL
            present = any(n == wanted or n.startswith(wanted + ":") for n in names)
            return {
                "connected": True,
                "base_url": settings.OLLAMA_EMBEDDINGS_BASE_URL,
                "embedding_model": wanted,
                "embedding_model_present": present,
            }
    except Exception as exc:  # noqa: BLE001
        return {
            "connected": False,
            "base_url": settings.OLLAMA_EMBEDDINGS_BASE_URL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_model_present": False,
            "error": str(exc),
        }
