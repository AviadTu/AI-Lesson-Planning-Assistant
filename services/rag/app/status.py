"""
Ingestion status lifecycle (thread-safe, in-memory).

Internal states: ``ready`` | ``ingesting`` | ``failed``.

The public snapshot maps these to the status strings the frontend already
understands (``ready`` / ``ingesting`` / ``ingestion_failed``) so existing UI
polling behaviour is preserved. ``ready`` is only reported once no ingestion is
in progress and the last job did not fail — i.e. never before ingestion
completes.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_ingesting: set[str] = set()
_state: str = "ready"          # ready | ingesting | failed
_last_document: str | None = None
_last_error: str | None = None
_last_chunk_count: int | None = None

# Internal -> frontend-compatible status string.
_STATUS_MAP = {
    "ready": "ready",
    "ingesting": "ingesting",
    "failed": "ingestion_failed",
}


def start(doc_id: str) -> None:
    global _state, _last_document, _last_error
    with _lock:
        _ingesting.add(doc_id)
        _state = "ingesting"
        _last_document = doc_id
        _last_error = None


def complete(doc_id: str, chunk_count: int) -> None:
    global _state, _last_chunk_count
    with _lock:
        _ingesting.discard(doc_id)
        _last_chunk_count = chunk_count
        if not _ingesting and _state != "failed":
            _state = "ready"


def fail(doc_id: str, error: str) -> None:
    global _state, _last_document, _last_error
    with _lock:
        _ingesting.discard(doc_id)
        _state = "failed"
        _last_document = doc_id
        _last_error = error


def snapshot() -> dict:
    """Return the status payload for /status (frontend-compatible)."""
    with _lock:
        internal = _state
        return {
            "ready": internal == "ready",
            "status": _STATUS_MAP[internal],
            "ingestion": {
                "state": internal,
                "in_progress": sorted(_ingesting),
                "last_document": _last_document,
                "last_error": _last_error,
                "last_chunk_count": _last_chunk_count,
            },
        }
