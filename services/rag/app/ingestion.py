"""
Ingestion orchestrator.

Pipeline: extract → chunk → embed (Ollama) → store (Chroma), while tracking the
status lifecycle (ingesting → ready | failed). A document is never marked
``ready`` until all of its chunks are embedded and written to Chroma.

Ingestion runs on a background thread so uploads return promptly and the
frontend can poll /status, matching the previous behaviour.
"""

from __future__ import annotations

import logging
import threading

from app import status as status_mgr
from app.chunking import chunk_document
from app.config import settings
from app.embeddings import ollama_embeddings
from app.extraction import extract_segments
from app.vectorstore import chroma

logger = logging.getLogger("rag.ingestion")


def ingest_document(doc_id: str, original_filename: str, path: str) -> None:
    """Run the full ingestion pipeline synchronously for one document."""
    status_mgr.start(doc_id)
    try:
        segments = extract_segments(path, original_filename)
        chunks = chunk_document(
            segments, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
        )
        texts = [chunk.text for chunk in chunks]
        embeddings = ollama_embeddings.embed_texts(texts)
        stored = chroma.replace_document(
            doc_id, original_filename, chunks, embeddings
        )
        status_mgr.complete(doc_id, stored)
        logger.info("Ingested '%s' (%s): %d chunks", original_filename, doc_id, stored)
    except Exception as exc:  # noqa: BLE001 — record failure, keep service up
        logger.exception("Ingestion failed for '%s' (%s)", original_filename, doc_id)
        status_mgr.fail(doc_id, str(exc))


def ingest_document_async(doc_id: str, original_filename: str, path: str) -> None:
    """Schedule ingestion on a daemon thread and return immediately."""
    # Mark ingesting synchronously so an immediate /status poll sees progress
    # even before the thread is scheduled.
    status_mgr.start(doc_id)
    thread = threading.Thread(
        target=ingest_document,
        args=(doc_id, original_filename, path),
        name=f"ingest-{doc_id}",
        daemon=True,
    )
    thread.start()
