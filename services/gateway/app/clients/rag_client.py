"""
HTTP client for the RAG service.

Synchronous REST/JSON for chat + document metadata; multipart for uploads and
a streamed binary passthrough for downloads.  All calls target
``settings.RAG_SERVICE_URL``.
"""

from __future__ import annotations

import httpx

from app.config import settings

_BASE = settings.RAG_SERVICE_URL.rstrip("/")
_TIMEOUT = settings.HTTP_TIMEOUT


async def query(session_id: str, message: str, history: list[dict]) -> dict:
    """Ask the RAG service for an answer. Returns {"answer", "context"}."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{_BASE}/query",
            json={"session_id": session_id, "message": message, "history": history},
        )
        resp.raise_for_status()
        return resp.json()


async def status() -> dict:
    """Return the RAG ingestion/readiness status."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{_BASE}/status")
        resp.raise_for_status()
        return resp.json()


async def list_documents() -> dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{_BASE}/documents")
        resp.raise_for_status()
        return resp.json()


async def upload_documents(files: list[tuple[str, bytes, str]]) -> dict:
    """
    Forward uploaded files to the RAG service as multipart/form-data.

    ``files`` is a list of ``(filename, content, content_type)`` tuples.
    """
    multipart = [
        ("files", (filename, content, content_type))
        for filename, content, content_type in files
    ]
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(f"{_BASE}/documents/upload", files=multipart)
        return _json_or_raise(resp)


async def delete_document(doc_id: str) -> dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.request(
            "DELETE", f"{_BASE}/documents", json={"doc_id": doc_id}
        )
        return _json_or_raise(resp)


def download_stream(doc_id: str) -> httpx.Response:
    """
    Open a streamed GET to the RAG download endpoint.

    Returns an *open* streaming response; the caller is responsible for
    forwarding it and closing it (see routers/documents.py).
    """
    client = httpx.AsyncClient(timeout=_TIMEOUT)
    request = client.build_request(
        "GET", f"{_BASE}/documents/download", params={"doc_id": doc_id}
    )
    return client, request


def _json_or_raise(resp: httpx.Response) -> dict:
    """Return JSON for any response the RAG service produced (incl. 4xx)."""
    try:
        return resp.json()
    except ValueError:
        resp.raise_for_status()
        return {}
