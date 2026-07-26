"""
RAG service — application entry point.

Owns uploaded source files, the document metadata registry and ChromaDB. The
ingestion pipeline (extract → chunk → embed via Ollama → store) and retrieval
(embed query → ChromaDB search → threshold filtering) are implemented; answer
generation (behind an LLM provider abstraction) is deferred. Exposes document
management, retrieval, status and diagnostics endpoints.
"""

from __future__ import annotations

import threading

from fastapi import FastAPI

from app.config import settings
from app.routers import diagnostics, documents, query, retrieve, status
from app.storage import registry
from app.vectorstore import chroma

app = FastAPI(title="Homori RAG Service", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    registry.init_db()
    # Warm the Chroma client on a background thread so the first request does
    # not pay the (slow) import/open cost, and so client init never runs on the
    # event loop (Chroma's blocking init deadlocks there).
    threading.Thread(target=_warm_chroma, name="chroma-warmup", daemon=True).start()


def _warm_chroma() -> None:
    try:
        chroma.get_collection()
    except Exception:  # noqa: BLE001 — warmup is best-effort
        pass


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "rag"}


app.include_router(query.router)
app.include_router(retrieve.router)
app.include_router(status.router)
app.include_router(documents.router)
app.include_router(diagnostics.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.RAG_HOST,
        port=settings.RAG_PORT,
        reload=settings.RAG_DEBUG,
    )
