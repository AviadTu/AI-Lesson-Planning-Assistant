"""
FastAPI Gateway — application entry point.

Responsibilities
  * Serve the existing frontend (index.html / logout.html) on the same origin.
  * Expose the public API the frontend already calls (relative paths preserved).
  * Own SQLite (sessions + chat history).
  * Proxy chat to the RAG service and document operations to the RAG service;
    trigger n8n workflows (lesson creation / export) when wired up.

Business logic (retrieval, generation, guardrails, image analysis, n8n
workflows) lives in the downstream services and is not implemented here yet.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.config import settings
from app.database import init_db
from app.routers import chat, documents, pages, status

app = FastAPI(title="Homori Gateway", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "gateway"}


# Public routes (relative paths the frontend already uses).
app.include_router(pages.router)
app.include_router(chat.router)
app.include_router(status.router)
app.include_router(documents.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.GATEWAY_HOST,
        port=settings.GATEWAY_PORT,
        reload=settings.GATEWAY_DEBUG,
    )
