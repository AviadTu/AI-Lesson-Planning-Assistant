"""
Internal diagnostics for the RAG ingestion pipeline.

GET /diagnostics — document count, chunk count, collection name, embedding
model, Ollama connection status, plus chunking config, current ingestion status
and a per-document summary. Intended for operators/tests, not the public UI.
"""

from __future__ import annotations

from fastapi import APIRouter

from app import status as status_mgr
from app.config import settings
from app.embeddings import ollama_embeddings
from app.schemas import DiagnosticsResponse
from app.vectorstore import chroma

router = APIRouter()


@router.get("/diagnostics", response_model=DiagnosticsResponse)
def diagnostics():
    # Sync route: FastAPI runs it in a worker thread, so the (blocking) Chroma
    # client calls never run on the event loop (which would deadlock).
    counts = chroma.counts()
    return DiagnosticsResponse(
        collection_name=settings.CHROMA_COLLECTION,
        embedding_model=settings.EMBEDDING_MODEL,
        document_count=counts["document_count"],
        chunk_count=counts["chunk_count"],
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        ollama=ollama_embeddings.health(),
        ingestion_status=status_mgr.snapshot(),
        documents=chroma.document_summaries(),
    )
