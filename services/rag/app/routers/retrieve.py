"""
Internal /retrieve route.

Returns the most relevant chunks for a query (embedding + ChromaDB search +
threshold filtering). No LLM / answer generation is involved. Intended for
internal use and tests, not exposed publicly through the Gateway yet.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app import retrieval
from app.config import settings
from app.schemas import RetrieveRequest, RetrieveResponse

router = APIRouter()


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(payload: RetrieveRequest):
    # Sync route: the (blocking) Chroma + Ollama calls run in a worker thread,
    # never on the event loop.
    top_k = settings.RETRIEVAL_K if payload.top_k is None else payload.top_k
    threshold = (
        settings.SIMILARITY_THRESHOLD
        if payload.threshold is None
        else payload.threshold
    )
    try:
        chunks = retrieval.retrieve(payload.query, top_k=top_k, threshold=threshold)
    except retrieval.EmptyQueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except retrieval.EmbeddingUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Embedding service unavailable: {exc}",
        ) from exc

    return RetrieveResponse(
        query=payload.query.strip(),
        top_k=top_k,
        threshold=threshold,
        count=len(chunks),
        chunks=chunks,
    )
