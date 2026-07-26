"""
/query route — retrieval-augmented generation for one conversational turn.

Runs the full RAG flow: retrieve top-K chunks -> build prompt -> generate with
the local GGUF model (llama.cpp) -> attach citations from retrieval metadata.
Contract: {session_id, message, history} -> {answer, context}.

Sync route: the blocking model call runs in a worker thread, never on the event
loop.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app import generation, retrieval
from app.schemas import QueryRequest, QueryResponse

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest):
    try:
        return generation.answer(
            payload.message,
            [turn.model_dump() for turn in payload.history],
        )
    except retrieval.EmptyQueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except retrieval.EmbeddingUnavailableError as exc:
        raise HTTPException(
            status_code=503, detail=f"Embedding service unavailable: {exc}"
        ) from exc
