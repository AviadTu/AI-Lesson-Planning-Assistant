"""
Chat routes: /chat, /history, /clear.

Orchestration flow for /chat:
  1. Validate the message and resolve the per-session id (X-Session-Id).
  2. Try a local memory answer (Hebrew self-referential shortcut).
  3. Otherwise call the RAG service with a bounded history window so the
     stateless model can resolve conversational context.
  4. Persist the raw user + assistant messages to SQLite.

Session isolation is preserved: every request carries the browser session id
in the X-Session-Id header, and all persistence is keyed by that id.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse

from app import memory
from app.clients import rag_client
from app.database import clear_session, get_history, save_message
from app.schemas import ChatRequest, ChatResponse, HistoryResponse

router = APIRouter()


def _resolve_session_id(x_session_id: str | None) -> str:
    """Return a valid session id from the header, or a fresh UUID."""
    sid = (x_session_id or "").strip()
    if sid:
        try:
            uuid.UUID(sid)
            return sid
        except (ValueError, AttributeError, TypeError):
            pass
    return str(uuid.uuid4())


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    x_session_id: str | None = Header(default=None),
):
    user_message = (payload.message or "").strip()
    if not user_message:
        return JSONResponse({"error": "Message must not be empty."}, status_code=400)

    session_id = _resolve_session_id(x_session_id)

    # ── Local conversation-memory shortcut ───────────────────────────
    local_answer = memory.try_local_memory_answer(session_id, user_message)
    if local_answer is not None:
        save_message(session_id, role="user", content=user_message)
        save_message(session_id, role="assistant", content=local_answer, retrieved=[])
        return {"answer": local_answer, "context": []}

    # ── RAG round-trip (bounded history window carries the memory) ────
    history_window = memory.build_history_window(session_id)
    try:
        result = await rag_client.query(session_id, user_message, history_window)
    except Exception:  # noqa: BLE001 — surface a single safe error to the UI
        return JSONResponse(
            {"error": "An internal error occurred. Please try again."},
            status_code=500,
        )

    answer = result.get("answer", "")
    context = result.get("context", []) or []

    save_message(session_id, role="user", content=user_message)
    save_message(session_id, role="assistant", content=answer, retrieved=context)

    return {"answer": answer, "context": context}


@router.get("/history", response_model=HistoryResponse)
async def history(x_session_id: str | None = Header(default=None)):
    session_id = _resolve_session_id(x_session_id)
    return {"messages": get_history(session_id)}


@router.post("/clear")
async def clear(x_session_id: str | None = Header(default=None)):
    session_id = _resolve_session_id(x_session_id)
    clear_session(session_id)
    return {"ok": True}
