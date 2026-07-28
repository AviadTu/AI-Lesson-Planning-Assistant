"""
Chat routes: /chat, /history, /clear.

/chat is the single conversational entry point. The Gateway routes each turn
(see app/routing.py):

  * ordinary knowledge question  → RAG  (the existing, independent path)
  * lesson-plan create/continue  → n8n  (Lesson Planning Orchestrator)

The app feels like one intelligent chat: the user never switches modes. Once a
session enters lesson-planning it continues with n8n until a new chat/clear. If
RAG has no relevant answer, the Gateway OFFERS to build a lesson and waits for
the user's confirmation (never auto-starts generation).

Session isolation is preserved via the X-Session-Id header. The Gateway stays
lightweight: it routes and persists, but holds no lesson-planning logic.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse

from app import memory, routing
from app.clients import n8n_client, rag_client
from app.database import (
    clear_session,
    get_history,
    get_lesson_mode,
    pop_pending_sources,
    save_message,
    set_lesson_mode,
)
from app.schemas import ChatRequest, ChatResponse, HistoryResponse

router = APIRouter()

# Localised offer shown when RAG has nothing relevant (awaits user's yes/no).
_LESSON_OFFER = (
    "לא מצאתי מידע מספיק על כך במאגר הפנימי. "
    "תרצה/י שאבנה יחד איתך מערך שיעור חדש בנושא זה?"
)
# The RAG "no relevant information" sentinel (see RAG app/prompting.py).
_RAG_NO_INFO = "לא נמצא מידע רלוונטי"

# Deterministic greeting/thanks replies (no RAG/embeddings/n8n/agent/LLM).
_GREETING_REPLY = (
    "שלום! 👋 אני העוזר הפדגוגי. אפשר לשאול אותי שאלות על מאגר הידע, "
    "או לבקש שאבנה יחד איתך מערך שיעור בנושא כלשהו. במה אפשר לעזור?"
)
_THANKS_REPLY = "בשמחה! 😊 אם תרצה/י עוד עזרה — אני כאן."


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


def _rag_insufficient(answer: str, context: list) -> bool:
    return (not context) or (_RAG_NO_INFO in (answer or ""))


async def _handle_lesson(session_id: str, user_message: str, mode: dict) -> dict:
    """Forward a lesson-planning turn to n8n and persist the exchange."""
    # If the user just accepted a "build a lesson?" offer, seed n8n with the
    # original topic they asked about rather than the bare "yes".
    forward_message = user_message
    if mode.get("pending_offer") and routing.is_affirmative(user_message):
        forward_message = mode.get("pending_topic") or user_message

    set_lesson_mode(session_id, active=True, pending_offer=False, pending_topic="")
    save_message(session_id, role="user", content=user_message)

    # Attach any files/images uploaded during this lesson session, and flag an
    # explicit web-search request, so n8n's Source Processing can use them.
    attachments = [
        {
            "kind": s["kind"],
            "filename": s["filename"],
            "mime": s["mime"],
            "content_base64": s["content_base64"],
        }
        for s in pop_pending_sources(session_id)
    ]
    want_web = bool(routing.is_web_search_request(user_message))

    # A PDF/download/email request routes to the "Lesson PDF and Email" workflow;
    # everything else is a normal lesson-planning turn (Lesson Orchestrator).
    payload = {
        "session_id": session_id,
        "message": forward_message,
        "want_web": want_web,
        "attachments": attachments,
    }
    try:
        if routing.is_export_request(user_message):
            result = await n8n_client.trigger_export(payload)
        else:
            result = await n8n_client.trigger_lesson_creation(payload)
    except n8n_client.WorkflowNotConfigured:
        answer = (
            "מנגנון בניית מערכי השיעור עדיין אינו מחובר (n8n). "
            "אפשר בינתיים לשאול שאלות על המאגר."
        )
        save_message(session_id, role="assistant", content=answer, retrieved=[])
        return {"answer": answer, "context": []}
    except Exception:  # noqa: BLE001 — surface one safe error, stay in lesson mode
        answer = "אירעה תקלה בבניית מערך השיעור. נסה/י שוב בבקשה."
        save_message(session_id, role="assistant", content=answer, retrieved=[])
        return {"answer": answer, "context": []}

    answer = result.get("answer", "") or ""
    context = result.get("context", []) or []
    save_message(session_id, role="assistant", content=answer, retrieved=context)
    return {"answer": answer, "context": context}


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, x_session_id: str | None = Header(default=None)):
    user_message = (payload.message or "").strip()
    if not user_message:
        return JSONResponse({"error": "Message must not be empty."}, status_code=400)

    session_id = _resolve_session_id(x_session_id)

    # ── Deterministic greetings / thanks ─────────────────────────────
    # Answered instantly with no RAG, embeddings, Chroma, n8n, lesson-agent or
    # any LLM call. Keeps trivial turns well under a second.
    greeting = routing.greeting_kind(user_message)
    if greeting is not None:
        reply = _THANKS_REPLY if greeting == "thanks" else _GREETING_REPLY
        save_message(session_id, role="user", content=user_message)
        save_message(session_id, role="assistant", content=reply, retrieved=[])
        return {"answer": reply, "context": []}

    mode = get_lesson_mode(session_id)

    # ── Route: lesson planning (n8n) vs knowledge question (RAG) ──────
    if routing.decide_route(mode, user_message) == "lesson":
        return await _handle_lesson(session_id, user_message, mode)

    # ── RAG path (unchanged behaviour) ───────────────────────────────
    # Any stale "build a lesson?" offer is declined by moving on.
    if mode.get("pending_offer"):
        set_lesson_mode(session_id, pending_offer=False, pending_topic="")

    local_answer = memory.try_local_memory_answer(session_id, user_message)
    if local_answer is not None:
        save_message(session_id, role="user", content=user_message)
        save_message(session_id, role="assistant", content=local_answer, retrieved=[])
        return {"answer": local_answer, "context": []}

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

    # If RAG found nothing relevant, offer to build a lesson (await user's yes).
    if _rag_insufficient(answer, context):
        set_lesson_mode(session_id, pending_offer=True, pending_topic=user_message)
        answer = (answer + "\n\n" + _LESSON_OFFER).strip() if answer else _LESSON_OFFER

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
