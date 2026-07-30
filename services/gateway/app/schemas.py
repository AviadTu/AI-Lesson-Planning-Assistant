"""Pydantic request/response models for the Gateway public API."""

from __future__ import annotations

from pydantic import BaseModel


# ── /chat ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str


class Source(BaseModel):
    source: str
    doc_id: str | None = None


class LessonState(BaseModel):
    active: bool = False
    pending_offer: bool = False


class ChatResponse(BaseModel):
    answer: str
    context: list[Source] = []
    # Post-turn lesson-planning state, so the frontend can show/hide the
    # temporary lesson-source panel (shown while active or a pending offer).
    lesson: LessonState | None = None


# ── /history ─────────────────────────────────────────────────────────
class HistoryMessage(BaseModel):
    id: int
    role: str
    content: str
    retrieved: list[Source] | None = None
    created_at: str


class HistoryResponse(BaseModel):
    messages: list[HistoryMessage]


# ── /documents ───────────────────────────────────────────────────────
class DeleteDocumentRequest(BaseModel):
    doc_id: str


# ── /internal/pdf ────────────────────────────────────────────────────
class PdfRequest(BaseModel):
    lesson: dict
    session_id: str | None = None


# ── /internal/extract ────────────────────────────────────────────────
class ExtractRequest(BaseModel):
    filename: str | None = None
    mime: str | None = None
    content_base64: str
