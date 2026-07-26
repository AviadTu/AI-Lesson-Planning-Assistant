"""Pydantic request/response models for the Gateway public API."""

from __future__ import annotations

from pydantic import BaseModel


# ── /chat ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str


class Source(BaseModel):
    source: str
    doc_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    context: list[Source] = []


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
