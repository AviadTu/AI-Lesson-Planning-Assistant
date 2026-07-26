"""Pydantic request/response models for the RAG service API."""

from __future__ import annotations

from pydantic import BaseModel


# ── /query ───────────────────────────────────────────────────────────
class HistoryTurn(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    session_id: str
    message: str
    history: list[HistoryTurn] = []


class Source(BaseModel):
    source: str
    doc_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    context: list[Source] = []


# ── /documents ───────────────────────────────────────────────────────
class Document(BaseModel):
    original_filename: str
    doc_id: str
    upload_timestamp: str | None = None


class DocumentsResponse(BaseModel):
    documents: list[Document]


class UploadedItem(BaseModel):
    original_filename: str
    doc_id: str


class UploadError(BaseModel):
    filename: str
    error: str


class UploadResponse(BaseModel):
    uploaded: list[UploadedItem] = []
    errors: list[UploadError] = []
    ingesting: bool = False


class DeleteDocumentRequest(BaseModel):
    doc_id: str


# ── /status ──────────────────────────────────────────────────────────
class StatusResponse(BaseModel):
    ready: bool = True
    status: str = "ready"
    ingestion: dict | None = None


# ── /retrieve ────────────────────────────────────────────────────────
class RetrieveRequest(BaseModel):
    query: str
    top_k: int | None = None
    threshold: float | None = None


class RetrievedChunk(BaseModel):
    text: str
    doc_id: str | None = None
    original_filename: str | None = None
    chunk_index: int | None = None
    page_number: int | None = None
    similarity: float


class RetrieveResponse(BaseModel):
    query: str
    top_k: int
    threshold: float
    count: int
    chunks: list[RetrievedChunk] = []


# ── /diagnostics ─────────────────────────────────────────────────────
class DiagnosticsResponse(BaseModel):
    collection_name: str
    embedding_model: str
    document_count: int
    chunk_count: int
    chunk_size: int
    chunk_overlap: int
    ollama: dict
    ingestion_status: dict
    documents: list[dict] = []
