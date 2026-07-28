"""
Lightweight text extraction for lesson-source documents (Gateway side).

Used by the ``/internal/extract`` endpoint, which n8n's Source Processing
document branch calls for DOCX (no native n8n DOCX extractor) and can use as a
fallback for PDF/TXT. Mirrors the RAG service's extraction behaviour (Hebrew
preserved) but stays independent — the RAG service is not modified.
"""

from __future__ import annotations

import io


class ExtractError(RuntimeError):
    """Raised when a document cannot be parsed."""


def _ext(filename: str) -> str:
    _, dot, ext = (filename or "").rpartition(".")
    return ext.lower() if dot else ""


def extract_text(filename: str, mime: str, data: bytes) -> str:
    """Extract text from a document by (filename/mime) type. Raises ExtractError."""
    ext = _ext(filename)
    mime = (mime or "").lower()
    try:
        if ext == "txt" or mime.startswith("text/"):
            try:
                return data.decode("utf-8").strip()
            except UnicodeDecodeError:
                return data.decode("windows-1255", errors="replace").strip()
        if ext == "docx" or "word" in mime or "officedocument" in mime:
            from docx import Document  # python-docx

            doc = Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip()).strip()
        if ext == "pdf" or "pdf" in mime:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            parts = [(page.extract_text() or "").strip() for page in reader.pages]
            return "\n".join(p for p in parts if p).strip()
    except Exception as exc:  # noqa: BLE001 — normalise parser errors
        raise ExtractError(f"Failed to extract '{filename}': {exc}") from exc
    raise ExtractError(f"Unsupported file type: '{ext or mime}'.")
