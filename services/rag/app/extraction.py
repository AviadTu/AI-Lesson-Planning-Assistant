"""
Text extraction for uploaded source documents.

Supports TXT, DOCX and PDF, preserving Hebrew (and other Unicode) text, and
keeping a page number where the format exposes one (PDF).  TXT and DOCX have no
intrinsic page concept, so their page number is ``None``.

Extraction returns a list of ``Segment`` objects.  A segment is a natural unit
of the document (one PDF page, or the whole TXT/DOCX body); segments are chunked
downstream (see app/chunking.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class ExtractionError(RuntimeError):
    """Raised when a document cannot be parsed/extracted."""


@dataclass
class Segment:
    text: str
    page_number: int | None


def _extension(filename: str) -> str:
    _, dot, ext = filename.rpartition(".")
    return ext.lower() if dot else ""


def extract_segments(path: str | Path, original_filename: str) -> list[Segment]:
    """
    Extract text segments from a document, dispatching on its extension.

    Raises ``ExtractionError`` for unsupported types or parse failures.
    """
    ext = _extension(original_filename) or _extension(str(path))
    try:
        if ext == "txt":
            return _extract_txt(path)
        if ext == "docx":
            return _extract_docx(path)
        if ext == "pdf":
            return _extract_pdf(path)
    except ExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001 — normalise parser errors
        raise ExtractionError(f"Failed to extract '{original_filename}': {exc}") from exc

    raise ExtractionError(f"Unsupported file type for extraction: '{ext}'.")


def _extract_txt(path: str | Path) -> list[Segment]:
    """Read a TXT file, preferring UTF-8 and falling back to Windows-1255."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("windows-1255", errors="replace")
    text = text.strip()
    return [Segment(text=text, page_number=None)] if text else []


def _extract_docx(path: str | Path) -> list[Segment]:
    """Extract paragraph text from a DOCX (Hebrew preserved). No page numbers."""
    from docx import Document  # python-docx

    document = Document(str(path))
    parts = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    text = "\n".join(parts).strip()
    return [Segment(text=text, page_number=None)] if text else []


def _extract_pdf(path: str | Path) -> list[Segment]:
    """Extract text per page from a PDF, keeping the 1-based page number."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    segments: list[Segment] = []
    for index, page in enumerate(reader.pages):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            segments.append(Segment(text=page_text, page_number=index + 1))
    return segments
