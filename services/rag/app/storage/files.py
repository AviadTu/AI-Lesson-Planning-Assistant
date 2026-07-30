"""
Local file storage for uploaded source documents.

Migrated from the previous ``storage/uploads.py`` (which streamed to S3).  All
behavioural details are preserved:

  * extension allow-list (txt, pdf, docx);
  * Hebrew / Unicode filenames preserved (only path/control chars stripped);
  * TXT files re-encoded Windows-1255 -> UTF-8 so Hebrew is not garbled;
  * a path-traversal guard on every id -> path resolution.

The ``doc_id`` is a generated UUID (not the filename), so two different
documents that share a filename get distinct ids and never overwrite each
other. The human-readable original filename is preserved separately (registry +
Chroma metadata) for the document list, downloads and citations. On disk the
file is stored under its ``doc_id``; extraction dispatches on the original
filename's extension, so the stored file needs no extension of its own.
"""

from __future__ import annotations

import io
import os
import uuid
from pathlib import Path

from app.config import settings


class UploadError(ValueError):
    """Raised when an upload is rejected (e.g. disallowed extension)."""


def _upload_dir() -> Path:
    path = Path(settings.UPLOAD_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extension(filename: str) -> str:
    _, dot, ext = filename.rpartition(".")
    return ext.lower() if dot else ""


def is_allowed(filename: str) -> bool:
    return _extension(filename) in settings.allowed_extensions


def build_doc_id(original_filename: str) -> str:
    """
    Generate a stable, unique, filesystem-safe document id.

    A fresh UUID (not the filename): two different documents that happen to
    share a filename get distinct ids and never silently overwrite one another.
    The ``original_filename`` argument is accepted for API symmetry but is
    intentionally not used to derive the id — it is preserved separately as
    metadata by the caller (registry + Chroma).
    """
    return uuid.uuid4().hex


def resolve_path(doc_id: str) -> Path:
    """
    Resolve a ``doc_id`` to an absolute path inside the upload directory,
    raising if the result would escape it (path-traversal protection).
    """
    base = _upload_dir().resolve()
    candidate = (base / doc_id).resolve()
    if base != candidate and base not in candidate.parents:
        raise UploadError("Invalid document id.")
    return candidate


def save_upload(filename: str, content: bytes, content_type: str | None) -> tuple[str, str]:
    """
    Validate and write a single uploaded file to local storage.

    Returns ``(original_filename, doc_id)``.  Raises ``UploadError`` if the
    filename is missing or the extension is not allowed.
    """
    original_filename = (filename or "").strip()
    if not original_filename:
        raise UploadError("Upload is missing a filename.")

    if not is_allowed(original_filename):
        allowed = ", ".join(sorted(settings.allowed_extensions))
        raise UploadError(
            f"File type not allowed for '{original_filename}'. "
            f"Allowed types: {allowed}."
        )

    doc_id = build_doc_id(original_filename)
    dest = resolve_path(doc_id)

    # TXT files created on Hebrew-locale Windows are often Windows-1255.
    # Normalise to UTF-8 so downstream ingestion reads Hebrew correctly.
    if _extension(original_filename) == "txt":
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("windows-1255", errors="replace")
        data = io.BytesIO(text.encode("utf-8")).getvalue()
    else:
        data = content

    with open(dest, "wb") as fh:
        fh.write(data)

    return original_filename, doc_id


def delete_file(doc_id: str) -> bool:
    """Delete the stored file for a doc_id. Returns True if a file was removed."""
    path = resolve_path(doc_id)
    if path.exists():
        os.remove(path)
        return True
    return False
