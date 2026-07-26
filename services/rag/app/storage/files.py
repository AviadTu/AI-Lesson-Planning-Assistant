"""
Local file storage for uploaded source documents.

Migrated from the previous ``storage/uploads.py`` (which streamed to S3).  All
behavioural details are preserved:

  * extension allow-list (txt, pdf, docx);
  * Hebrew / Unicode filenames preserved (only path/control chars stripped);
  * TXT files re-encoded Windows-1255 -> UTF-8 so Hebrew is not garbled;
  * re-uploading the same name overwrites the existing file;
  * pathological names fall back to ``document_<uuid>``.

The ``doc_id`` is the sanitised filename (no UUID prefix), so it appears with
its real name in the document list and in citations — exactly as the S3 key
did before.  A path-traversal guard replaces the old S3-prefix check.
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


def _sanitise(name: str) -> str:
    """
    Preserve visible characters (incl. Hebrew) while removing path- and
    control-unsafe characters: slashes, NUL/control chars, surrounding
    whitespace.
    """
    stripped = "".join(
        ch
        for ch in name
        if ch not in ("/", "\\") and (ord(ch) >= 32 and ord(ch) != 127)
    )
    return stripped.strip()


def build_doc_id(original_filename: str) -> str:
    """Build a clean, filesystem-safe document id from the filename."""
    safe_name = _sanitise(original_filename)
    if not safe_name:
        ext = _extension(original_filename)
        token = uuid.uuid4().hex
        safe_name = f"document_{token}.{ext}" if ext else f"document_{token}"
    return safe_name


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
