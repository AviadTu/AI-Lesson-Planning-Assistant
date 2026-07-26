"""
Document metadata registry (RAG-owned SQLite).

Migrated from the previous ``database/models.py`` ``uploaded_documents`` table.
Maps ``doc_id`` -> original filename + upload timestamp so the UI can show
human-friendly labels (including Hebrew) while internal code uses the id.

Idempotent on ``doc_id``: re-uploading the same file updates the record rather
than raising on the UNIQUE constraint (mirrors the file overwrite behaviour).
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from app.config import settings


@contextmanager
def _get_db():
    db_path = settings.REGISTRY_DB_PATH
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with _get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                original_filename TEXT    NOT NULL,
                doc_id            TEXT    NOT NULL UNIQUE,
                upload_timestamp  TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_documents_doc_id
                ON documents (doc_id);
            """
        )


def record_document(original_filename: str, doc_id: str) -> None:
    with _get_db() as conn:
        conn.execute(
            """
            INSERT INTO documents (original_filename, doc_id, upload_timestamp)
            VALUES (?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                original_filename = excluded.original_filename,
                upload_timestamp  = excluded.upload_timestamp
            """,
            (original_filename, doc_id, _now()),
        )


def list_documents() -> list[dict]:
    """Return all document records, newest first."""
    with _get_db() as conn:
        rows = conn.execute(
            """
            SELECT original_filename, doc_id, upload_timestamp
            FROM documents
            ORDER BY id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def delete_document(doc_id: str) -> bool:
    with _get_db() as conn:
        cursor = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        return cursor.rowcount > 0


def get_display_name(doc_id: str) -> str | None:
    with _get_db() as conn:
        row = conn.execute(
            "SELECT original_filename FROM documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
    return row["original_filename"] if row else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
