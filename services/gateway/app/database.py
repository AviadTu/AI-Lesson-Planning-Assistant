"""
Gateway SQLite persistence layer (sessions + chat history).

Migrated from the previous ``database/models.py``.  The Gateway is the sole
owner of application state (sessions and messages).  Document metadata now
lives in the RAG service, so the legacy ``uploaded_documents`` table has been
removed from here.

Schema
──────
sessions
  session_id  TEXT  PRIMARY KEY
  created_at  TEXT  ISO-8601 timestamp

messages
  id              INTEGER  PRIMARY KEY AUTOINCREMENT
  session_id      TEXT     FK -> sessions.session_id
  role            TEXT     'user' | 'assistant'
  content         TEXT     message text
  retrieved_json  TEXT     JSON array of source references (assistant turns)
  created_at      TEXT     ISO-8601 timestamp
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from app.config import settings


@contextmanager
def _get_db():
    """Yield a sqlite3 connection; commit on success, roll back on error."""
    db_path = settings.SQLITE_PATH
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
    """Create tables if they do not already exist. Safe to call repeatedly."""
    with _get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT    NOT NULL,
                role            TEXT    NOT NULL,
                content         TEXT    NOT NULL,
                retrieved_json  TEXT,
                created_at      TEXT    NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages (session_id, id);
            """
        )


def _ensure_session(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, created_at) VALUES (?, ?)",
        (session_id, _now()),
    )


def save_message(
    session_id: str,
    role: str,
    content: str,
    retrieved: list[dict] | None = None,
) -> None:
    """Persist one message (user or assistant) to the database."""
    retrieved_json = json.dumps(retrieved, ensure_ascii=False) if retrieved else None
    with _get_db() as conn:
        _ensure_session(conn, session_id)
        conn.execute(
            """
            INSERT INTO messages (session_id, role, content, retrieved_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, role, content, retrieved_json, _now()),
        )


def get_history(session_id: str) -> list[dict]:
    """Return all messages for a session in chronological order."""
    with _get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, role, content, retrieved_json, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()

    history = []
    for row in rows:
        entry = dict(row)
        raw = entry.pop("retrieved_json", None)
        entry["retrieved"] = json.loads(raw) if raw else None
        history.append(entry)
    return history


def clear_session(session_id: str) -> None:
    """Delete all messages for a session (the session row is kept)."""
    with _get_db() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
