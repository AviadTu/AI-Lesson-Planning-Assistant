"""
Conversation-memory helpers (migrated from the previous ``chat_service.py``).

Two responsibilities, both preserved from the original project:

1. ``try_local_memory_answer`` — answers trivial self-referential Hebrew
   questions ("what's my name?", "what did I ask before?") directly from
   SQLite, without a round-trip to the RAG service.

2. ``build_history_window`` — assembles a short, bounded transcript of the
   most recent turns.  In the previous architecture Bedrock kept conversation
   memory natively via ``sessionId``.  A local GGUF model is stateless per
   call, so this window is the destination for that memory: the Gateway sends
   it to the RAG service on every turn so the model can resolve follow-ups.
"""

from __future__ import annotations

import re

from app.config import settings
from app.database import get_history

# Trailing/closing punctuation that should not affect intent detection.
_TRIM_CHARS = " \t\n\r?.!,;:،؟"

# "what is my name?"
_NAME_QUESTION_RE = re.compile(
    r"(?:איך\s+קוראים\s+לי|מה\s+שמי|מה\s+השם\s+שלי)",
    re.UNICODE,
)

# "what did I ask before?" / "what did I just say?"
_PREV_QUESTION_RE = re.compile(
    r"(?:"
    r"מה\s+השאלה\s+(?:האחרונה|הקודמת)(?:\s+ששאלתי)?"
    r"|מה\s+שאלתי(?:\s+קודם|\s+לפני\s+זה|\s+קודם\s+לכן)?"
    r"|מה\s+אמרתי(?:\s+לך)?(?:\s+הרגע|\s+קודם|\s+לפני\s+זה)?"
    r")",
    re.UNICODE,
)

# Name introductions: "קוראים לי X", "שמי X", "השם שלי X" (optional "הוא").
_NAME_EXTRACT_RE = re.compile(
    r"(?:קוראים\s+לי|שמי(?:\s+הוא)?|השם\s+שלי(?:\s+הוא)?)\s+"
    r"([^\s.!?,;:\n]+(?:\s+[^\s.!?,;:\n]+){0,2})",
    re.UNICODE,
)


def try_local_memory_answer(session_id: str, user_message: str) -> str | None:
    """
    Return a locally-computed answer for trivial memory questions, or ``None``
    to indicate the caller should fall through to the RAG service.
    """
    normalized = (user_message or "").strip().strip(_TRIM_CHARS).strip()
    if not normalized:
        return None

    # 1) "What did I ask / say before?" – more specific intent first.
    if _PREV_QUESTION_RE.search(normalized):
        current = (user_message or "").strip()
        prior_user_msgs = [
            (m.get("content") or "").strip()
            for m in get_history(session_id)
            if m.get("role") == "user"
            and (m.get("content") or "").strip() != current
        ]
        if prior_user_msgs:
            last_question = prior_user_msgs[-1]
            return f'השאלה הקודמת ששאלת הייתה: "{last_question}"'

    # 2) "What is my name?" – scan user messages newest-to-oldest.
    if _NAME_QUESTION_RE.search(normalized):
        for msg in reversed(get_history(session_id)):
            if msg.get("role") != "user":
                continue
            match = _NAME_EXTRACT_RE.search(msg.get("content") or "")
            if not match:
                continue
            name = match.group(1).strip().strip(_TRIM_CHARS).strip()
            if name:
                return f"שמך {name}."

    return None


def build_history_window(session_id: str) -> list[dict]:
    """
    Return the last few raw turns as ``[{"role", "content"}, ...]``, each
    clipped to ``HISTORY_MAX_CHARS`` and capped at ``HISTORY_MAX_MESSAGES``.

    Sent to the RAG service so the (stateless) model can resolve context.
    Only raw originals are used, so there is no recursive prompt growth.
    """
    prior = get_history(session_id)[-settings.HISTORY_MAX_MESSAGES:]
    window: list[dict] = []
    for msg in prior:
        content = (msg.get("content") or "").strip()
        if len(content) > settings.HISTORY_MAX_CHARS:
            content = content[: settings.HISTORY_MAX_CHARS].rstrip() + "…"
        window.append({"role": msg.get("role", ""), "content": content})
    return window
