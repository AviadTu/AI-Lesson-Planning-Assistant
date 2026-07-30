"""
Prompt builder for RAG answer generation.

Assembles chat messages from the retrieved context, the (bounded) conversation
history and the user's question. Kept separate from the LLM provider so the
provider stays a thin backend and the prompt can evolve independently.

Rules encoded here:
  * Answer strictly from the provided document context.
  * Answer in the same language as the user's question (Hebrew included).
  * If the context does not contain the answer, say so — never invent.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "ענה על שאלת המשתמש רק לפי המסמכים שסופקו.\n"
    "\n"
    "התייחס לכל מסמך כמערך שלם, ולכותרות המשנה כחלקים פנימיים שלו.\n"
    "אל תציג תת־נושא, פעילות או סעיף כאילו הוא מסמך או נושא ראשי נפרד.\n"
    "אם אין במקורות תשובה מספקת, אמור זאת בקצרה.\n"
    "ציין את שמות המסמכים שעליהם הסתמכת."
)

# Localised "answer not found" text used when the context is unusable.
NO_CONTEXT_HE = "לא נמצא מידע רלוונטי במסמכים שהועלו."
NO_CONTEXT_EN = "No relevant information was found in the uploaded documents."


def is_hebrew(text: str) -> bool:
    """True if the text contains any Hebrew character."""
    return any("\u0590" <= ch <= "\u05FF" for ch in text or "")


def no_context_message(query: str) -> str:
    """Return the 'no relevant information' message in the query's language."""
    return NO_CONTEXT_HE if is_hebrew(query) else NO_CONTEXT_EN


def build_context_block(chunks: list[dict]) -> str:
    """Render retrieved chunks into a numbered, source-labelled context block."""
    parts: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        name = ch.get("original_filename") or ch.get("doc_id") or f"source {i}"
        page = ch.get("page_number")
        label = f"[{i}] {name}" + (f" (page {page})" if page is not None else "")
        parts.append(f"{label}\n{ch.get('text', '')}")
    return "\n\n".join(parts)


def build_messages(
    query: str,
    chunks: list[dict],
    history: list[dict] | None = None,
) -> list[dict]:
    """
    Build the chat messages for generation.

    ``history`` is an optional bounded list of prior turns ({role, content})
    supplied by the Gateway to preserve conversational memory.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    context_block = build_context_block(chunks)
    user_content = (
        "Context:\n"
        f"{context_block}\n\n"
        "Question:\n"
        f"{query}\n\n"
        "Answer using only the context above, in the same language as the "
        "question."
    )
    messages.append({"role": "user", "content": user_content})
    return messages
