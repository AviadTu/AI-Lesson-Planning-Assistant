"""
Local Hebrew/RTL lesson-plan PDF rendering.

The simplest reliable local approach: ``fpdf2`` with HarfBuzz text shaping
(``uharfbuzz``), which handles Hebrew bidi/RTL correctly, plus a Unicode TTF
that contains Hebrew glyphs. No external service or headless browser needed.

The font is configurable (``PDF_FONT_PATH``); it defaults to the system Arial on
Windows for the local demo. In a Linux container, mount/point it at a Hebrew
TTF (e.g. Noto Sans Hebrew). Rendering degrades to a clear error if no usable
font is found — callers handle that as a recoverable PDF failure.
"""

from __future__ import annotations

from fpdf import FPDF

from app.config import settings

# Section order + Hebrew headings — mirrors the lesson-agent's approved template.
_SECTIONS: list[tuple[str, str]] = [
    ("opening", "פתיחה"),
    ("central_idea", "רעיון מרכזי"),
    ("case_study", "מקרה בוחן"),
    ("activity", "פעילות"),
    ("central_discussion", "דיון מרכזי"),
    ("source", "מקור"),
    ("summary", "סיכום"),
    ("personal_writing", "כתיבה אישית"),
]


class PdfRenderError(RuntimeError):
    """Raised when a lesson PDF cannot be produced (e.g. no usable font)."""


def _stringify(value) -> str:
    """Render a section value (str / list / dict) into readable Hebrew text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(f"• {_stringify(v)}" for v in value if v)
    if isinstance(value, dict):
        return "\n".join(f"{k}: {_stringify(v)}" for k, v in value.items() if v)
    return str(value)


def render_lesson_pdf(lesson: dict) -> bytes:
    """Render a structured lesson dict to a Hebrew/RTL PDF; return the bytes."""
    if not isinstance(lesson, dict) or not lesson:
        raise PdfRenderError("Empty lesson: nothing to render.")

    pdf = FPDF()
    pdf.set_margins(16, 16, 16)
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    try:
        pdf.add_font("heb", "", settings.PDF_FONT_PATH)
    except Exception as exc:  # noqa: BLE001 — missing/unreadable font
        raise PdfRenderError(
            f"Hebrew font not available at '{settings.PDF_FONT_PATH}': {exc}"
        ) from exc
    pdf.set_text_shaping(True)

    def para(text: str, size: int, gap: int = 8, spacing: int = 0) -> None:
        text = (text or "").strip()
        if not text:
            return
        pdf.set_font("heb", size=size)
        pdf.multi_cell(pdf.epw, gap, text, align="R", new_x="LMARGIN", new_y="NEXT")
        if spacing:
            pdf.ln(spacing)

    meta = lesson.get("meta") or {}
    title = (meta.get("title") if isinstance(meta, dict) else None) or "מערך שיעור"
    para(title, 20, 12, spacing=1)

    if isinstance(meta, dict):
        bits = [
            f"כיתה/גיל: {meta.get('grade')}" if meta.get("grade") else "",
            f"משך: {meta.get('duration')}" if meta.get("duration") else "",
        ]
        line = "  |  ".join(b for b in bits if b)
        if line:
            para(line, 11, 7)
        if meta.get("objective"):
            para(f"מטרת השיעור: {meta.get('objective')}", 12, 8, spacing=2)

    for key, label in _SECTIONS:
        body = _stringify(lesson.get(key))
        if not body:
            continue
        para(label, 15, 10, spacing=1)
        para(body, 12, 8, spacing=2)

    out = pdf.output()
    return bytes(out)
