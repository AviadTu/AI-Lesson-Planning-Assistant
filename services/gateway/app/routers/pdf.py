"""
PDF routes: render a lesson plan to a Hebrew/RTL PDF and serve the download.

  POST /internal/pdf     — n8n calls this with a lesson dict; returns a
                           download URL the frontend can link to.
  GET  /pdf/download     — serves a generated PDF (forces download, Hebrew
                           filename preserved via RFC 5987).

Generated PDFs are stored locally under ``settings.PDF_DIR`` (runtime data,
git-ignored). The lesson content itself is produced by the lesson-agent; the
Gateway only renders + serves it, consistent with its download responsibilities.
"""

from __future__ import annotations

import logging
import os
import uuid
from urllib.parse import quote

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from app.config import settings
from app.pdf_render import PdfRenderError, render_lesson_pdf
from app.schemas import PdfRequest

router = APIRouter()
logger = logging.getLogger("gateway.pdf")


def _pdf_dir() -> str:
    os.makedirs(settings.PDF_DIR, exist_ok=True)
    return settings.PDF_DIR


def _safe_name(title: str) -> str:
    base = (title or "מערך שיעור").strip() or "מערך שיעור"
    base = "".join(ch for ch in base if ch not in '\\/:*?"<>|').strip()
    return (base[:80] or "מערך שיעור") + ".pdf"


@router.post("/internal/pdf")
def create_pdf(payload: PdfRequest):
    """Render a lesson to PDF, store it, and return a download URL."""
    try:
        data = render_lesson_pdf(payload.lesson)
    except PdfRenderError as exc:
        logger.warning("PDF render failed: %s", exc)
        return JSONResponse({"error": f"PDF generation failed: {exc}"}, status_code=502)
    except Exception as exc:  # noqa: BLE001
        logger.exception("PDF render error")
        return JSONResponse({"error": "PDF generation failed."}, status_code=500)

    pdf_id = uuid.uuid4().hex
    meta = payload.lesson.get("meta") or {}
    filename = _safe_name(meta.get("title") if isinstance(meta, dict) else None)

    d = _pdf_dir()
    with open(os.path.join(d, f"{pdf_id}.pdf"), "wb") as fh:
        fh.write(data)
    with open(os.path.join(d, f"{pdf_id}.name"), "w", encoding="utf-8") as fh:
        fh.write(filename)

    return {
        "ok": True,
        "pdf_id": pdf_id,
        "filename": filename,
        # Ends in .pdf so the frontend recognises it as a download link.
        "download_url": f"/pdf/download/{pdf_id}.pdf",
        "bytes": len(data),
    }


@router.get("/pdf/download/{pdf_id}.pdf")
def download_pdf(pdf_id: str):
    """Serve a previously generated PDF (forces download)."""
    pdf_id = "".join(ch for ch in pdf_id if ch.isalnum())  # id is a uuid4 hex
    path = os.path.join(_pdf_dir(), f"{pdf_id}.pdf")
    if not os.path.exists(path):
        return JSONResponse({"error": "PDF not found."}, status_code=404)

    display = f"{pdf_id}.pdf"
    name_path = os.path.join(_pdf_dir(), f"{pdf_id}.name")
    if os.path.exists(name_path):
        with open(name_path, encoding="utf-8") as fh:
            display = fh.read().strip() or display

    ascii_fallback = display.encode("ascii", "ignore").decode("ascii") or "lesson.pdf"
    disposition = (
        f'attachment; filename="{ascii_fallback}"; '
        f"filename*=UTF-8''{quote(display, safe='')}"
    )
    return FileResponse(
        path, media_type="application/pdf",
        headers={"Content-Disposition": disposition},
    )
