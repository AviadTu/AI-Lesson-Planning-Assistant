"""
Static page routes.

The Gateway serves the existing frontend on the same origin as the API so the
UI's relative ``fetch()`` paths keep working unchanged.  Templates are rendered
with Jinja2 to preserve the original ``{% raw %}`` block in ``index.html``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import settings

router = APIRouter()


def _resolve_templates_dir() -> Path:
    """
    Find the frontend templates directory, checking (in order):
      1. FRONTEND_TEMPLATES_DIR env override,
      2. <cwd>/frontend/templates       (container: WORKDIR /app),
      3. <repo-root>/frontend/templates (local: uvicorn from repo root).
    """
    candidates: list[Path] = []
    if settings.FRONTEND_TEMPLATES_DIR:
        candidates.append(Path(settings.FRONTEND_TEMPLATES_DIR))
    candidates.append(Path.cwd() / "frontend" / "templates")
    candidates.append(
        Path(__file__).resolve().parents[4] / "frontend" / "templates"
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    # Fall back to the repo-root guess so the error message is actionable.
    return candidates[-1]


templates = Jinja2Templates(directory=str(_resolve_templates_dir()))


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@router.get("/logout", response_class=HTMLResponse)
async def logout(request: Request):
    return templates.TemplateResponse(request, "logout.html")
