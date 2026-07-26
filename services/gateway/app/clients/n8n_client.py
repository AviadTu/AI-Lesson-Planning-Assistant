"""
HTTP client for triggering n8n workflows (skeleton).

Two workflows are planned (business logic lives in n8n, not here):
  * Lesson Creation Workflow
  * Export and Email Workflow (PDF generation + email delivery)

Webhook URLs are injected via environment variables and may be empty until the
workflows are imported into n8n.
"""

from __future__ import annotations

import httpx

from app.config import settings

_TIMEOUT = settings.HTTP_TIMEOUT


class WorkflowNotConfigured(RuntimeError):
    """Raised when a workflow's webhook URL has not been configured yet."""


async def trigger_lesson_creation(payload: dict) -> dict:
    """Trigger the Lesson Creation n8n workflow."""
    url = settings.N8N_LESSON_CREATION_WEBHOOK_URL
    if not url:
        raise WorkflowNotConfigured("N8N_LESSON_CREATION_WEBHOOK_URL is not set.")
    return await _post(url, payload)


async def trigger_export(payload: dict) -> dict:
    """Trigger the Export and Email (PDF + email) n8n workflow."""
    url = settings.N8N_EXPORT_WEBHOOK_URL
    if not url:
        raise WorkflowNotConfigured("N8N_EXPORT_WEBHOOK_URL is not set.")
    return await _post(url, payload)


async def _post(url: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            return {}
