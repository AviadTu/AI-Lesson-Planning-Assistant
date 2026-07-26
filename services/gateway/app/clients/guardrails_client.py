"""HTTP client for the Guardrails service (skeleton)."""

from __future__ import annotations

import httpx

from app.config import settings

_BASE = settings.GUARDRAILS_URL.rstrip("/")
_TIMEOUT = settings.HTTP_TIMEOUT


async def check(text: str, stage: str = "input") -> dict:
    """
    Run a guardrails check on a piece of text.

    ``stage`` is "input" (user message) or "output" (model answer).
    Returns the Guardrails service verdict.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{_BASE}/check", json={"text": text, "stage": stage}
        )
        resp.raise_for_status()
        return resp.json()
