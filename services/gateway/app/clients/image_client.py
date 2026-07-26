"""HTTP client for the Image Analyzer service (skeleton)."""

from __future__ import annotations

import httpx

from app.config import settings

_BASE = settings.IMAGE_ANALYZER_URL.rstrip("/")
_TIMEOUT = settings.HTTP_TIMEOUT


async def analyze(filename: str, content: bytes, content_type: str) -> dict:
    """Send an image to the Image Analyzer service for analysis."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{_BASE}/analyze",
            files={"image": (filename, content, content_type)},
        )
        resp.raise_for_status()
        return resp.json()
