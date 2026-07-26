"""
/status route.

Proxies the RAG service readiness/ingestion status.  The response shape is
preserved from the previous project (``ready`` / ``ingesting`` /
``ingestion_failed``) so the frontend's polling logic is unchanged.  Chat is
never blocked on ingestion.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter

from app.clients import rag_client

router = APIRouter()


@router.get("/status")
async def status():
    try:
        return await rag_client.status()
    except httpx.HTTPError:
        # Non-fatal: report a safe default so the UI keeps chat enabled.
        return {"ready": True, "status": "ready", "ingestion": None}
