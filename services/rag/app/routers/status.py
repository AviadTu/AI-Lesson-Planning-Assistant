"""
/status route.

Reports ingestion status in the shape the frontend expects
(``ready`` / ``ingesting`` / ``ingestion_failed``). ``ready`` is only True once
no ingestion is in progress and the last job did not fail — a document is never
reported ready before its ingestion completes.
"""

from __future__ import annotations

from fastapi import APIRouter

from app import status as status_mgr
from app.schemas import StatusResponse

router = APIRouter()


@router.get("/status", response_model=StatusResponse)
async def status():
    return StatusResponse(**status_mgr.snapshot())
