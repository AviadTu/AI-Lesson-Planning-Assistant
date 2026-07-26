"""
/check route (skeleton).

Replaces Bedrock Guardrails.  Will enforce content-safety policies on user
input and/or model output.  Business logic is not implemented yet, so this
returns 501.  The Bedrock Guardrails policy definitions (AWS-side) must be
recovered and re-encoded here.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import CheckRequest, CheckResponse

router = APIRouter()


@router.post("/check", response_model=CheckResponse)
async def check(payload: CheckRequest):
    # TODO: evaluate payload.text against the content-safety policy for the
    # given stage and return allowed/blocked + a safe replacement message.
    raise HTTPException(status_code=501, detail="Guardrails not implemented yet.")
