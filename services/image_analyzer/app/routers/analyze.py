"""
/analyze route (skeleton).

Accepts an image (multipart/form-data) and will return a structured analysis.
Business logic is not implemented yet, so this returns 501.  This is a net-new
capability with no equivalent in the previous project.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import AnalyzeResponse

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(image: UploadFile = File(...)):
    # TODO: validate extension/size, run the image model, return AnalyzeResponse.
    raise HTTPException(status_code=501, detail="Image analysis not implemented yet.")
