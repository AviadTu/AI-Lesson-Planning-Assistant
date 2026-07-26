"""Pydantic models for the Image Analyzer service."""

from __future__ import annotations

from pydantic import BaseModel


class AnalyzeResponse(BaseModel):
    description: str = ""
    labels: list[str] = []
    metadata: dict = {}
