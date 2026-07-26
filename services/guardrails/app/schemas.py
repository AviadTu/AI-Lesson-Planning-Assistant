"""Pydantic models for the Guardrails service."""

from __future__ import annotations

from pydantic import BaseModel


class CheckRequest(BaseModel):
    text: str
    stage: str = "input"  # "input" (user message) | "output" (model answer)


class CheckResponse(BaseModel):
    allowed: bool = True
    # When blocked: a safe replacement message and the triggered categories.
    message: str | None = None
    categories: list[str] = []
