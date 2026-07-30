"""Request/response models for the lesson-agent HTTP API (called by n8n)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A normalized source snippet collected + processed by n8n."""

    type: str = "text"                 # text | pdf | docx | image | url | youtube | web | rag
    title: str | None = None
    text: str = ""
    url: str | None = None
    citation: str | None = None


class TurnRequest(BaseModel):
    """One conversational turn, forwarded by n8n from the Gateway."""

    session_id: str
    message: str = ""
    # Sources already fetched/extracted by n8n's Source Processing workflow.
    sources: list[Source] = Field(default_factory=list)
    # Per-source failures/notes from n8n (e.g. "youtube unavailable"), surfaced
    # in the grouped source review so provenance/omissions stay visible.
    warnings: list[str] = Field(default_factory=list)
    failed_sources: list[dict] = Field(default_factory=list)
    # Optional explicit control action (e.g. "reset"). Content-only; PDF/email
    # are orchestrated by n8n, not by the agent.
    action: str | None = None


class Idea(BaseModel):
    title: str
    description: str
    direction: str


class TurnResponse(BaseModel):
    session_id: str
    stage: str                          # intake | ideas | drafting | final | done
    reply: str                          # assistant text to display to the user
    ideas: list[Idea] = Field(default_factory=list)
    lesson: dict | None = None          # structured final lesson (template sections)
    needs_input: bool = True            # True while waiting for the user
    meta: dict = Field(default_factory=dict)
