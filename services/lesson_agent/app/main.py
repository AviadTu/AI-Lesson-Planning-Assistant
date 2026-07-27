"""
lesson-agent — application entry point.

A focused conversational service: it turns one user message (plus any sources
n8n already collected) into the next assistant turn of the lesson-planning
conversation, using a real LangGraph state machine. It owns ONLY educational
reasoning + conversation memory; orchestration (sources, PDF, email) stays in
n8n, and retrieval stays in the independent RAG service.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app.config import settings
from app.graph import run_turn
from app.schemas import TurnRequest, TurnResponse

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Homori Lesson Agent", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "lesson_agent"}


@app.post("/turn", response_model=TurnResponse)
def turn(payload: TurnRequest) -> TurnResponse:
    """Advance the lesson-planning conversation by one turn."""
    state = run_turn(
        payload.session_id,
        payload.message,
        [s.model_dump() for s in payload.sources],
    )
    return TurnResponse(
        session_id=payload.session_id,
        stage=state.get("stage", "intake"),
        reply=state.get("reply", ""),
        ideas=state.get("ideas", []) or [],
        lesson=state.get("lesson"),
        needs_input=state.get("needs_input", True),
        meta={"intent": state.get("intent"), "stage": state.get("stage")},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.LESSON_AGENT_HOST,
        port=settings.LESSON_AGENT_PORT,
        reload=settings.LESSON_AGENT_DEBUG,
    )
