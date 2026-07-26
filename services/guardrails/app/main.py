"""Guardrails service — application entry point (skeleton)."""

from __future__ import annotations

from fastapi import FastAPI

from app.config import settings
from app.routers import check

app = FastAPI(title="Homori Guardrails", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "guardrails"}


app.include_router(check.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.GUARDRAILS_HOST,
        port=settings.GUARDRAILS_PORT,
        reload=settings.GUARDRAILS_DEBUG,
    )
