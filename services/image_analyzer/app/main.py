"""Image Analyzer service — application entry point (skeleton)."""

from __future__ import annotations

from fastapi import FastAPI

from app.config import settings
from app.routers import analyze

app = FastAPI(title="Homori Image Analyzer", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "image_analyzer"}


app.include_router(analyze.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.IMAGE_ANALYZER_HOST,
        port=settings.IMAGE_ANALYZER_PORT,
        reload=settings.IMAGE_ANALYZER_DEBUG,
    )
