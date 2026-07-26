# syntax=docker/dockerfile:1.6
# ─────────────────────────────────────────────────────────────────────
# Homori — Image Analyzer Service (skeleton)
# ─────────────────────────────────────────────────────────────────────
# Build from the repository root:
#   docker build -f services/image_analyzer/Dockerfile -t homori-image-analyzer .
#   docker run --rm -p 8002:8002 --env-file services/image_analyzer/.env \
#     homori-image-analyzer
# ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY services/image_analyzer/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/image_analyzer/app ./app

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8002
ENV IMAGE_ANALYZER_HOST=0.0.0.0 \
    IMAGE_ANALYZER_PORT=8002

CMD ["python", "-m", "app.main"]
