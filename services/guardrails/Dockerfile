# syntax=docker/dockerfile:1.6
# ─────────────────────────────────────────────────────────────────────
# Homori — Guardrails Service (skeleton)
# ─────────────────────────────────────────────────────────────────────
# Build from the repository root:
#   docker build -f services/guardrails/Dockerfile -t homori-guardrails .
#   docker run --rm -p 8003:8003 --env-file services/guardrails/.env \
#     homori-guardrails
# ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY services/guardrails/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/guardrails/app ./app

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8003
ENV GUARDRAILS_HOST=0.0.0.0 \
    GUARDRAILS_PORT=8003

CMD ["python", "-m", "app.main"]
