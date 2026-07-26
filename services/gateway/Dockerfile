# syntax=docker/dockerfile:1.6
# ─────────────────────────────────────────────────────────────────────
# Homori — FastAPI Gateway
# ─────────────────────────────────────────────────────────────────────
# The Gateway also serves the frontend, so this image is built from the
# REPOSITORY ROOT (it needs ./frontend as well as ./services/gateway):
#
#   docker build -f services/gateway/Dockerfile -t homori-gateway .
#   docker run --rm -p 8000:8000 --env-file services/gateway/.env homori-gateway
# ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY services/gateway/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Application code + the frontend it serves.
COPY services/gateway/app ./app
COPY frontend ./frontend

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
ENV GATEWAY_HOST=0.0.0.0 \
    GATEWAY_PORT=8000

CMD ["python", "-m", "app.main"]
