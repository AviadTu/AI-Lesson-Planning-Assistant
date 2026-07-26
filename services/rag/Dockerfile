# syntax=docker/dockerfile:1.6
# ─────────────────────────────────────────────────────────────────────
# Homori — RAG Service (LangChain + ChromaDB + llama.cpp)
# ─────────────────────────────────────────────────────────────────────
# Build from the repository root:
#   docker build -f services/rag/Dockerfile -t homori-rag .
#
# Run (mount the GGUF model and a data volume):
#   docker run --rm -p 8001:8001 \
#     --env-file services/rag/.env \
#     -v /path/to/models:/models:ro \
#     -v homori-rag-data:/app/data \
#     homori-rag
# ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build toolchain required to compile llama-cpp-python.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/rag/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/rag/app ./app

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001
ENV RAG_HOST=0.0.0.0 \
    RAG_PORT=8001

CMD ["python", "-m", "app.main"]
