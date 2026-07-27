# syntax=docker/dockerfile:1.6
# ─────────────────────────────────────────────────────────────────────
# Homori — lesson-agent (LangGraph conversational lesson planning)
# ─────────────────────────────────────────────────────────────────────
# Build from the repository root:
#   docker build -f services/lesson_agent/Dockerfile -t homori-lesson-agent .
#   docker run --rm -p 8004:8004 --env-file services/lesson_agent/.env \
#     -v homori-lesson-agent-data:/app/data homori-lesson-agent
# ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY services/lesson_agent/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/lesson_agent/app ./app

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8004
ENV LESSON_AGENT_HOST=0.0.0.0 \
    LESSON_AGENT_PORT=8004

CMD ["python", "-m", "app.main"]
