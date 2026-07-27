"""
Gateway configuration.

All values come from environment variables (optionally loaded from a local
``.env`` file).  Service URLs are injected so the Gateway can reach the other
FastAPI services over plain HTTP/REST.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── HTTP server ──────────────────────────────────────────────────
    GATEWAY_HOST: str = "0.0.0.0"
    GATEWAY_PORT: int = 8000
    GATEWAY_DEBUG: bool = False

    # ── Persistence (Gateway owns SQLite: sessions + chat history) ────
    SQLITE_PATH: str = "./data/gateway.db"

    # ── Frontend ─────────────────────────────────────────────────────
    # Directory containing index.html / logout.html.  Empty = auto-detect
    # (works when running from the repo root or inside the container).
    FRONTEND_TEMPLATES_DIR: str = ""

    # ── Downstream services (URLs injected via env) ──────────────────
    RAG_SERVICE_URL: str = "http://localhost:8001"
    IMAGE_ANALYZER_URL: str = "http://localhost:8002"
    GUARDRAILS_URL: str = "http://localhost:8003"

    # ── n8n orchestration webhooks (populated once workflows exist) ──
    N8N_LESSON_CREATION_WEBHOOK_URL: str = ""
    N8N_EXPORT_WEBHOOK_URL: str = ""

    # ── Intent routing ───────────────────────────────────────────────
    # A lightweight LOCAL classifier decides ambiguous "knowledge question vs
    # lesson-plan request" cases; explicit requests are caught by deterministic
    # rules first (see app/routing.py). Local so it is cheap and offline.
    LOCAL_OLLAMA_BASE_URL: str = "http://localhost:11434"
    INTENT_MODEL: str = "llama3.2:latest"

    # ── Conversation-memory tuning (see app/memory.py) ───────────────
    HISTORY_MAX_MESSAGES: int = 6
    HISTORY_MAX_CHARS: int = 500

    # ── Inter-service HTTP timeouts (seconds) ────────────────────────
    # Generous: local GGUF generation on CPU can take a while per answer.
    HTTP_TIMEOUT: float = 600.0

    # ── Upload limit (mirrors the RAG service; rejects early) ────────
    MAX_UPLOAD_MB: int = 20


settings = Settings()
