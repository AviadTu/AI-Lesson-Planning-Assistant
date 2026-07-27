"""
lesson-agent configuration.

The lesson-agent is a focused conversational service (LangGraph + LangChain).
It reuses the project's existing model configuration from the repo-root ``.env``
(the same file the RAG service reads) so model choices stay consistent:

  * LOCAL model  (Ollama, localhost) — intent/conversational classification and
    missing-information detection. Cheap, fast, fully local.
  * CLOUD model  (Ollama Cloud, HTTPS + key) — lesson ideas, full lesson
    generation and revisions. Stronger model for educational writing.

Generation never touches the RAG service; the RAG service stays independent.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Read the shared repo-root .env first (operator config: OLLAMA_API_KEY, ...),
# then a service-local .env override if present. OS env wins over both.
_ROOT_ENV = Path(__file__).resolve().parents[3] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(_ROOT_ENV), ".env"), extra="ignore"
    )

    # ── HTTP server ──────────────────────────────────────────────────
    LESSON_AGENT_HOST: str = "0.0.0.0"
    LESSON_AGENT_PORT: int = 8004
    LESSON_AGENT_DEBUG: bool = False

    # ── LangGraph conversation state (checkpointer) ──────────────────
    # Persists per-session conversational state/memory across turns.
    CHECKPOINT_DB_PATH: str = "./data/lesson_agent_checkpoints.sqlite"

    # ── LOCAL model (classification / missing-info detection) ─────────
    LOCAL_OLLAMA_BASE_URL: str = "http://localhost:11434"
    LOCAL_MODEL: str = "llama3.1:latest"
    LOCAL_TEMPERATURE: float = 0.0
    LOCAL_TIMEOUT: float = 120.0

    # ── CLOUD model (ideas / lesson generation / revisions) ──────────
    # Reuses the same Ollama Cloud endpoint + key as the RAG service.
    OLLAMA_CLOUD_BASE_URL: str = "https://ollama.com"
    OLLAMA_API_KEY: str = ""
    OLLAMA_CLOUD_MODEL: str = "gemma4:31b"
    CLOUD_TEMPERATURE: float = 0.4
    CLOUD_MAX_TOKENS: int = 1500
    CLOUD_TIMEOUT: float = 300.0


settings = Settings()
