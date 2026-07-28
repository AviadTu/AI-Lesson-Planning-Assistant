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

    # ── LLM provider selection ───────────────────────────────────────
    # Generation is kept behind a provider abstraction (see app/llm/). The
    # LangGraph flow never talks to a concrete backend.
    #   LLM_PROVIDER = "ollama"  -> Ollama daemon (fast) + Ollama Cloud (strong)
    #   LLM_PROVIDER = "openai"  -> OpenAI Chat Completions (implemented; opt-in)
    # Default is Ollama; "ollama_cloud" is accepted as an alias (the repo-root
    # .env currently sets LLM_PROVIDER=ollama_cloud).
    LLM_PROVIDER: str = "ollama"

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

    # ── OpenAI provider (implemented but NOT active) ─────────────────
    # Used only when LLM_PROVIDER=openai. No key is required until then; the
    # key is validated at call time, never at import. fast→cheap model,
    # strong→stronger model (they reuse the shared tier temperatures above).
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_FAST_MODEL: str = "gpt-4o-mini"
    OPENAI_STRONG_MODEL: str = "gpt-4o"
    OPENAI_TIMEOUT: float = 300.0

    # ── Observability: Langfuse (optional, feature-flagged) ──────────
    # Disabled by default and a strict no-op when keys are absent or the SDK
    # is not installed — tracing must never block or break a request.
    LANGFUSE_ENABLED: bool = False
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"


settings = Settings()

