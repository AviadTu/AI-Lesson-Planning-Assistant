"""
RAG service configuration.

The RAG service owns the uploaded source files, the document metadata registry,
ChromaDB, the ingestion + retrieval pipeline (Ollama embeddings) and answer
generation via a local GGUF model (llama.cpp), behind the LLM provider
abstraction.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The monorepo keeps a single shared .env at the repository root (where the
# operator configures LLM_MODEL_PATH etc.). We read it first, then a
# service-local .env (if present) which overrides it; real OS environment
# variables still take precedence over both.
_ROOT_ENV = Path(__file__).resolve().parents[3] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(_ROOT_ENV), ".env"), extra="ignore"
    )

    # ── HTTP server ──────────────────────────────────────────────────
    RAG_HOST: str = "0.0.0.0"
    RAG_PORT: int = 8001
    RAG_DEBUG: bool = False

    # ── Local storage ────────────────────────────────────────────────
    UPLOAD_DIR: str = "./data/uploads"          # source documents
    REGISTRY_DB_PATH: str = "./data/registry.db"  # document metadata (SQLite)

    # ── Uploads ──────────────────────────────────────────────────────
    MAX_UPLOAD_MB: int = 20
    ALLOWED_UPLOAD_EXTENSIONS: str = "txt,pdf,docx"

    # ── ChromaDB (vector store) ──────────────────────────────────────
    CHROMA_DIR: str = "./data/chroma"
    CHROMA_COLLECTION: str = "lesson_plans"

    # ── Chunking ─────────────────────────────────────────────────────
    # Character-based, sensible defaults for Hebrew lesson-plan documents.
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # ── Embeddings (local Ollama daemon, HTTP API) ───────────────────
    # Embeddings ALWAYS run against the local Ollama daemon (never cloud),
    # using a dedicated embedding model (NOT the generation LLM).
    OLLAMA_EMBEDDINGS_BASE_URL: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    EMBEDDING_TIMEOUT: float = 120.0

    # ── LLM provider selection ───────────────────────────────────────
    # Generation is kept behind a provider abstraction (see app/llm/).
    # Retrieval and ingestion never talk to the LLM directly.
    #   LLM_PROVIDER = "llama_cpp"     -> local GGUF via llama-cpp-python
    #   LLM_PROVIDER = "ollama_cloud"  -> direct Ollama Cloud API (HTTPS + key)
    #   LLM_PROVIDER = "openai"        -> OpenAI Chat Completions (implemented;
    #                                     opt-in only, no key required until set)
    LLM_PROVIDER: str = "llama_cpp"

    # ── Ollama Cloud generation (direct API, NOT the local daemon) ────
    # Generation connects straight to the Ollama Cloud endpoint over HTTPS
    # with a Bearer API key. It must never be routed through localhost.
    OLLAMA_CLOUD_BASE_URL: str = "https://ollama.com"
    OLLAMA_API_KEY: str = ""                    # required when LLM_PROVIDER=ollama_cloud
    OLLAMA_CLOUD_MODEL: str = "gemma4:31b"      # cloud model name (no ":cloud" suffix)

    # ── OpenAI generation (direct API, NOT active) ───────────────────
    # Used only when LLM_PROVIDER=openai. No key required until then; the key
    # is validated at call time, never at import. Embeddings stay local Ollama.
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-5"

    # ── llama.cpp / local GGUF model ─────────────────────────────────
    LLM_MODEL_PATH: str = ""                  # absolute path to the .gguf file
    LLM_CONTEXT_SIZE: int = 4096              # n_ctx
    LLM_MAX_TOKENS: int = 700                 # max tokens generated per answer
    LLM_TEMPERATURE: float = 0.2
    LLM_THREADS: int = 8                      # CPU threads
    LLM_N_GPU_LAYERS: int = 0                 # 0 = CPU only; raise to offload

    # ── Retrieval ────────────────────────────────────────────────────
    # Default number of chunks returned by /retrieve (top_k).
    RETRIEVAL_K: int = 3
    # Minimum cosine similarity (1 - cosine distance) a chunk must reach to be
    # returned. Chunks below this are dropped. Range roughly [0, 1].
    SIMILARITY_THRESHOLD: float = 0.3

    @property
    def allowed_extensions(self) -> set[str]:
        return {
            ext.strip().lower().lstrip(".")
            for ext in self.ALLOWED_UPLOAD_EXTENSIONS.split(",")
            if ext.strip()
        }


settings = Settings()
