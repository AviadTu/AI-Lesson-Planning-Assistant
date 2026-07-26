"""
llama.cpp provider — local GGUF model via llama-cpp-python.

The model is loaded lazily on first use and cached at module level, so it is
loaded exactly once and reused across requests. A generation lock serialises
calls because a single llama.cpp context is not safe for concurrent decoding.
"""

from __future__ import annotations

import logging
import threading

from app.config import settings
from app.llm.base import LLMProvider

logger = logging.getLogger("rag.llm.llama_cpp")

_model = None
_load_lock = threading.Lock()
_gen_lock = threading.Lock()


def _get_model():
    """Return the cached Llama model, loading it once on first call."""
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                if not settings.LLM_MODEL_PATH:
                    raise RuntimeError("LLM_MODEL_PATH is not configured.")
                from llama_cpp import Llama

                logger.info("Loading GGUF model: %s", settings.LLM_MODEL_PATH)
                _model = Llama(
                    model_path=settings.LLM_MODEL_PATH,
                    n_ctx=settings.LLM_CONTEXT_SIZE,
                    n_threads=settings.LLM_THREADS,
                    n_gpu_layers=settings.LLM_N_GPU_LAYERS,
                    verbose=False,
                )
                logger.info("GGUF model loaded.")
    return _model


class LlamaCppProvider(LLMProvider):
    name = "llama_cpp"

    def generate(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> str:
        model = _get_model()
        with _gen_lock:
            result = model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
                temperature=(
                    settings.LLM_TEMPERATURE if temperature is None else temperature
                ),
            )
        return (result["choices"][0]["message"]["content"] or "").strip()


def is_model_loaded() -> bool:
    return _model is not None
