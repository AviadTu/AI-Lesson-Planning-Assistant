"""
Ollama provider — the lesson-agent's current (default) backend.

  * fast   tier — LangChain ``ChatOllama`` against the LOCAL Ollama daemon.
  * strong tier — a direct Ollama Cloud HTTPS call (Bearer auth) that verifies
                  TLS against the OS trust store, so it works behind TLS-
                  inspecting corporate proxies.

This is the code that previously lived in app/llms.py, moved here unchanged so
behavior is identical; only its location changed.
"""

from __future__ import annotations

import logging
import re
import ssl
import threading

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app import tracing
from app.config import settings
from app.llm.base import LLMProvider

logger = logging.getLogger("lesson_agent.llm.ollama")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Verify cloud TLS via the OS trust store (see RAG ollama_cloud_provider).
_SSL_CONTEXT = ssl.create_default_context()


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self) -> None:
        self._local_llm: ChatOllama | None = None
        self._local_lock = threading.Lock()

    # ── fast tier: local Ollama daemon via ChatOllama ────────────────
    def _get_local_llm(self) -> ChatOllama:
        if self._local_llm is None:
            with self._local_lock:
                if self._local_llm is None:
                    self._local_llm = ChatOllama(
                        model=settings.LOCAL_MODEL,
                        base_url=settings.LOCAL_OLLAMA_BASE_URL,
                        temperature=settings.LOCAL_TEMPERATURE,
                        client_kwargs={"timeout": settings.LOCAL_TIMEOUT},
                    )
        return self._local_llm

    def complete_fast(self, system: str, user: str) -> str:
        llm = self._get_local_llm()
        resp = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        text = _THINK_RE.sub("", str(resp.content or "")).strip()
        tracing.log_generation("local_chat", settings.LOCAL_MODEL, user, text)
        return text

    # ── strong tier: direct Ollama Cloud API (Bearer auth) ───────────
    def complete_strong(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        key = (settings.OLLAMA_API_KEY or "").strip()
        if not key:
            raise RuntimeError(
                "OLLAMA_API_KEY is not configured for cloud generation."
            )
        url = settings.OLLAMA_CLOUD_BASE_URL.rstrip("/") + "/api/chat"
        payload = {
            "model": settings.OLLAMA_CLOUD_MODEL,
            "stream": False,
            "options": {
                "temperature": (
                    settings.CLOUD_TEMPERATURE if temperature is None else temperature
                ),
                "num_predict": max_tokens or settings.CLOUD_MAX_TOKENS,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        with httpx.Client(timeout=settings.CLOUD_TIMEOUT, verify=_SSL_CONTEXT) as client:
            resp = client.post(
                url, json=payload, headers={"Authorization": f"Bearer {key}"}
            )
            if resp.status_code >= 400:
                detail = resp.text
                try:
                    detail = resp.json().get("error") or detail
                except Exception:  # noqa: BLE001
                    pass
                raise RuntimeError(
                    f"Ollama Cloud failed (HTTP {resp.status_code}): {detail}"
                )
            data = resp.json()
        content = (data.get("message") or {}).get("content") or ""
        text = _THINK_RE.sub("", str(content)).strip()
        tracing.log_generation("cloud_chat", settings.OLLAMA_CLOUD_MODEL, user, text)
        return text
