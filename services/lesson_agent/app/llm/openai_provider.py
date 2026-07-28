"""
OpenAI provider — implemented and ready, but NOT active.

Selected only when ``LLM_PROVIDER=openai``; the default is Ollama, so this code
is never instantiated or called until an operator opts in. No API key is
required until then — the key is validated at call time, never at import.

Uses the OpenAI Chat Completions REST API directly via httpx (the same HTTP
client the rest of the service already depends on — no extra package), and
verifies TLS against the OS trust store for parity with the Ollama provider
(works behind TLS-inspecting corporate proxies).

Tier → model mapping:
  * fast   → ``OPENAI_FAST_MODEL``   (cheap labelling)
  * strong → ``OPENAI_STRONG_MODEL`` (content generation)
"""

from __future__ import annotations

import logging
import re
import ssl

import httpx

from app import tracing
from app.config import settings
from app.llm.base import LLMProvider

logger = logging.getLogger("lesson_agent.llm.openai")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SSL_CONTEXT = ssl.create_default_context()


class OpenAIProvider(LLMProvider):
    name = "openai"

    def _chat(
        self,
        model: str,
        system: str,
        user: str,
        *,
        temperature: float,
        max_tokens: int | None,
        trace_name: str,
    ) -> str:
        key = (settings.OPENAI_API_KEY or "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        if not model:
            raise RuntimeError("OpenAI model is not configured.")

        url = settings.OPENAI_BASE_URL.rstrip("/") + "/chat/completions"
        payload: dict = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        logger.info("OpenAI chat: model=%s", model)
        with httpx.Client(timeout=settings.OPENAI_TIMEOUT, verify=_SSL_CONTEXT) as client:
            resp = client.post(
                url, json=payload, headers={"Authorization": f"Bearer {key}"}
            )
            if resp.status_code >= 400:
                detail = resp.text
                try:
                    err = resp.json().get("error")
                    detail = (err.get("message") if isinstance(err, dict) else err) or detail
                except Exception:  # noqa: BLE001
                    pass
                raise RuntimeError(
                    f"OpenAI request failed (HTTP {resp.status_code}, model={model}): {detail}"
                )
            data = resp.json()

        choices = data.get("choices") or []
        content = (choices[0].get("message") or {}).get("content") if choices else None
        text = _THINK_RE.sub("", str(content or "")).strip()
        tracing.log_generation(trace_name, model, user, text)
        return text

    def complete_fast(self, system: str, user: str) -> str:
        return self._chat(
            settings.OPENAI_FAST_MODEL,
            system,
            user,
            temperature=settings.LOCAL_TEMPERATURE,
            max_tokens=None,
            trace_name="local_chat",
        )

    def complete_strong(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return self._chat(
            settings.OPENAI_STRONG_MODEL,
            system,
            user,
            temperature=(
                settings.CLOUD_TEMPERATURE if temperature is None else temperature
            ),
            max_tokens=max_tokens or settings.CLOUD_MAX_TOKENS,
            trace_name="cloud_chat",
        )
