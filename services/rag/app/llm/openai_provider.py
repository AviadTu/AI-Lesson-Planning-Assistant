"""
OpenAI provider — implemented and ready, but NOT active.

Selected only when ``LLM_PROVIDER=openai``; the default is ``ollama_cloud``, so
this code is never instantiated or called until an operator opts in. No API key
is required until then — the key is validated at call time, never at import.

Uses the OpenAI Chat Completions REST API directly via httpx (the client this
service already depends on — no extra package), verifying TLS against the OS
trust store for parity with the Ollama Cloud provider. Retrieval/ingestion and
embeddings are untouched; only generation flows through here.
"""

from __future__ import annotations

import logging
import ssl

import httpx

from app.config import settings
from app.llm.base import LLMProvider

logger = logging.getLogger("rag.llm.openai")

_CHAT_TIMEOUT = 300.0
_SSL_CONTEXT = ssl.create_default_context()


class OpenAIProvider(LLMProvider):
    name = "openai"

    def generate(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> str:
        model = (settings.OPENAI_MODEL or "").strip()
        if not model:
            raise RuntimeError("OPENAI_MODEL is not configured.")

        api_key = (settings.OPENAI_API_KEY or "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        url = settings.OPENAI_BASE_URL.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": (
                settings.LLM_TEMPERATURE if temperature is None else temperature
            ),
        }
        # Send max_tokens only when set. GPT-5 reasoning models reject
        # 'max_tokens' (they use 'max_completion_tokens'); leaving it out lets
        # them run, while classic models still honour LLM_MAX_TOKENS.
        _mt = max_tokens or settings.LLM_MAX_TOKENS
        if _mt:
            payload["max_tokens"] = _mt

        logger.info("OpenAI chat: model=%s", model)
        with httpx.Client(timeout=_CHAT_TIMEOUT, verify=_SSL_CONTEXT) as client:
            resp = client.post(url, json=payload, headers=headers)
            if resp.status_code >= 400:
                detail = resp.text
                try:
                    err = resp.json().get("error")
                    detail = (err.get("message") if isinstance(err, dict) else err) or detail
                except Exception:  # noqa: BLE001
                    pass
                raise RuntimeError(
                    f"OpenAI request failed "
                    f"(HTTP {resp.status_code}, model={model}): {detail}"
                )
            data = resp.json()

        choices = data.get("choices") or []
        content = (choices[0].get("message") or {}).get("content") if choices else None
        if content is None:
            raise RuntimeError(f"OpenAI returned no message content (model={model}).")
        return str(content).strip()
