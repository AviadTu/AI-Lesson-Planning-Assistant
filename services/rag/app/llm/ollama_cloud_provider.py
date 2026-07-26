"""
Ollama Cloud provider — generation via the direct Ollama Cloud HTTP API.

Requests go straight to the cloud endpoint (``OLLAMA_CLOUD_BASE_URL``, default
``https://ollama.com``) over HTTPS, authenticated with a Bearer API key
(``OLLAMA_API_KEY``). Generation is NEVER routed through the local Ollama
daemon; only embeddings use the local daemon (see app/embeddings/).
"""

from __future__ import annotations

import logging
import ssl

import httpx

from app.config import settings
from app.llm.base import LLMProvider

logger = logging.getLogger("rag.llm.ollama_cloud")

# Cloud round-trips can be slower than local inference.
_CHAT_TIMEOUT = 300.0

# Verify TLS against the OS trust store. httpx's default (certifi) does not
# include corporate/proxy root CAs, so a TLS-inspecting proxy breaks the direct
# HTTPS call; ssl.create_default_context() loads the system store instead.
# Verification stays ON — this is not an insecure bypass.
_SSL_CONTEXT = ssl.create_default_context()


class OllamaCloudProvider(LLMProvider):
    name = "ollama_cloud"

    def generate(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> str:
        model = (settings.OLLAMA_CLOUD_MODEL or "").strip()
        if not model:
            raise RuntimeError("OLLAMA_CLOUD_MODEL is not configured.")

        api_key = (settings.OLLAMA_API_KEY or "").strip()
        if not api_key:
            raise RuntimeError("OLLAMA_API_KEY is not configured.")

        url = settings.OLLAMA_CLOUD_BASE_URL.rstrip("/") + "/api/chat"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": (
                    settings.LLM_TEMPERATURE if temperature is None else temperature
                ),
                "num_predict": max_tokens or settings.LLM_MAX_TOKENS,
            },
        }

        logger.info("Ollama Cloud chat: model=%s", model)
        with httpx.Client(timeout=_CHAT_TIMEOUT, verify=_SSL_CONTEXT) as client:
            resp = client.post(url, json=payload, headers=headers)
            if resp.status_code >= 400:
                detail = resp.text
                try:
                    detail = resp.json().get("error") or detail
                except Exception:  # noqa: BLE001
                    pass
                raise RuntimeError(
                    f"Ollama Cloud request failed "
                    f"(HTTP {resp.status_code}, model={model}): {detail}"
                )
            data = resp.json()

        message = data.get("message") or {}
        content = message.get("content")
        if content is None:
            raise RuntimeError(
                f"Ollama Cloud returned no message content (model={model})."
            )
        return str(content).strip()
