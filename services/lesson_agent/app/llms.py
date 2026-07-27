"""
Model access for the lesson-agent.

Two clearly-separated responsibilities (documented in config.py):

  * LOCAL model  — via LangChain ``ChatOllama`` against the local Ollama daemon.
                   Used for classification and missing-information detection.
  * CLOUD model  — via a direct Ollama Cloud HTTPS call (Bearer auth) that
                   verifies TLS against the OS trust store, reusing the proven
                   approach from the RAG service so it works behind TLS-
                   inspecting proxies. Used for ideas, lesson generation and
                   revisions.

Both expose a JSON helper that robustly extracts a JSON object from the model
output (stripping markdown fences and any ``<think>`` reasoning blocks).
"""

from __future__ import annotations

import json
import logging
import re
import ssl
import threading

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.config import settings

logger = logging.getLogger("lesson_agent.llms")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Verify cloud TLS via the OS trust store (see RAG ollama_cloud_provider).
_SSL_CONTEXT = ssl.create_default_context()

_local_llm: ChatOllama | None = None
_local_lock = threading.Lock()


def get_local_llm() -> ChatOllama:
    """Return a cached local ChatOllama instance."""
    global _local_llm
    if _local_llm is None:
        with _local_lock:
            if _local_llm is None:
                _local_llm = ChatOllama(
                    model=settings.LOCAL_MODEL,
                    base_url=settings.LOCAL_OLLAMA_BASE_URL,
                    temperature=settings.LOCAL_TEMPERATURE,
                    client_kwargs={"timeout": settings.LOCAL_TIMEOUT},
                )
    return _local_llm


def local_chat(system: str, user: str) -> str:
    """One local-model completion. Returns the assistant text."""
    llm = get_local_llm()
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return _THINK_RE.sub("", str(resp.content or "")).strip()


def cloud_chat(
    system: str,
    user: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """One cloud-model completion via the direct Ollama Cloud API (Bearer auth)."""
    key = (settings.OLLAMA_API_KEY or "").strip()
    if not key:
        raise RuntimeError("OLLAMA_API_KEY is not configured for cloud generation.")
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
        resp = client.post(url, json=payload, headers={"Authorization": f"Bearer {key}"})
        if resp.status_code >= 400:
            detail = resp.text
            try:
                detail = resp.json().get("error") or detail
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(f"Ollama Cloud failed (HTTP {resp.status_code}): {detail}")
        data = resp.json()
    content = (data.get("message") or {}).get("content") or ""
    return _THINK_RE.sub("", str(content)).strip()


def extract_json(text: str) -> dict:
    """
    Robustly parse a JSON object from model output: strip code fences, then take
    the outermost ``{ ... }`` span. Raises ValueError if nothing parses.
    """
    cleaned = _THINK_RE.sub("", text or "").strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise ValueError("No JSON object found in model output.")


def local_json(system: str, user: str) -> dict:
    return extract_json(local_chat(system, user))


def cloud_json(system: str, user: str, *, temperature: float | None = None) -> dict:
    return extract_json(cloud_chat(system, user, temperature=temperature))
