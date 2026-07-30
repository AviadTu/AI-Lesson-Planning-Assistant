"""
Model-access facade for the lesson-agent (provider-agnostic).

The LangGraph nodes (app/graph.py) call these helpers and never see which
backend is configured. All provider-specific code lives behind the
``LLMProvider`` interface (app/llm/); the active provider is chosen once from
``settings.LLM_PROVIDER`` (default Ollama — see app/llm/provider.py).

Two tiers, matching the agent's needs:
  * fast   (``local_chat`` / ``local_json``)  — labelling: intent, missing-info,
    selection, assist.
  * strong (``cloud_chat`` / ``cloud_json``)  — content: ideas, full lesson,
    revisions.

The ``local_*`` / ``cloud_*`` names are kept for a stable call surface; they are
tier labels, not backend names. Each JSON helper robustly extracts a JSON object
from the model output (stripping markdown fences and any ``<think>`` blocks).
"""

from __future__ import annotations

import json
import re

from app.llm.provider import get_llm_provider

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def local_chat(system: str, user: str) -> str:
    """One FAST-tier completion. Returns the assistant text."""
    return get_llm_provider().complete_fast(system, user)


def cloud_chat(
    system: str,
    user: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """One STRONG-tier completion. Returns the assistant text."""
    return get_llm_provider().complete_strong(
        system, user, temperature=temperature, max_tokens=max_tokens
    )


def extract_json(text: str) -> dict:
    """
    Robustly parse a JSON object from model output: strip code fences, then parse
    the first complete ``{ ... }`` object. Uses ``raw_decode`` so trailing prose
    the model may append after the JSON (which yields "Extra data") is ignored.
    Raises ValueError if nothing parses.
    """
    cleaned = _THINK_RE.sub("", text or "").strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
            return obj
        except json.JSONDecodeError:
            pass
    raise ValueError("No JSON object found in model output.")


def local_json(system: str, user: str) -> dict:
    return extract_json(local_chat(system, user))


def cloud_json(system: str, user: str, *, temperature: float | None = None) -> dict:
    return extract_json(cloud_chat(system, user, temperature=temperature))
