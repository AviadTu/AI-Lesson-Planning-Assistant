"""
Intent routing for the Gateway (keeps /chat lightweight).

Decides whether a chat turn is an ordinary knowledge question (→ RAG, the
existing independent path) or a lesson-plan create/continue request (→ n8n).

Order of decision (cheapest / most certain first):
  1. Session already in lesson-planning mode  → n8n.
  2. Pending "shall I build a lesson?" offer + an affirmative reply → n8n.
  3. Explicit lesson request (deterministic keyword rules) → n8n.
  4. Clearly a knowledge question (question form, no lesson keyword) → RAG.
  5. Otherwise ambiguous → a lightweight LOCAL LLM classifier decides.

The local classifier is only called for genuinely ambiguous turns, so the
common RAG path stays fast.
"""

from __future__ import annotations

import logging
import re

import httpx

from app.config import settings

logger = logging.getLogger("gateway.routing")

# Explicit lesson-plan intent (Hebrew + English). "מערך שיעור" is a strong
# domain signal; creation/finding verbs next to "שיעור"; idea requests; and the
# common English phrasings.
_LESSON_RE = re.compile(
    # "מערך שיעור" is the strongest domain signal. Note: Hebrew prefixes attach
    # to the word (e.g. "לשיעור"), so we do NOT require a \b before Hebrew nouns.
    r"מער(?:ך|כי)\s*שיעור"
    r"|(?:בנ[הי]|לבנות|תבנ[הי]|נבנה|צור|צרי|ליצור|תיצור|הפק|תפיק|להפיק|תכנן|"
    r"לתכנן|מצא|תמצא)[^\n]{0,40}שיעור"
    r"|רעיונ(?:ות|ים)?[^\n]{0,30}(?:שיעור|מערך)"
    r"|lesson\s*plan|(?:create|build|generate|design|plan|make)\b[^\n]{0,30}\blesson",
    re.IGNORECASE,
)

# Affirmative reply to a "shall I build a lesson?" offer.
_AFFIRM_RE = re.compile(
    r"^\s*(?:כן|בטח|בסדר|אשמח|בוא[יי]?|נסה|תבנ[הי]|יאללה|סבבה|"
    r"ok|okay|sure|yes|yep|yeah|please)\b",
    re.IGNORECASE,
)

# Knowledge-question form (question mark or leading interrogative) with no
# lesson keyword → route straight to RAG without the classifier.
_QUESTION_RE = re.compile(
    r"\?\s*$"
    r"|^\s*(?:מה|מי|מתי|איפה|כמה|האם|למה|מדוע|איך|כיצד|היכן|"
    r"what|who|when|where|why|how|is|are|does|do|can)\b",
    re.IGNORECASE,
)

# Teaching-context signal: words that hint the turn is about a lesson/class even
# though it isn't an explicit build request. Used ONLY to gate the rare LLM
# fallback — everything without this signal is decided deterministically.
_LESSON_SIGNAL_RE = re.compile(
    r"שיעור|מער(?:ך|כי)|כית[הת]|תלמיד|מור[הת]|הורא[הת]|פעילות|לימוד|מחנכ|"
    r"lesson|class(?:room)?|teach|student|pupil|grade|curriculum|activity",
    re.IGNORECASE,
)


def is_explicit_lesson(message: str) -> bool:
    return bool(_LESSON_RE.search(message or ""))


def is_affirmative(message: str) -> bool:
    return bool(_AFFIRM_RE.search(message or ""))


# PDF / download / email request (checked only within an active lesson session,
# to route to the "Lesson PDF and Email" workflow instead of a normal turn).
_EXPORT_RE = re.compile(
    r"\bpdf\b|פי\s?די\s?אף"
    r"|תוריד|להוריד|הורד[הת]|להורדה"
    r"|למייל|במייל|אימייל|\bemail\b|דוא[\"׳']?ל|שלח.{0,12}מייל",
    re.IGNORECASE,
)


def is_export_request(message: str) -> bool:
    return bool(_EXPORT_RE.search(message or ""))


# Explicit web-search (Perplexity) request signal.
_WEB_SEARCH_RE = re.compile(
    r"חפש\s+(?:ב)?(?:אינטרנט|רשת|גוגל|רשת)|באינטרנט|ברשת|perplexity|web\s*search",
    re.IGNORECASE,
)


def is_web_search_request(message: str) -> bool:
    return bool(_WEB_SEARCH_RE.search(message or ""))


# Pure greetings / thanks — handled deterministically (no RAG/embeddings/Chroma/
# n8n/lesson-agent/LLM). Matched only when the whole message is a greeting.
_GREETING_RE = re.compile(
    r"^\s*(?:הי+|היי+|שלום|אהלן|אהלה|היה?לו|בוקר\s+טוב|צהריים\s+טובים|ערב\s+טוב|"
    r"לילה\s+טוב|מה\s+נשמע|מה\s+קורה|מה\s+המצב|"
    r"hi|hello|hey|yo|good\s+(?:morning|evening|afternoon|day))"
    r"[\s!.,?׃־]*$",
    re.IGNORECASE,
)
_THANKS_RE = re.compile(
    r"^\s*(?:תודה(?:\s+רבה)?|תודות|יישר\s+כוח|thanks|thank\s+you|thx|ty|cheers)"
    r"[\s!.,?׃־]*$",
    re.IGNORECASE,
)


def greeting_kind(message: str) -> str | None:
    """Return 'greeting' | 'thanks' for a pure greeting message, else None."""
    m = message or ""
    if _THANKS_RE.match(m):
        return "thanks"
    if _GREETING_RE.match(m):
        return "greeting"
    return None


def _looks_like_question(message: str) -> bool:
    return bool(_QUESTION_RE.search((message or "").strip()))


def _has_lesson_signal(message: str) -> bool:
    return bool(_LESSON_SIGNAL_RE.search(message or ""))


def classify_intent_llm(message: str) -> str:
    """LOCAL lightweight classifier for ambiguous turns → 'lesson' | 'knowledge'."""
    system = (
        "Classify the teacher's message as exactly one word: 'lesson' if they "
        "want to create/build/plan or get ideas for a lesson plan, or "
        "'knowledge' if they are asking a general/factual question. Answer with "
        "only that one word."
    )
    url = settings.LOCAL_OLLAMA_BASE_URL.rstrip("/") + "/api/chat"
    payload = {
        "model": settings.INTENT_MODEL,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 5},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": message or ""},
        ],
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            content = (resp.json().get("message") or {}).get("content", "").lower()
        return "lesson" if "lesson" in content else "knowledge"
    except Exception as exc:  # noqa: BLE001 — on failure, prefer the safe RAG path
        logger.warning("intent classifier failed (%s); defaulting to knowledge", exc)
        return "knowledge"


def decide_route(mode: dict, message: str) -> str:
    """
    Return "lesson" (→ n8n) or "rag" (→ RAG) for this turn.

    Routing is DETERMINISTIC on the hot path (no LLM): active lesson session,
    explicit lesson request, confirmation of a RAG-insufficient offer, and
    ordinary knowledge questions are all decided by cheap rules. The local LLM
    classifier is a *rare* fallback, used only when the message carries
    teaching-context signal yet wasn't an explicit request — a genuinely
    ambiguous case. Everything else defaults to RAG (a bare topic the user
    actually wanted a lesson for is still caught later by the RAG-insufficient
    offer). ``mode`` holds the session's persisted flags; this function is pure.
    """
    if mode.get("active"):                                     # active session
        return "lesson"
    if mode.get("pending_offer") and is_affirmative(message):  # accepted offer
        return "lesson"
    if is_explicit_lesson(message):                            # explicit request
        return "lesson"
    if _looks_like_question(message):                          # knowledge question
        return "rag"
    if _has_lesson_signal(message):                            # ambiguous → rare LLM
        return "lesson" if classify_intent_llm(message) == "lesson" else "rag"
    return "rag"                                               # deterministic default
