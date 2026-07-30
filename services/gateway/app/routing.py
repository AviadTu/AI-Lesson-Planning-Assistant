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


# Bare "create a NEW lesson" request with no topic and no sources — e.g.
# "צור מערך שיעור חדש", "מערך שיעור חדש", "בנה מערך שיעור". Matched only when the
# whole message is the create command (no trailing topic), so the Gateway can
# enter Lesson Mode and explain the source options WITHOUT running any search.
_NEW_LESSON_RE = re.compile(
    r"^\s*(?:אני\s+רוצה\s+)?"
    r"(?:צור|צרי|ליצור|תיצור|יצירת|בנה|בני|לבנות|תבנ[הי]|נבנה|בוא[יי]?\s+ניצור)?"
    r"\s*(?:לי\s+)?מער(?:ך|כי)\s*שיעור(?:\s+חדש)?\s*[.!?]*$",
    re.IGNORECASE,
)


def is_new_lesson_request(message: str) -> bool:
    return bool(_NEW_LESSON_RE.match(message or ""))


# Natural lesson-planning intent WITHOUT the explicit "צור"/"בנה" verbs — e.g.
# "אני רוצה שיעור על…", "נעשה שיעור על…", "שיעור בנושא…", "אני צריך מערך בנושא…".
# Matched only when the wording clearly expresses wanting/needing a lesson (a
# wanting verb next to שיעור/מערך, or a lesson explicitly framed "בנושא …"), so a
# bare topic with no lesson context does NOT trigger planning.
_LESSON_INTENT_RE = re.compile(
    r"(?:רוצ[הים]|צריכ[הים]?|צריך|מעוניינ|אשמח|נעשה|נכין|נלמד|בוא[יו]?\s+נ)"
    r"[^\n]{0,25}(?:שיעור|מער(?:ך|כי))"
    r"|(?:שיעור|מער(?:ך|כי))\s+בנושא\s+\S+",
    re.IGNORECASE,
)


def is_lesson_intent(message: str) -> bool:
    """Deterministic lesson-planning intent (explicit or natural phrasing)."""
    m = message or ""
    return bool(_LESSON_RE.search(m) or _LESSON_INTENT_RE.search(m))


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


def _deterministic_intent(message: str) -> str:
    """
    Rule-based intent used when the local classifier is UNAVAILABLE (no model
    call). Explicit/natural lesson requests are already routed deterministically
    before the classifier, so here we only separate a knowledge search (question
    form) from normal conversation — we never silently force a request to RAG.
    """
    if is_lesson_intent(message):
        return "lesson"
    if _looks_like_question(message):
        return "knowledge"
    return "conversation"


def classify_intent_llm(message: str) -> str:
    """
    LOCAL semantic router → 'lesson' | 'knowledge' | 'conversation'.

    Lets the model decide meaning instead of matching phrasings: a knowledge-base
    question, a lesson request, or normal conversation about the assistant/system.
    On failure/timeout it falls back to the deterministic rules (never a silent
    default to RAG).
    """
    system = (
        "You are a router for a teaching-assistant app. Read one user message "
        "and reply with EXACTLY ONE word.\n"
        "- 'conversation' = small talk, greetings, or a question about the "
        "assistant itself, what it can do, how to use the system, or what the "
        "system is for. Questions about YOU or the SYSTEM are conversation, not "
        "knowledge.\n"
        "- 'lesson' = a request to create, build, plan, brainstorm, or continue "
        "a lesson or lesson ideas.\n"
        "- 'knowledge' = a request for factual information/content found in the "
        "uploaded teaching documents.\n"
        "Reply with only one word: conversation, lesson, or knowledge."
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
        # Tolerant match: the local model often replies with a variant
        # ("conversación"/"conversazione"/"conversational") — accept the stem.
        if "convers" in content:
            return "conversation"
        if "lesson" in content:
            return "lesson"
        return "knowledge"
    except Exception as exc:  # noqa: BLE001 — classifier unavailable → deterministic fallback
        logger.warning(
            "intent classifier unavailable (%s); using deterministic fallback", exc
        )
        return _deterministic_intent(message)


def decide_route(mode: dict, message: str) -> str:
    """
    Return "lesson" (→ n8n), "conversation" (→ assistant) or "rag" (→ RAG).

    DETERMINISTIC only for the truly obvious cases: an active lesson session, a
    confirmed "build a lesson?" offer, and an explicit lesson-creation command.
    Everything else — including questions — is decided SEMANTICALLY by the local
    LLM router, which distinguishes a knowledge-base question, a lesson request,
    and normal conversation about the assistant/system. ``mode`` holds the
    session's persisted flags; this function is pure.
    """
    if mode.get("active"):                                     # active session
        return "lesson"
    if mode.get("pending_offer") and is_affirmative(message):  # accepted offer
        return "lesson"
    if is_explicit_lesson(message):
        # A lesson QUESTION ("is there a lesson on X?") is a search of the KB →
        # send it to RAG, whose fallback may offer to create one. A lesson
        # COMMAND ("build a lesson on X") goes straight to creation.
        return "rag" if _looks_like_question(message) else "lesson"
    if is_lesson_intent(message):                              # M4: natural lesson phrasing
        return "lesson"
    intent = classify_intent_llm(message)                      # semantic router
    if intent == "lesson":
        return "lesson"
    if intent == "conversation":
        return "conversation"
    return "rag"                                               # knowledge (default)
