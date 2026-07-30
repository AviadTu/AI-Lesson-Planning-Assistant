"""
The lesson-planning conversation, as a real LangGraph state machine.

One HTTP turn = one ``graph.invoke``. A SqliteSaver checkpointer persists the
full conversational state per ``thread_id`` (= session_id), which is the
agent's conversation memory. Each turn: absorb the message (local classifier,
appends it to history) → route on (stage, intent) → run the right node → return.

History is managed explicitly (each node returns the full updated transcript)
so a node always sees the messages appended by earlier nodes in the same turn.

Stages: intake → ideas → drafting → final (revisions continue in `final`).
Models (config.py): LOCAL for absorb/missing-info/selection; CLOUD for idea
generation, full lesson generation and revisions.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app import llms, prompts
from app.config import settings

logger = logging.getLogger("lesson_agent.graph")


class LessonState(TypedDict, total=False):
    session_id: str
    stage: str                    # source_review | intake | ideas | drafting | final
    user_message: str
    intent: str
    sources: list                 # per-turn sources as delivered by n8n
    in_warnings: list             # per-turn warnings input (from n8n, this turn)
    kept_sources: list            # reviewed/edited source set, persisted across turns
    warnings: list                # per-source failures reported by n8n (e.g. youtube)
    sources_reviewed: bool        # user approved the source review → generation allowed
    source_weights: str           # free-text "rely more/less" guidance for generation
    ideas: list
    selected_idea: dict
    lesson: dict
    reply: str
    needs_input: bool
    clarify_count: int            # how many clarification questions asked (capped)
    gate_ready: bool              # intake_gate → generate_ideas routing flag
    history: list                 # [{"role": "user"|"assistant", "content": str}]


def _with_reply(state: LessonState, reply: str, **extra) -> dict:
    """Build a node result that appends the assistant reply to history."""
    history = list(state.get("history") or []) + [{"role": "assistant", "content": reply}]
    return {"reply": reply, "needs_input": True, "history": history, **extra}


# ── presentation helpers (Hebrew rendering for the chat reply) ───────
def _render_ideas(ideas: list[dict]) -> str:
    lines = ["הנה כמה כיווני מערך אפשריים — אפשר לבחור אחד, לשלב, לבקש חלופות או לשנות:"]
    for i, idea in enumerate(ideas, start=1):
        lines.append(
            f"\n{i}. **{idea.get('title','')}**\n"
            f"{idea.get('description','')}\n"
            f"כיוון חינוכי: {idea.get('direction','')}"
        )
    lines.append("\nמה מתאים לך?")
    return "\n".join(lines)


def _render_lesson(lesson: dict) -> str:
    meta = lesson.get("meta") or {}
    head = "**מערך שיעור מוכן** 📘"
    if isinstance(meta, dict) and meta:
        head += (
            f"\nנושא: {meta.get('title','')} | כיתה/גיל: {meta.get('grade','')} | "
            f"משך: {meta.get('duration','')} | מטרה: {meta.get('objective','')}"
        )
    parts = [head]
    for key, label in prompts.LESSON_SECTIONS:
        if key == "meta":
            continue
        val = lesson.get(key)
        if val:
            parts.append(f"\n### {label}\n{val}")
    parts.append("\nאפשר לבקש שינויים, או לבקש שאפיק PDF / אשלח למייל.")
    return "\n".join(parts)


# ── source review (provenance-preserving) ───────────────────────────
import re as _re

# Hebrew label per normalized source type (provenance stays identifiable).
_SRC_LABEL = {
    "rag": "מאגר השיעורים", "document": "מסמך שהועלה", "pdf": "מסמך שהועלה",
    "docx": "מסמך שהועלה", "txt": "מסמך שהועלה", "image": "תמונה שהועלתה",
    "url": "מאמר / דף אינטרנט", "youtube": "יוטיוב", "web": "חיפוש באינטרנט (Perplexity)",
    "text": "טקסט",
}

# Approve vs edit intents in the source-review stage (deterministic keywords —
# the local model is unreliable here).
_APPROVE_RE = _re.compile(
    r"אשר|מאשר|מאשרת|המשך|תמשיך|נמשיך|בסדר|מקובל|מתאים|יאללה|קדימה|"
    r"בנה|תבנה|נבנה|צור|תפיק|להפיק|approve|proceed|continue|generate|go\b|ok\b|yes\b",
    _re.IGNORECASE,
)
_REMOVE_RE = _re.compile(
    r"הסר|הסירי|תסיר|הורד|תוריד|התעלם|בטל|בלי|remove|ignore|drop|delete|exclude",
    _re.IGNORECASE,
)
_RELY_RE = _re.compile(
    r"הסתמך|תסתמך|להסתמך|יותר|פחות|תתמקד|התמקד|דגש|rely|more|less|focus|weight|emphas",
    _re.IGNORECASE,
)
# Cancel lesson creation, and change the lesson topic — natural actions at the
# source-review stage (so the user is never stuck repeating the same review).
_CANCEL_RE = _re.compile(
    r"בטל|ביטול|לבטל|עצור|תעצור|הפסק|התחל\s+מחדש|לא\s+משנה|עזוב|cancel|stop\b|reset",
    _re.IGNORECASE,
)
_CHANGE_TOPIC_RE = _re.compile(
    r"שנה\s+(?:את\s+ה)?נושא|נושא\s+אחר|במקום\s+זה|בעצם|החלף\s+נושא|"
    r"(?:רוצ[הים]|צריכ[הים]?|צריך|נעשה|נלמד|אשמח)[^\n]{0,25}(?:שיעור|מער(?:ך|כי))",
    _re.IGNORECASE,
)
# Explicit or natural lesson REQUEST → start structured planning (intake), not
# a free-form chat. A bare topic with no lesson wording does NOT match, so we
# still don't start planning without the user asking for a lesson.
_LESSON_REQUEST_RE = _re.compile(
    r"מער(?:ך|כי)\s*שיעור"
    r"|(?:בנ[הי]|לבנות|תבנ[הי]|נבנה|צור|צרי|ליצור|הפק|להפיק|תכנן|לתכנן|"
    r"רוצ[הים]|צריכ[הים]?|צריך|מעוניינ|אשמח|נעשה|נכין|נלמד)[^\n]{0,25}(?:שיעור|מער(?:ך|כי))"
    r"|(?:שיעור|מער(?:ך|כי))\s+בנושא\s+\S+",
    _re.IGNORECASE,
)


# Teacher explicitly opts into the lesson repository (otherwise retrieval stays
# an invisible implementation detail and never surfaces as a source).
_REPO_RE = _re.compile(
    r"מאגר\s+השיעורים|מהמאגר|במאגר\s+השיעורים|שיעורים\s+קיימים|חפש[^\n]{0,12}מאגר",
    _re.IGNORECASE,
)


def _chosen_sources(state: LessonState) -> list:
    """
    Sources the TEACHER actually chose — the only ones that appear in the Source
    Review or ground generation. Uploaded document/image, article, YouTube and
    web search are always chosen; repository (RAG) results count only when the
    teacher explicitly asked for the lesson repository. A bare topic (RAG only,
    no explicit request) yields [] — so no Source Review appears and retrieval
    is never exposed to the teacher.
    """
    srcs = state.get("sources") or []
    chosen = [s for s in srcs if (s.get("type") or "").lower() != "rag"]
    if _REPO_RE.search(state.get("user_message", "") or ""):
        chosen += [s for s in srcs if (s.get("type") or "").lower() == "rag"]
    return chosen


def _effective_sources(state: LessonState) -> list:
    """Generation uses the approved reviewed set if present, else the chosen set."""
    kept = state.get("kept_sources")
    return kept if kept is not None else _chosen_sources(state)


def _history_with_weights(state: LessonState) -> list:
    """History plus the user's rely-more/less guidance, so generation honours it."""
    h = list(state.get("history") or [])
    w = state.get("source_weights")
    if w:
        h = h + [{"role": "user", "content": "(הנחיית משקל למקורות: " + w + ")"}]
    return h


def _merge_sources(existing: list, incoming: list) -> list:
    """Append incoming sources not already present (dedupe by type+title/url)."""
    out = list(existing or [])
    seen = {(s.get("type"), s.get("title"), s.get("url")) for s in out}
    for s in incoming or []:
        key = (s.get("type"), s.get("title"), s.get("url"))
        if key not in seen:
            out.append(s)
            seen.add(key)
    return out


def _render_source_review(kept: list, warnings: list, weights: str = "") -> str:
    """Grouped, per-source provenance review (never merged into one summary)."""
    lines = ["**סקירת המקורות שנאספו** — בדוק/י לפני יצירת המערך:"]
    if not kept:
        lines.append("\n(לא נאספו מקורות תקינים.)")
    for i, s in enumerate(kept, start=1):
        label = _SRC_LABEL.get((s.get("type") or "").lower(), s.get("type") or "מקור")
        ident = s.get("title") or s.get("url") or s.get("citation") or "—"
        summary = _re.sub(r"\s+", " ", (s.get("text") or "")).strip()[:200]
        lines.append(
            f"\n{i}. **[{label}]** {ident}"
            f"\n   סטטוס: ✅ עובד"
            f"\n   תקציר: {summary or '(אין טקסט)'}"
        )
    if warnings:
        lines.append("\n\n**מקורות שלא נכללו / אזהרות:**")
        for w in warnings:
            lines.append(f"- ⚠️ {w}")
    if weights:
        lines.append(f"\n\n**הנחיית משקל:** {weights}")
    lines.append(
        "\n\nאפשר: לאשר (\"אשר\" / \"המשך\") ליצירת המערך, להסיר מקור (\"הסר מקור 2\"), "
        "לבקש להסתמך יותר/פחות על מקור, להוסיף מקור חדש בפאנל, או לחזור על חיפוש הרשת."
    )
    return "\n".join(lines)


# ── nodes ────────────────────────────────────────────────────────────
def absorb(state: LessonState) -> dict:
    """LOCAL: classify only this turn's intent (a label); append the message."""
    msg = state.get("user_message", "") or ""
    intent = "other"
    try:
        res = llms.local_json(
            prompts.CLASSIFY_SYSTEM,
            prompts.classify_user(state.get("stage", "intake"), msg),
        )
        intent = res.get("intent", "other")
    except Exception as exc:  # noqa: BLE001 — classifier is best-effort
        logger.warning("absorb classify failed: %s", exc)
    history = list(state.get("history") or [])
    if msg:
        history = history + [{"role": "user", "content": msg}]
    return {"intent": intent, "history": history}


def present_source_review(state: LessonState) -> dict:
    """Show a grouped, per-source provenance review and wait for approval/edits."""
    incoming = _chosen_sources(state)
    kept = state.get("kept_sources")
    kept = list(incoming) if kept is None else _merge_sources(kept, incoming)
    inc_warn = state.get("in_warnings") or []
    warnings = inc_warn if inc_warn else (state.get("warnings") or [])
    review = _render_source_review(kept, warnings, state.get("source_weights", ""))
    return _with_reply(
        state, review, stage="source_review", kept_sources=kept, warnings=warnings,
        sources_reviewed=False,
    )


def apply_source_edit(state: LessonState) -> dict:
    """Apply a source edit (remove / rely-more-less / add) and re-present the review."""
    kept = list(state.get("kept_sources") or [])
    kept = _merge_sources(kept, _chosen_sources(state))  # newly added via panel
    msg = state.get("user_message", "") or ""
    weights = state.get("source_weights", "") or ""
    m = _re.search(r"\d+", msg)
    idx = int(m.group()) if m else None
    if _REMOVE_RE.search(msg) and idx and 1 <= idx <= len(kept):
        kept.pop(idx - 1)
    elif _RELY_RE.search(msg):
        weights = (weights + " | " if weights else "") + msg.strip()
    warnings = state.get("warnings") or []
    review = _render_source_review(kept, warnings, weights)
    return _with_reply(
        state, review, stage="source_review", kept_sources=kept, source_weights=weights,
        sources_reviewed=False,
    )


def approve_sources(state: LessonState) -> dict:
    """User approved the source review → latch it; generation may proceed."""
    return {"sources_reviewed": True, "stage": "intake"}


def _reset_lesson_state() -> dict:
    """Clear all per-lesson working state so planning can restart cleanly."""
    return {
        "stage": "intake", "sources_reviewed": False, "kept_sources": None,
        "source_weights": "", "ideas": [], "selected_idea": {}, "lesson": None,
        "clarify_count": 0,
    }


def cancel_lesson(state: LessonState) -> dict:
    """Cancel lesson creation from the source review; reset and offer a restart."""
    reply = (
        "ביטלתי את יצירת מערך השיעור. אפשר להתחיל מחדש — כתבו נושא חדש לשיעור, "
        "או פשוט לשאול אותי כל דבר אחר."
    )
    history = list(state.get("history") or []) + [{"role": "assistant", "content": reply}]
    return {"reply": reply, "needs_input": True, "history": history, **_reset_lesson_state()}


def change_topic(state: LessonState) -> dict:
    """Discard current sources/ideas and restart intake with the new topic."""
    # No reply here — flow continues to intake_gate, which advances to ideas
    # (or asks one clarification) using the new topic already in history.
    return _reset_lesson_state()


def source_review_help(state: LessonState) -> dict:
    """Unexpected input at source review → explain the available choices."""
    reply = (
        "לא בטוח/ה שהבנתי. בשלב סקירת המקורות אפשר:\n"
        "• לאשר ולהמשיך — כתבו \"אשר\" או \"המשך\".\n"
        "• להסיר מקור — למשל \"הסר מקור 2\".\n"
        "• להוסיף מקור — צרפו אותו בפאנל \"מקורות לשיעור\".\n"
        "• לשנות נושא — למשל \"שנה נושא לנושא אחר\".\n"
        "• לבטל — כתבו \"בטל\"."
    )
    return _with_reply(state, reply, stage="source_review")


def intake_gate(state: LessonState) -> dict:
    """LOCAL: enough info to propose ideas? If not, ask one focused question.

    The local model only reports a sufficiency boolean + which field is missing
    (a label); the question text itself is a deterministic Hebrew template.

    The local judgment is a signal, not a hard gate: to keep the conversation
    reliably advancing (weak local models are non-deterministic on this call),
    we ask at most ONE clarification, then proceed to ideas regardless — the
    cloud model works from whatever the teacher provided.
    """
    history = state.get("history") or []
    clarify_count = int(state.get("clarify_count") or 0)
    sufficient = False
    missing = "topic"
    try:
        res = llms.local_json(
            prompts.MISSING_SYSTEM,
            prompts.missing_info(history, bool(state.get("sources"))),
        )
        sufficient = bool(res.get("sufficient"))
        missing = (res.get("missing") or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("intake_gate failed: %s", exc)
    if sufficient or clarify_count >= 1:
        return {"stage": "intake", "gate_ready": True}
    q = prompts.CLARIFY_QUESTIONS.get(missing, prompts.CLARIFY_QUESTIONS[""])
    result = _with_reply(state, q, stage="intake", gate_ready=False)
    result["clarify_count"] = clarify_count + 1
    return result


def generate_ideas(state: LessonState) -> dict:
    """CLOUD: propose 2–4 lesson ideas (title, description, direction)."""
    ideas: list[dict] = []
    try:
        res = llms.cloud_json(
            prompts.IDEAS_SYSTEM,
            prompts.ideas_prompt(_history_with_weights(state), _effective_sources(state)),
        )
        ideas = res.get("ideas") or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("generate_ideas failed: %s", exc)
    if not ideas:
        return _with_reply(
            state,
            "לא הצלחתי להפיק רעיונות כרגע. אפשר לנסות שוב או לחדד את הנושא?",
            stage="intake",
        )
    ideas = ideas[:4]
    return _with_reply(state, _render_ideas(ideas), stage="ideas", ideas=ideas)


def resolve_selection(state: LessonState) -> dict:
    """LOCAL: resolve which idea the user chose → selected_idea (drafting)."""
    ideas = state.get("ideas") or []
    msg = state.get("user_message", "") or ""
    idx = 0
    try:
        res = llms.local_json(prompts.SELECT_SYSTEM, prompts.select_prompt(ideas, msg))
        idx = int(res.get("index", 0) or 0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("resolve_selection failed: %s", exc)
    chosen = ideas[idx - 1] if 1 <= idx <= len(ideas) else (ideas[0] if ideas else {})
    title = chosen.get("title", "הכיוון שבחרת")
    confirm = (
        f'נהדר — נתקדם עם "{title}". '
        "רוצה להוסיף דגשים או שינויים, או שאפיק עכשיו את מערך השיעור המלא?"
    )
    return _with_reply(state, confirm, stage="drafting", selected_idea=chosen)


# A complete lesson must carry meta (with a title) and every core section with
# real content — the guarantee that a partial/truncated plan is never returned.
_REQUIRED_SECTIONS = ["opening", "central_idea", "activity", "central_discussion", "summary"]


def _lesson_complete(lesson: dict) -> tuple[bool, str]:
    """Return (ok, reason). A lesson is complete only when meta + every required
    section are present with non-trivial content (truncation leaves them missing
    or empty)."""
    if not isinstance(lesson, dict) or not lesson:
        return False, "empty"
    meta = lesson.get("meta")
    if not isinstance(meta, dict) or not (meta.get("title") or "").strip():
        return False, "meta"
    missing = [
        k for k in _REQUIRED_SECTIONS
        if not isinstance(lesson.get(k), str) or len(lesson.get(k, "").strip()) < 30
    ]
    if missing:
        return False, "missing:" + ",".join(missing)
    return True, ""


def _try_generate_lesson(state: LessonState) -> dict:
    """One full-lesson generation attempt; returns the lesson dict (may be {})."""
    try:
        res = llms.cloud_json(
            prompts.LESSON_SYSTEM,
            prompts.lesson_prompt(
                _history_with_weights(state),
                state.get("selected_idea") or {},
                _effective_sources(state),
            ),
        )
        return res.get("lesson") or {}
    except Exception as exc:  # noqa: BLE001 — treat as an incomplete attempt
        logger.warning("generate_lesson attempt failed: %s", exc)
        return {}


def generate_lesson(state: LessonState) -> dict:
    """CLOUD: generate a COMPLETE lesson; validate, regenerate once, else fail.

    A partial/truncated plan is never returned — if both attempts are incomplete
    we return a clear error and stay in `drafting` so the user can retry.
    """
    reason = "empty"
    for attempt in range(2):
        lesson = _try_generate_lesson(state)
        ok, reason = _lesson_complete(lesson)
        if ok:
            return _with_reply(state, _render_lesson(lesson), stage="final", lesson=lesson)
        logger.warning("generate_lesson incomplete (%s) on attempt %d", reason, attempt + 1)
    return _with_reply(
        state,
        "לא הצלחתי להפיק מערך שיעור שלם כרגע (ייתכן שהתשובה נקטעה). אפשר לנסות שוב "
        "ואפיק את המערך המלא מחדש.",
        stage="drafting",
    )


def revise_lesson(state: LessonState) -> dict:
    """CLOUD: apply the user's requested change to the existing lesson."""
    current = state.get("lesson") or {}
    lesson = current
    try:
        res = llms.cloud_json(
            prompts.REVISE_SYSTEM,
            prompts.revise_prompt(current, state.get("user_message", ""), _effective_sources(state)),
        )
        cand = res.get("lesson") or {}
        # Only accept a revision that is still complete — never downgrade a full
        # lesson to a truncated one.
        ok, reason = _lesson_complete(cand)
        if ok:
            lesson = cand
        else:
            logger.warning("revise_lesson produced incomplete lesson (%s); keeping current", reason)
    except Exception as exc:  # noqa: BLE001
        logger.warning("revise_lesson failed: %s", exc)
    return _with_reply(state, _render_lesson(lesson), stage="final", lesson=lesson)


def assist(state: LessonState) -> dict:
    """LOCAL: a natural, on-topic fallback reply that keeps the flow going."""
    reply = "אני הַמּוֹרִי — אפשר לשאול אותי שאלות על מאגר השיעורים או לבנות יחד מערך שיעור. במה לסייע?"
    try:
        reply = llms.local_chat(
            prompts.SYSTEM_BASE,
            f"Stage: {state.get('stage')}\nUser: {state.get('user_message','')}\n"
            "Reply briefly and naturally in Hebrew.",
        ) or reply
    except Exception as exc:  # noqa: BLE001
        logger.warning("assist failed: %s", exc)
    return _with_reply(state, reply)


# ── routing ──────────────────────────────────────────────────────────
def after_absorb(state: LessonState) -> str:
    stage = state.get("stage") or "intake"
    intent = state.get("intent") or "other"
    msg = state.get("user_message", "") or ""

    # ── Source-review gate: sources must be reviewed & approved before any
    # generation. Present the grouped review first; then edit or approve. ──
    if stage == "source_review":
        if _CANCEL_RE.search(msg):
            return "cancel_lesson"        # cancel creation → reset
        if _CHANGE_TOPIC_RE.search(msg):
            return "change_topic"         # new topic → restart intake
        if _REMOVE_RE.search(msg) or _RELY_RE.search(msg):
            return "apply_source_edit"    # remove / rely-more-less
        if _APPROVE_RE.search(msg):
            return "approve_sources"      # approved → latch, then generate
        if _chosen_sources(state):
            return "apply_source_edit"    # a new source was added in the panel
        return "source_review_help"       # unexpected → explain the choices
    if not state.get("sources_reviewed") and _chosen_sources(state):
        return "present_source_review"    # teacher-chosen sources → review first

    if stage == "ideas":
        if intent in ("request_alternatives", "revise"):
            return "generate_ideas"
        return "resolve_selection"
    if stage == "drafting":
        if intent == "request_alternatives":
            return "generate_ideas"
        return "generate_lesson"
    if stage == "final":
        if intent == "request_alternatives":
            return "generate_ideas"
        if intent in ("revise", "provide_info", "generate_lesson", "approve"):
            return "revise_lesson"
        return "assist"
    # intake / unknown. Before any lesson intake has begun (no clarification
    # asked yet), a message that isn't an explicit lesson command is normal
    # conversation → answer it naturally (assist). An explicit lesson command
    # ('generate_lesson') starts intake; and once intake is underway, keep
    # collecting the lesson info rather than chatting.
    clarify_count = int(state.get("clarify_count") or 0)
    if _LESSON_REQUEST_RE.search(msg):    # explicit/natural lesson request → plan
        return "intake_gate"
    if clarify_count == 0 and intent in ("question", "other", "provide_info"):
        return "assist"
    return "intake_gate"


def after_gate(state: LessonState) -> str:
    return "generate_ideas" if state.get("gate_ready") else END


def after_selection(state: LessonState) -> str:
    return "generate_lesson" if state.get("intent") == "generate_lesson" else END


# ── build & compile ──────────────────────────────────────────────────
def _build():
    b = StateGraph(LessonState)
    b.add_node("absorb", absorb)
    b.add_node("present_source_review", present_source_review)
    b.add_node("apply_source_edit", apply_source_edit)
    b.add_node("approve_sources", approve_sources)
    b.add_node("cancel_lesson", cancel_lesson)
    b.add_node("change_topic", change_topic)
    b.add_node("source_review_help", source_review_help)
    b.add_node("intake_gate", intake_gate)
    b.add_node("generate_ideas", generate_ideas)
    b.add_node("resolve_selection", resolve_selection)
    b.add_node("generate_lesson", generate_lesson)
    b.add_node("revise_lesson", revise_lesson)
    b.add_node("assist", assist)

    b.add_edge(START, "absorb")
    b.add_conditional_edges("absorb", after_absorb,
                            ["present_source_review", "apply_source_edit",
                             "approve_sources", "cancel_lesson", "change_topic",
                             "source_review_help", "intake_gate", "generate_ideas",
                             "resolve_selection", "generate_lesson", "revise_lesson",
                             "assist"])
    b.add_edge("approve_sources", "intake_gate")
    b.add_edge("change_topic", "intake_gate")
    b.add_conditional_edges("intake_gate", after_gate, ["generate_ideas", END])
    b.add_conditional_edges("resolve_selection", after_selection, ["generate_lesson", END])
    b.add_edge("present_source_review", END)
    b.add_edge("apply_source_edit", END)
    b.add_edge("cancel_lesson", END)
    b.add_edge("source_review_help", END)
    b.add_edge("generate_ideas", END)
    b.add_edge("generate_lesson", END)
    b.add_edge("revise_lesson", END)
    b.add_edge("assist", END)
    return b


_graph = None


def get_graph():
    """Compile once with a persistent SqliteSaver checkpointer."""
    global _graph
    if _graph is None:
        os.makedirs(os.path.dirname(settings.CHECKPOINT_DB_PATH) or ".", exist_ok=True)
        conn = sqlite3.connect(settings.CHECKPOINT_DB_PATH, check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        _graph = _build().compile(checkpointer=saver)
    return _graph


def run_turn(
    session_id: str, message: str, sources: list[dict], warnings: list | None = None
) -> dict:
    """Advance the conversation one turn; returns the merged state snapshot."""
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}
    graph.invoke(
        {
            "session_id": session_id,
            "user_message": message,
            "sources": sources or [],
            "in_warnings": warnings or [],
        },
        config,
    )
    return dict(graph.get_state(config).values)


def get_session_state(session_id: str) -> dict:
    """Read the persisted conversational state for a session (no new turn)."""
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}
    snap = graph.get_state(config)
    return dict(snap.values) if snap and snap.values else {}
