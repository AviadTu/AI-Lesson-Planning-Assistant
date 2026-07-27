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
    stage: str                    # intake | ideas | drafting | final
    user_message: str
    intent: str
    sources: list
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
            prompts.ideas_prompt(state.get("history") or [], state.get("sources") or []),
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


def generate_lesson(state: LessonState) -> dict:
    """CLOUD: generate the full lesson plan from the chosen direction."""
    lesson: dict = {}
    try:
        res = llms.cloud_json(
            prompts.LESSON_SYSTEM,
            prompts.lesson_prompt(
                state.get("history") or [],
                state.get("selected_idea") or {},
                state.get("sources") or [],
            ),
        )
        lesson = res.get("lesson") or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("generate_lesson failed: %s", exc)
    if not lesson:
        return _with_reply(state, "אירעה תקלה בהפקת מערך השיעור המלא. נסה/י שוב בבקשה.")
    return _with_reply(state, _render_lesson(lesson), stage="final", lesson=lesson)


def revise_lesson(state: LessonState) -> dict:
    """CLOUD: apply the user's requested change to the existing lesson."""
    current = state.get("lesson") or {}
    lesson = current
    try:
        res = llms.cloud_json(
            prompts.REVISE_SYSTEM,
            prompts.revise_prompt(current, state.get("user_message", ""), state.get("sources") or []),
        )
        lesson = res.get("lesson") or current
    except Exception as exc:  # noqa: BLE001
        logger.warning("revise_lesson failed: %s", exc)
    return _with_reply(state, _render_lesson(lesson), stage="final", lesson=lesson)


def assist(state: LessonState) -> dict:
    """LOCAL: a natural, on-topic fallback reply that keeps the flow going."""
    reply = "אני כאן כדי לעזור לבנות מערך שיעור. על איזה נושא נעבוד, או מה תרצה/י לשנות?"
    try:
        reply = llms.local_chat(
            prompts.SYSTEM_BASE,
            f"Stage: {state.get('stage')}\nUser: {state.get('user_message','')}\n"
            "Reply briefly in Hebrew, guiding them toward building/adjusting a lesson.",
        ) or reply
    except Exception as exc:  # noqa: BLE001
        logger.warning("assist failed: %s", exc)
    return _with_reply(state, reply)


# ── routing ──────────────────────────────────────────────────────────
def after_absorb(state: LessonState) -> str:
    stage = state.get("stage") or "intake"
    intent = state.get("intent") or "other"
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
    return "intake_gate"          # intake / unknown


def after_gate(state: LessonState) -> str:
    return "generate_ideas" if state.get("gate_ready") else END


def after_selection(state: LessonState) -> str:
    return "generate_lesson" if state.get("intent") == "generate_lesson" else END


# ── build & compile ──────────────────────────────────────────────────
def _build():
    b = StateGraph(LessonState)
    b.add_node("absorb", absorb)
    b.add_node("intake_gate", intake_gate)
    b.add_node("generate_ideas", generate_ideas)
    b.add_node("resolve_selection", resolve_selection)
    b.add_node("generate_lesson", generate_lesson)
    b.add_node("revise_lesson", revise_lesson)
    b.add_node("assist", assist)

    b.add_edge(START, "absorb")
    b.add_conditional_edges("absorb", after_absorb,
                            ["intake_gate", "generate_ideas", "resolve_selection",
                             "generate_lesson", "revise_lesson", "assist"])
    b.add_conditional_edges("intake_gate", after_gate, ["generate_ideas", END])
    b.add_conditional_edges("resolve_selection", after_selection, ["generate_lesson", END])
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


def run_turn(session_id: str, message: str, sources: list[dict]) -> dict:
    """Advance the conversation one turn; returns the merged state snapshot."""
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}
    graph.invoke(
        {"session_id": session_id, "user_message": message, "sources": sources or []},
        config,
    )
    return dict(graph.get_state(config).values)
