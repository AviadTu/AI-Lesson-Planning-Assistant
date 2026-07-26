"""
LangGraph orchestration for the RAG service (skeleton).

Used "when needed" for multi-step reasoning over retrieved lesson plans
(retrieve -> combine/adapt -> generate).  Business logic intentionally not
implemented yet.

Planned nodes:
  * retrieve   — ChromaDB similarity search (app.vectorstore.chroma)
  * generate   — LLM via the provider abstraction (app.llm.provider), using the
                 Hebrew lesson-plan system prompt + the bounded conversation
                 history passed in by the Gateway (conversation memory).
  * cite       — map retrieved chunks back to {source, doc_id} for the UI.
"""

from __future__ import annotations

_graph = None


def get_graph():
    """Return a cached compiled LangGraph app.

    TODO: build a StateGraph wiring retrieve -> generate -> cite and compile it.
    A checkpointer may back conversation memory if the injected history window
    proves insufficient.
    """
    raise NotImplementedError("LangGraph pipeline not implemented yet.")


def run(session_id: str, message: str, history: list[dict]) -> dict:
    """Execute the RAG graph for one turn. Returns {"answer", "context"}."""
    raise NotImplementedError("LangGraph run not implemented yet.")
