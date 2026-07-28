# n8n Workflows — Lesson Planning

Three workflows orchestrate lesson-plan creation. RAG stays an independent
service; n8n only orchestrates. Exported JSON in this folder is the source of
truth (re-import to restore). **Modify in place — do not create duplicates;
preserve the IDs below** (the orchestrator references the sub-workflows by ID).

| Workflow | ID | Nodes | Role |
|---|---|---|---|
| Lesson Planning Orchestrator | `UEfy0HlFG540p1N1` | 10 | main pipeline (webhook → validate → source processing → agent → respond) |
| Lesson Source Processing | `yulzQcUFjXD4fVg7` | 30 | 6 source branches → unified `{sources, warnings, failed_sources, count}` |
| Lesson PDF and Email | `b8eVEHPDWsKEGztX` | 8 | render PDF (Gateway) → optional email |

## Source branches (Lesson Source Processing)

`Source Input → Detect Source Types → Switch →` one branch per source type,
each its own node(s), then `Merge Sources → Remove Duplicates → Preserve
Citations → Return Unified Context`:

1. **RAG** — HTTP Request → RAG `/retrieve` (independent service) → Normalize.
2. **Article URL** — Fetch Article (HTTP) → Extract HTML Content (native **HTML**) → Normalize.
3. **YouTube** — Extract Video ID → YouTube Page (HTTP) → Extract Caption URL → Retrieve Transcript (HTTP) → Normalize. *(HTTP transcript: no native captions node.)*
4. **Perplexity** — native **Perplexity** node (v2, `chat`→`complete`, `sonar`), connected to the **Perplexity account** credential → Normalize Web (reads the simplified `message` string + `citations`).
5. **Document** — To Binary (native **Convert to File**) → Detect File Type → Switch → **Extract From File** (PDF/TXT, native) / Gateway `/internal/extract` (DOCX) → Normalize.
6. **Image** — Image To Binary → **Analyze Image** (native **OpenAI**), connected to the **OpenAI account** credential → Normalize.

Every branch is partial-failure tolerant (`onError: continueRegularOutput`);
failures are recorded in `failed_sources` + `warnings` and the workflow always
returns exactly one item.

## Credentials

| Credential | Used by | Status |
|---|---|---|
| **Perplexity account** (`perplexityApi`) | Perplexity Search node | connected & enabled ✓ |
| **OpenAI account** (`openAiApi`) | Analyze Image node | connected & enabled ✓ |
| SMTP | Send Email node (PDF/Email wf) | **not configured** — node stays disabled; email is optional (the PDF download works without it) |

Perplexity and OpenAI use the operator's already-authenticated credentials. The
Send Email node is the only credential-gated node still **disabled**: add an SMTP
credential (or switch it to the native Gmail node, since a Gmail credential
exists) and toggle it on. No redesign needed.

## Environment variables

**n8n container** (`$env`, read inside Code nodes / expressions):

| Var | Effect |
|---|---|
| `SMTP_FROM` | from-address for the Send Email node (only if email is enabled) |

The web-search branch is gated by the request itself (`want_web` / a "search the
web" keyword) — the Perplexity credential handles auth — and image attachments
are processed whenever an image is uploaded. No `PERPLEXITY_API_KEY` /
`IMAGE_PROVIDER` env flags are needed.

**Gateway** (`services/gateway/.env`): `N8N_LESSON_CREATION_WEBHOOK_URL`,
`N8N_EXPORT_WEBHOOK_URL`, `LOCAL_OLLAMA_BASE_URL`, `INTENT_MODEL`,
`PDF_FONT_PATH`, `PDF_DIR`.

**lesson-agent** (`services/lesson_agent/.env`): `OLLAMA_API_KEY`,
`OLLAMA_CLOUD_MODEL`, `LOCAL_MODEL`, and optional `LANGFUSE_ENABLED` /
`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST`.

Missing optional credentials/flags never break other branches — the branch
detects the absence, emits a warning, and the workflow continues.

## Gateway ↔ n8n contract

Orchestrator webhook body:
```json
{ "session_id": "…", "message": "…", "want_web": false,
  "attachments": [ { "kind": "document|image", "filename": "…",
                     "mime": "…", "content_base64": "…" } ] }
```
Files uploaded **while a session is in lesson mode** become temporary lesson
sources (forwarded here as `attachments`); uploads outside lesson mode still go
to the RAG knowledge base (unchanged). URLs/YouTube links are detected from the
`message`.
