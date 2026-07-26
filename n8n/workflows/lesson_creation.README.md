# Lesson Creation Workflow (placeholder)

**Status:** not implemented. Export `lesson_creation.json` here once built.

## Purpose
Create / assemble a lesson plan (the "generate a new lesson" capability),
kept separate from retrieval-only RAG chat.

## Proposed contract (subject to the architecture spec)

**Trigger:** Webhook (POST). URL injected into the Gateway as
`N8N_LESSON_CREATION_WEBHOOK_URL`.

**Input (JSON):**
```json
{
  "session_id": "string",
  "topic": "string",
  "requirements": "string",
  "source_doc_ids": ["string"]
}
```

**Output (JSON):**
```json
{
  "lesson_title": "string",
  "content": "string"
}
```

> Business logic (prompting, retrieval calls, formatting) is defined inside the
> n8n workflow, not in this repo.
