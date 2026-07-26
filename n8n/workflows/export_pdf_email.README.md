# Export and Email Workflow (placeholder)

**Status:** not implemented. Export `export_pdf_email.json` here once built.

## Purpose
Render a lesson plan to **PDF** and **email** a download link. This replaces
the previous project's AWS Lambda `create_pdf` + `send_email` (SES) Action
Groups. PDF/email are intentionally **not** part of the RAG service.

## Proposed contract (subject to the architecture spec)

**Trigger:** Webhook (POST). URL injected into the Gateway as
`N8N_EXPORT_WEBHOOK_URL`.

**Input (JSON):**
```json
{
  "lesson_title": "string",
  "content": "string",
  "recipient_email": "string"
}
```

**Output (JSON):**
```json
{
  "download_url": "string",
  "emailed": true
}
```

## Preserved frontend behaviour
When a chat answer contains a PDF download link, the frontend already rewrites
it to the label "מערך מעודכן להורדה📄" (see `frontend/templates/index.html`,
`appendAssistantBubble`). Keep the returned link on a path the UI recognises
(contains `/documents/download` or ends in `.pdf`).

> Business logic (PDF rendering, SMTP/email provider) is defined inside the
> n8n workflow, not in this repo.
