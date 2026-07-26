# n8n Workflows

Orchestration and integrations that are **not** part of the RAG service live
here as n8n workflows. n8n runs as its own application (its official Docker
image); it is **not** a FastAPI service and has no Dockerfile in this repo.

Exported workflow JSON files will be committed to this folder once the
workflows are built in the n8n editor. Business logic is **not** implemented
yet — only the contracts below are defined.

## Planned workflows

| Workflow | File (to be added) | Triggered by | Purpose |
|---|---|---|---|
| Lesson Creation | `lesson_creation.json` | Gateway webhook | Generate/assemble a lesson plan |
| Export and Email | `export_pdf_email.json` | Gateway webhook | Render a PDF and email a download link |

See the per-workflow README files for their input/output contracts.

## How the Gateway triggers workflows

The Gateway calls n8n **webhook URLs** injected via environment variables
(`N8N_LESSON_CREATION_WEBHOOK_URL`, `N8N_EXPORT_WEBHOOK_URL`) — see
`services/gateway/app/clients/n8n_client.py`. Communication is synchronous
HTTP/JSON. No message broker or event bus is used.

## Running n8n (reference)

```bash
docker run --rm -p 5678:5678 -v homori-n8n:/home/node/.n8n n8nio/n8n
```

Import the JSON files from this folder in the n8n editor, then copy each
workflow's Production webhook URL into the Gateway environment.
