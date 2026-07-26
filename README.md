# הַמֹּורִי — Educational Assistant (Microservices)

A Hebrew RTL educational lesson-plan assistant, rebuilt as independent
microservices. Retrieval-augmented generation runs **fully locally** (LangChain
+ ChromaDB + HuggingFace embeddings + llama.cpp / a local GGUF model), replacing
the previous AWS Bedrock / S3 / SES stack.

> **Status:** architecture + skeletons. Reusable code (frontend, SQLite chat
> history, conversation-memory shortcuts, file storage, document registry,
> download) is migrated and functional. Core AI logic (retrieval, generation,
> guardrails, image analysis) and the n8n workflows are **not implemented yet**.

---

## Architecture

```
Frontend (Hebrew RTL SPA, served by the Gateway)
        |
        v
FastAPI Gateway  ── SQLite (sessions + chat history)
        |
        ├──────────────┬───────────────────┬───────────────────┐
        v              v                   v                   v
      RAG            n8n            Image Analyzer          Guardrails
   (FastAPI)     (workflows)          (FastAPI)             (FastAPI)
        |
        └── LangGraph (when needed)
        |
        ├── ChromaDB (vector store)
        ├── HuggingFace embeddings
        └── llama.cpp / local GGUF model (Qwen3-14B-Instruct)
```

- **Gateway** — serves the frontend on the same origin, owns SQLite (sessions +
  chat history), exposes the public API, proxies documents/chat to RAG, and
  triggers n8n workflows.
- **RAG** — owns uploaded source files, the document metadata registry,
  ChromaDB, and the retrieval + generation pipeline (LangGraph when needed).
- **Image Analyzer** — analyzes uploaded images (net-new capability).
- **Guardrails** — content-safety checks on input/output (replaces Bedrock
  Guardrails).
- **n8n** — orchestration for two workflows: **Lesson Creation** and **Export
  and Email** (PDF + email; replaces the old Lambda/SES Action Groups).

Inter-service communication is synchronous **HTTP/JSON** (multipart for
uploads, binary for downloads). All service URLs come from environment
variables. No message broker or event bus.

---

## Repository layout

```
frontend/templates/        index.html (SPA) + logout.html — served by the Gateway
services/
  gateway/                 FastAPI gateway (frontend host, SQLite, API, proxy)
    app/                   config, database, memory, schemas, clients/, routers/
    requirements.txt  Dockerfile  .env.example
  rag/                     RAG service
    app/                   storage/ (files, registry), vectorstore/ (chroma),
                           embeddings/ (hf), llm/ (llama), langgraph/, routers/
    requirements.txt  Dockerfile  .env.example
  image_analyzer/          Image analysis service (skeleton)
  guardrails/              Content-safety service (skeleton)
n8n/workflows/             Placeholder READMEs + (later) exported workflow JSON
screenshots/  submissions/ Course deliverables
```

---

## Local development

Each service is a standalone FastAPI app. Create a virtualenv per service (or a
shared one), install its `requirements.txt`, copy its `.env.example` to `.env`,
and run it:

```bash
# RAG service (needs a local GGUF model on disk)
cd services/rag
pip install -r requirements.txt
cp .env.example .env            # set LLM_MODEL_PATH, EMBEDDING_MODEL, ...
python -m app.main              # http://localhost:8001

# Gateway (run from the repo root so it finds ./frontend)
cd ../..
pip install -r services/gateway/requirements.txt
cp services/gateway/.env.example services/gateway/.env
python -m app.main              # from within services/gateway, or:
uvicorn app.main:app --app-dir services/gateway --host 0.0.0.0 --port 8000
```

Open the app at **http://localhost:8000/**.

Start the Image Analyzer (8002) and Guardrails (8003) the same way when needed.

---

## Docker (no Docker Compose)

Each service has its own Dockerfile and runs independently with `docker run`.
Images are built from the **repository root** so the Gateway can include the
frontend.

```bash
# Gateway (serves the frontend)
docker build -f services/gateway/Dockerfile -t homori-gateway .
docker run --rm -p 8000:8000 --env-file services/gateway/.env homori-gateway

# RAG (mount the GGUF model and a data volume)
docker build -f services/rag/Dockerfile -t homori-rag .
docker run --rm -p 8001:8001 --env-file services/rag/.env \
  -v /path/to/models:/models:ro -v homori-rag-data:/app/data homori-rag

# Image Analyzer
docker build -f services/image_analyzer/Dockerfile -t homori-image-analyzer .
docker run --rm -p 8002:8002 --env-file services/image_analyzer/.env homori-image-analyzer

# Guardrails
docker build -f services/guardrails/Dockerfile -t homori-guardrails .
docker run --rm -p 8003:8003 --env-file services/guardrails/.env homori-guardrails
```

n8n runs from its official image; see `n8n/workflows/README.md`.

---

## Public API (served by the Gateway)

The frontend's original relative paths are preserved (the only field rename is
`s3_key` → `doc_id`):

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Chat UI |
| GET | `/logout` | Goodbye page |
| POST | `/chat` | One conversational turn → `{answer, context:[{source, doc_id}]}` |
| GET | `/history` | Session message history |
| POST | `/clear` | Clear the session's messages |
| GET | `/status` | RAG readiness / ingestion status |
| GET | `/documents` | List documents |
| POST | `/upload` | Upload files (multipart) |
| DELETE | `/documents` | Delete a document `{doc_id}` |
| GET | `/documents/download?doc_id=` | Download a document |
| GET | `/health` | Per-service health check |

---

## Migration notes

- **Removed:** AWS Bedrock (Agent + Knowledge Base), S3, SES, Lambda code,
  Flask, the unused legacy `static/` assets, and all dead code.
- **Preserved:** every user-facing feature from the previous project (chat +
  Markdown rendering, per-session history, source citations, upload / list /
  delete / download with Hebrew filenames, session isolation, status polling,
  toasts, modals, rename/search, Hebrew memory shortcuts, error handling).
- **To recover from the old AWS setup before go-live:** the Bedrock Agent
  instructions (Hebrew lesson-plan rules → local LLM system prompt), the
  Guardrails policies, and the PDF/email Lambda logic (→ n8n Export workflow).
