# DSEMATH AI Agent Chatbot

Multimodal HKDSE Math assistant with two runnable paths in one repository:

1. **Web app stack (React + Node/Express)** for full chat product flows.
2. **Streamlit stack (`app.py`)** for fast multimodal experimentation, OCR-first solving, and reporting/demo workflows.

Both paths use local **ChromaDB** retrieval and NVIDIA-hosted LLM APIs.

---

## What this project does

- Solves HKDSE Math Core / M1 / M2 questions.
- Supports text questions and uploaded files/images (`.pdf`, `.png`).
- Uses OCR for scanned math content, then performs grounded answering.
- Retrieves relevant chunks from a local vector database for RAG.
- Provides configurable model and retrieval behavior via `.env`.

---

## Repository structure

```text
api/                    # Node/Express backend (upload/chat/rag services)
src/                    # React + TypeScript frontend
scripts/                # TS ingestion/utility scripts for Node stack
data/                   # Local storage (Chroma persistence and seed files)
uploads/                # Uploaded files
app.py                  # Streamlit multimodal DSEMATH app
ingest_math_papers.py   # OCR + chunk + embed + store pipeline for scanned PDFs
docker-compose.yml      # Local Chroma container
python_backend/         # Optional FastAPI + LangGraph implementation
```

---

## Core architecture

### 1) Vector database

- **Engine**: ChromaDB (persistent local store)
- **Default collection (Streamlit path)**: `dsemath_math`
- **Default path (Streamlit path)**: `data/chroma_dsemath`
- **Similarity space**: cosine

### 2) Embeddings

- `SentenceTransformer` embeddings
- Default model: `BAAI/bge-small-en-v1.5`
- Fallback model candidate: `sentence-transformers/all-MiniLM-L6-v2`

### 3) LLMs

- **Text model (default)**: `meta/llama-3.1-8b-instruct`
- **Vision model (default)**: `microsoft/phi-3-vision-128k-instruct`
- API endpoint style: OpenAI-compatible chat completions via NVIDIA endpoint

### 4) Tool-calling (Streamlit path)

- `web_search`: DuckDuckGo search (`duckduckgo_search`)
- `rag_retrieve`: Chroma retrieval from local DSE corpus
- Tool loop max iterations: `MAX_TOOL_ITERATIONS` (default: 4)
- If images are uploaded, tool-calling is intentionally disabled and direct vision reasoning is used.

---

## Prompts used (Streamlit path)

### Main DSEMATH system prompt

> You are DSEMATH AI Agent. You help with HKDSE Math Core, M1, and M2. Use tools when needed: `web_search` for latest references and `rag_retrieve` for local past-paper knowledge. If a user uploaded a session file, treat it as temporary conversation context only.

### OCR transcription prompt (for uploaded image/PDF pages)

> You are a strict OCR transcriber for DSE Math exam images. Transcribe exactly. Preserve superscripts, subscripts, brackets, fraction structure, and symbols. Do not solve. If uncertain, include [unclear] markers.

---

## `ingest_math_papers.py` pipeline

This script builds your scanned-paper vector corpus.

1. Finds PDF files under input directory (default: `DSE Math`, with fallback directory resolution).
2. Renders each page to PNG (PyMuPDF).
3. Runs OCR per page:
	- preferred: Pix2Text
	- fallback: Tesseract (in `auto` mode if Pix2Text init fails)
4. Splits OCR markdown/text into chunks.
5. Embeds chunks with SentenceTransformers.
6. Writes chunks + metadata (subject/type/page/chunk/source) to Chroma.

Example:

```bash
./.venv/bin/python -u ingest_math_papers.py --input-dir "DSE Math"
```

---

## `app.py` (Streamlit DSEMATH app)

`app.py` provides:

- Chat UI (`streamlit`)
- Session file upload (`.pdf`, `.png`)
- PDF page-to-image conversion
- Dedicated OCR/transcription pass for math notation before solving
- NVIDIA text/vision chat calls
- Optional tool-calling for text-only flows (`web_search`, `rag_retrieve`)

Example run:

```bash
./.venv/bin/python -m streamlit run app.py --server.port 8504
```

---

## Local setup (recommended)

### 1) Prerequisites

- Node.js 20+
- Python 3.10+
- Docker Desktop (optional but recommended for Chroma)
- `tesseract` CLI installed (recommended OCR fallback)

### 2) Environment file

```bash
cp .env.example .env
```

Set at minimum:

- `NVIDIA_API_KEY`
- `NVIDIA_BASE_URL` (default already set)
- `NVIDIA_MODEL` / `NVIDIA_MATH_MODEL` / `NVIDIA_VLM_MODEL` as needed
- `CHROMA_URL` (Node stack)

### 3) Start ChromaDB

```bash
docker compose up -d
```

Default container mapping from this repo:

- host `8001` -> container `8000`

### 4) Install dependencies

Node stack:

```bash
npm install
```

Python stack:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5) Run the app(s)

Node web stack:

```bash
npm run dev
```

Streamlit stack:

```bash
./.venv/bin/python -m streamlit run app.py --server.port 8504
```

---

## Useful commands

```bash
# Node frontend + backend
npm run dev

# Type check
npm run check

# Lint
npm run lint

# Seed local markdown into Node RAG
npm run seed

# Ingest scanned math papers to Chroma (Python OCR pipeline)
./.venv/bin/python -u ingest_math_papers.py --input-dir "DSE Math"
```

---

## Troubleshooting

### Wrong Python interpreter (common)

Always run Streamlit with the project venv interpreter:

```bash
./.venv/bin/python -m streamlit run app.py --server.port 8504
```

### Port already in use

If a port is busy, switch to another (`8504`, `8505`, etc.).

### OCR quality issues for scanned math

- Use higher-resolution scans.
- Keep equations centered and avoid heavy compression artifacts.
- Re-run ingestion with Pix2Text available.
- Ensure Tesseract is installed for fallback mode.

---

## Notes

- `app.py` and `ingest_math_papers.py` are optimized for DSE math OCR + retrieval workflows.
- `python_backend/` contains a separate FastAPI + LangGraph implementation if you want API-first orchestration.
