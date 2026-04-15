# Python Rebuild (LangGraph + LangChain)

This is a Python backend rebuild for your math tutor chatbot using modern stack:
- LangGraph (workflow orchestration)
- LangChain (LLM + retrieval)
- FastAPI (API layer)
- Chroma (vector store)

## Endpoints
- `GET /api/health`
- `POST /api/chat`
- `POST /api/debug/chat` (returns per-agent outputs + reranker/debug metadata)
- `POST /api/upload`
- `POST /api/eval/run` (run DSE paper evaluation batch)
- `POST /api/eval/run-async` (start non-blocking eval job)
- `GET /api/eval/job/{job_id}` (poll progress)
- `GET /api/eval/summary` (load latest dashboard report)
- `GET /api/eval/report.csv?type=papers|failures` (download CSV reports)

### DSE Evaluation (Core / M1 / M2)
Expected folder structure:

```text
<DSE root>/
	core/
		pastpaper/
		answer/
	m1/
		pastpaper/
		answer/
	m2/
		pastpaper/
		answer/
```

The eval runner solves from `pastpaper` only. Answer files are never fed into generation prompts.
If `include_answer_scoring=true`, answer files are used only after generation for deterministic overlap scoring.

Failure mining outputs:
- `data/eval/dse_failure_mining_latest.jsonl`
- timestamped JSONL per run
- auto split with balanced category sampling:
	- `data/eval/dse_failure_mining_train_latest.jsonl`
	- `data/eval/dse_failure_mining_val_latest.jsonl`
	- `data/eval/dse_failure_mining_test_latest.jsonl`

These JSONL rows are ready as preference-style fine-tuning seeds (`instruction`, `rejected`, `chosen`, `metadata`).

Scanned PDF handling:
- Eval API supports `use_vlm_ocr_for_scanned_pdf=true` and `ocr_max_pages`.
- If native PDF text extraction is too short, pages are rendered as images and passed through NVIDIA VLM OCR before solving/scoring.
- This applies to both `pastpaper` and `answer` extraction, while preserving no-leak policy in generation.

## Setup

### 1) Create environment
```bash
cd python_backend
python -m venv .venv
source .venv/bin/activate
```

### 2) Install deps
```bash
pip install -U pip
pip install -e .
```

### 3) Configure env
Reuses root `.env` variables:
- `NVIDIA_API_KEY`
- `NVIDIA_BASE_URL`
- `NVIDIA_MODEL`
- Optional per-agent models:
	- `NVIDIA_MODEL_MANAGER`
	- `NVIDIA_MODEL_NEWTON`
	- `NVIDIA_MODEL_GAUSS`
	- `NVIDIA_MODEL_FEYNMAN`
	- `NVIDIA_MODEL_POLYA`
	- `NVIDIA_MODEL_GRIFFITHS`
	- `NVIDIA_MODEL_SAKURAI`
- `NVIDIA_VLM_MODEL` (for image QA)
- `ENABLE_VLM_PREPROCESS_FOR_AGENTS=true` (VLM extracts image text, then LLM agents solve)
- `ENABLE_TUTOR_MODE`
- `ENABLE_HYBRID_RETRIEVAL`
- `ENABLE_MATH_RERANKER`

### 4) Run
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

## LangGraph Flow
1. `classify` (intent + tutor prefix)
2. `retrieve` (hybrid retrieval + rerank)
3. `generate` (LLM or VLM if image)
4. `validate` (basic final-answer guard)

## Phase 3 Upgrade Included
- True cross-encoder reranking service (SentenceTransformers `CrossEncoder`).
- Domain metadata filtering at ingestion time (`domain`, `tags`, `chunk_index`).
- Retrieval can filter by inferred domain and fallback automatically if too strict.
- Multi-agent tutor mode (Newton/Gauss/Feynman experts + Manager synthesis).

Config knobs:
- `ENABLE_CROSS_ENCODER_RERANKER=true`
- `CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`
- `ENABLE_DOMAIN_METADATA_FILTER=true`
- `ENABLE_MULTI_AGENT_TUTOR=true`

You can point your frontend `VITE_API_URL` to this backend once ready.

## HKDSE 2012 PDF Inference Diagnostics (Root Cause + Fixes)

### Root causes identified
- Upload context was global-only and short-lived, so follow-up questions could lose document grounding.
- Upload text was truncated (`text[:20000]`), dropping later question regions and diagram references.
- Question navigation for Q1–Q3 relied on loose regex over unstructured text, causing boundary misses.
- There was no deterministic document attribution (fingerprint/page checksum), so provenance could not be verified.

### Implemented fixes
- Session-scoped persistent upload context keyed by `session_id`/`conversation_id` with on-disk state.
- Dual-path PDF processing: direct extraction from uploaded document + vector retrieval ingestion.
- HKDSE question segmentation for Questions 1–3 with stored question/page mapping and `GET /api/upload/questions`.
- Verification metadata: SHA-256 PDF fingerprint, per-page checksums, response attribution in debug/answers.
- OCR/scan issue detection flags for low text density, symbol corruption, and unstable complex notation.

### Reproducible diagnostics
1. Upload HKDSE Core Math 2012 PDF via `/api/upload` with `session_id=<chat_id>`.
2. Call `/api/upload/questions?session_id=<chat_id>` and verify Q1–Q3 extraction.
3. Ask `/api/debug/chat` with `conversation_id=<chat_id>` and inspect:
   - `upload_context.fingerprint`
   - `upload_context.page_checksums`
   - `retrieval_debug.attribution`
4. Validate persistence by restarting chat turns without re-upload in the same session id.

### Test suite added
- Unit: math OCR symbol normalization.
- Integration: question boundary detection for Q1–Q3.
- End-to-end style: marking-scheme alignment score check for HKDSE-style solution steps.

Run:
```bash
cd python_backend
python -m unittest discover -s tests -v
```
