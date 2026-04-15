from fastapi import FastAPI, UploadFile, File, Query, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import base64
import httpx
import re

from app.models import ChatRequest, ChatResponse, HealthResponse, EvalRunRequest
from app.graph import rag_graph
from app.services.vectorstore import add_documents
from app.services.eval_service import (
    run_dse_evaluation,
    read_latest_evaluation,
    latest_evaluation_csv,
    start_eval_job,
    get_eval_job,
)
from app.services.upload_context import (
    get_latest_image,
    get_latest_text,
    set_latest_image,
    set_latest_text,
    clear_latest_image,
    clear_latest_text,
)
from app.services.clarify_memory import (
    peek_pending_clarification,
    set_pending_clarification,
    clear_pending_clarification,
)
from app.services.pdf_math_parser import (
    compute_page_checksums,
    compute_pdf_fingerprint,
    detect_scan_issues,
    normalize_math_ocr_text,
    segment_hkdse_questions,
)


app = FastAPI(title="Math RAG Backend (LangGraph)")

splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180)

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover
    fitz = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(success=True, message="ok")


def _invoke_vlm_ocr(image_data_url: str, prompt: str) -> str:
    from app import config

    if not (config.NVIDIA_API_KEY and config.NVIDIA_VLM_MODEL):
        return ""

    endpoint = f"{config.NVIDIA_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": config.NVIDIA_VLM_MODEL,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
    }

    try:
        with httpx.Client(timeout=45) as client:
            res = client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {config.NVIDIA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if res.status_code >= 400:
                return ""
            data = res.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content if isinstance(content, str) else str(content)
    except Exception:
        return ""


def _extract_pdf_text_via_vlm(pdf_bytes: bytes, max_pages: int = 3) -> str:
    if fitz is None:
        return ""

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return ""

    parts: list[str] = []
    prompt = (
        "You are OCR assistant for math exams. Extract readable text, equations, symbols, and question statements from this page. "
        "Do NOT solve. Preserve math expressions and numbering."
    )

    try:
        for i in range(min(len(doc), max(1, max_pages))):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            png_bytes = pix.tobytes("png")
            data_url = f"data:image/png;base64,{base64.b64encode(png_bytes).decode()}"
            ocr_text = _invoke_vlm_ocr(data_url, prompt)
            if ocr_text.strip():
                parts.append(f"[Page {i + 1}]\n{ocr_text.strip()}")
    finally:
        doc.close()

    return "\n\n".join(parts)


def _resolve_session_id(raw_id: str | None) -> str:
    sid = (raw_id or "").strip()
    return sid if sid else "default"


def _extract_pdf_pages_text(pdf_bytes: bytes) -> list[str]:
    from io import BytesIO

    reader = PdfReader(BytesIO(pdf_bytes))
    return [normalize_math_ocr_text(p.extract_text() or "") for p in reader.pages]


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = _resolve_session_id(req.conversation_id)
    latest_image = get_latest_image(session_id=session_id)
    latest_text = get_latest_text(session_id=session_id)
    msg_lower = req.message.lower()
    use_image = latest_image and any(x in msg_lower for x in ["image", "png", "uploaded", "screenshot"])
    use_uploaded_text = bool(latest_text)

    pending = peek_pending_clarification()
    clarification_memory_used = False
    effective_question = req.message
    if pending and req.message.strip():
        clarification_memory_used = True
        effective_question = (
            f"Original unresolved question:\n{pending.question}\n\n"
            f"User clarification:\n{req.message}"
        )

    state = {
        "question": effective_question,
        "image_data_url": latest_image.data_url if use_image else "",
        "uploaded_text": latest_text.content if use_uploaded_text else "",
        "uploaded_source": latest_text.source if use_uploaded_text else "",
        "uploaded_question_blocks": latest_text.question_blocks if use_uploaded_text else {},
        "uploaded_question_pages": latest_text.question_pages if use_uploaded_text else {},
        "upload_fingerprint": latest_text.fingerprint if use_uploaded_text else "",
        "upload_page_checksums": latest_text.page_checksums if use_uploaded_text else {},
        "upload_scan_issues": latest_text.scan_issues if use_uploaded_text else [],
    }
    out = rag_graph.invoke(state)

    ambiguity = out.get("ambiguity_debug", {}) or {}
    if ambiguity.get("needs_clarification"):
        set_pending_clarification(effective_question, ambiguity)
    elif clarification_memory_used:
        clear_pending_clarification()

    return ChatResponse(
        success=True,
        content=out.get("answer", ""),
        sources=out.get("sources", []),
    )


@app.post("/api/debug/chat")
def debug_chat(req: ChatRequest):
    session_id = _resolve_session_id(req.conversation_id)
    latest_image = get_latest_image(session_id=session_id)
    latest_text = get_latest_text(session_id=session_id)
    msg_lower = req.message.lower()
    use_image = latest_image and any(x in msg_lower for x in ["image", "png", "uploaded", "screenshot"])
    use_uploaded_text = bool(latest_text)

    pending = peek_pending_clarification()
    clarification_memory_used = False
    effective_question = req.message
    if pending and req.message.strip():
        clarification_memory_used = True
        effective_question = (
            f"Original unresolved question:\n{pending.question}\n\n"
            f"User clarification:\n{req.message}"
        )

    state = {
        "question": effective_question,
        "image_data_url": latest_image.data_url if use_image else "",
        "uploaded_text": latest_text.content if use_uploaded_text else "",
        "uploaded_source": latest_text.source if use_uploaded_text else "",
        "uploaded_question_blocks": latest_text.question_blocks if use_uploaded_text else {},
        "uploaded_question_pages": latest_text.question_pages if use_uploaded_text else {},
        "upload_fingerprint": latest_text.fingerprint if use_uploaded_text else "",
        "upload_page_checksums": latest_text.page_checksums if use_uploaded_text else {},
        "upload_scan_issues": latest_text.scan_issues if use_uploaded_text else [],
    }
    out = rag_graph.invoke(state)

    upload_context = None
    if latest_text:
        upload_context = {
            "source": latest_text.source,
            "content_chars": len(latest_text.content or ""),
            "preview": (latest_text.content or "")[:300],
            "uploaded_at": latest_text.uploaded_at,
            "session_id": latest_text.session_id,
            "fingerprint": latest_text.fingerprint,
            "page_checksums": latest_text.page_checksums,
            "question_blocks": sorted((latest_text.question_blocks or {}).keys()),
            "question_pages": latest_text.question_pages,
            "scan_issues": latest_text.scan_issues,
            "active": bool(use_uploaded_text),
        }
    elif latest_image:
        upload_context = {
            "source": latest_image.source,
            "mime_type": latest_image.mime_type,
            "uploaded_at": latest_image.uploaded_at,
            "active": bool(use_image),
        }

    ambiguity = out.get("ambiguity_debug", {}) or {}
    if ambiguity.get("needs_clarification"):
        set_pending_clarification(effective_question, ambiguity)
    elif clarification_memory_used:
        clear_pending_clarification()

    return {
        "success": True,
        "answer": out.get("answer", ""),
        "sources": out.get("sources", []),
        "agent_outputs": out.get("agent_outputs", {}),
        "retrieval_debug": out.get("retrieval_debug", {}),
        "strict_validation": out.get("strict_validation", {}),
        "ambiguity_debug": out.get("ambiguity_debug", {}),
        "upload_context": upload_context,
        "clarification_memory_used": clarification_memory_used,
        "effective_question": effective_question,
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), session_id: str = Form(default="default")):
    session_id = _resolve_session_id(session_id)
    name = file.filename or "uploaded_file"
    mime = file.content_type or "application/octet-stream"
    data = await file.read()

    # Replace prior upload in the same session only.
    clear_latest_text(session_id=session_id)
    clear_latest_image(session_id=session_id)

    if mime in ("image/png", "image/x-png"):
        data_url = f"data:{mime};base64,{base64.b64encode(data).decode()}"
        set_latest_image(name, mime, data_url, session_id=session_id)
        return {
            "success": True,
            "message": "PNG uploaded for direct vision-model question answering",
            "chunks_count": 0,
            "session_id": session_id,
        }

    text = ""
    page_texts: list[str] = []
    extraction_method = "native"
    if mime in ("text/plain", "text/markdown"):
        text = normalize_math_ocr_text(data.decode("utf-8", errors="ignore"))
        page_texts = [text]
    elif mime == "application/pdf":
        page_texts = _extract_pdf_pages_text(data)
        text = normalize_math_ocr_text("\n".join(page_texts))
        if len(text.strip()) < 24:
            ocr_text = _extract_pdf_text_via_vlm(data, max_pages=3)
            if ocr_text.strip():
                text = normalize_math_ocr_text(ocr_text)
                page_texts = [text]
                extraction_method = "vlm_ocr"
    else:
        return {"success": False, "error": "Unsupported file format"}

    if not text.strip():
        return {
            "success": False,
            "error": "No readable text found in file. If this is a scanned PDF, enable NVIDIA_VLM_MODEL and NVIDIA_API_KEY for OCR fallback, or upload a clearer PNG screenshot.",
        }

    def infer_domain_tags(chunk: str) -> tuple[str, str]:
        c = chunk.lower()
        tags: list[str] = []

        if re.search(r"\b(matrix|determinant|eigen|rank|nullspace|vector space)\b", c):
            tags.append("linear_algebra")
        if re.search(r"\b(derivative|integral|limit|chain rule|partial derivative)\b", c):
            tags.append("calculus")
        if re.search(r"\b(grad|gradient|divergence|curl|nabla|line integral|surface integral)\b", c):
            tags.append("vector_calculus")
        if re.search(r"\b(proof|lemma|theorem|corollary|show that)\b", c):
            tags.append("proof")
        if re.search(r"\b(equation|quadratic|factor|polynomial|inequality)\b", c):
            tags.append("algebra")

        domain = tags[0] if tags else "general"
        unique_tags = ",".join(dict.fromkeys(tags)) if tags else "general"
        return domain, unique_tags

    base_doc = Document(page_content=text, metadata={"source": name, "file_type": mime})
    docs = splitter.split_documents([base_doc])

    enriched_docs = []
    for idx, d in enumerate(docs):
        domain, tags = infer_domain_tags(d.page_content or "")
        enriched_docs.append(
            Document(
                page_content=d.page_content,
                metadata={
                    **(d.metadata or {}),
                    "source": name,
                    "file_type": mime,
                    "domain": domain,
                    "tags": tags,
                    "chunk_index": idx,
                },
            )
        )

    add_documents(enriched_docs)
    fingerprint = compute_pdf_fingerprint(data) if mime == "application/pdf" else ""
    page_checksums = compute_page_checksums(page_texts)
    question_blocks = segment_hkdse_questions(text, page_texts, question_numbers=(1, 2, 3))
    question_block_map = {str(k): v.content for k, v in question_blocks.items()}
    question_page_map = {str(k): v.pages for k, v in question_blocks.items()}
    scan_issues = detect_scan_issues(page_texts)

    set_latest_text(
        name,
        text,
        session_id=session_id,
        fingerprint=fingerprint,
        page_checksums=page_checksums,
        question_blocks=question_block_map,
        question_pages=question_page_map,
        extraction_method=extraction_method,
        scan_issues=scan_issues,
    )

    return {
        "success": True,
        "message": "Document processed and ingested successfully",
        "chunks_count": len(enriched_docs),
        "preview": text[:220],
        "session_id": session_id,
        "fingerprint": fingerprint,
        "page_checksums": page_checksums,
        "question_blocks": sorted(question_block_map.keys()),
        "scan_issues": scan_issues,
    }


@app.get("/api/upload/questions")
def uploaded_questions(session_id: str = Query(default="default")):
    resolved = _resolve_session_id(session_id)
    latest_text = get_latest_text(session_id=resolved)
    if not latest_text:
        return {"success": False, "error": "No uploaded document found for session", "session_id": resolved}
    return {
        "success": True,
        "session_id": resolved,
        "source": latest_text.source,
        "fingerprint": latest_text.fingerprint,
        "questions": [
            {
                "question_no": int(q_no),
                "content": content,
                "pages": (latest_text.question_pages or {}).get(q_no, []),
            }
            for q_no, content in sorted((latest_text.question_blocks or {}).items(), key=lambda x: int(x[0]))
            if q_no in {"1", "2", "3"}
        ],
    }


@app.post("/api/eval/run")
def run_eval(req: EvalRunRequest):
    return run_dse_evaluation(
        dse_root=req.dse_root,
        include_answer_scoring=req.include_answer_scoring,
        max_papers=req.max_papers,
        use_vlm_ocr_for_scanned_pdf=req.use_vlm_ocr_for_scanned_pdf,
        ocr_max_pages=req.ocr_max_pages,
        split_train_ratio=req.split_train_ratio,
        split_val_ratio=req.split_val_ratio,
        split_test_ratio=req.split_test_ratio,
    )


@app.post("/api/eval/run-async")
def run_eval_async(req: EvalRunRequest):
    return start_eval_job(
        dse_root=req.dse_root,
        include_answer_scoring=req.include_answer_scoring,
        max_papers=req.max_papers,
        use_vlm_ocr_for_scanned_pdf=req.use_vlm_ocr_for_scanned_pdf,
        ocr_max_pages=req.ocr_max_pages,
        split_train_ratio=req.split_train_ratio,
        split_val_ratio=req.split_val_ratio,
        split_test_ratio=req.split_test_ratio,
    )


@app.get("/api/eval/job/{job_id}")
def eval_job_status(job_id: str):
    return get_eval_job(job_id)


@app.get("/api/eval/summary")
def eval_summary():
    return read_latest_evaluation()


@app.get("/api/eval/report.csv")
def eval_report_csv(report_type: str = Query(default="papers", alias="type")):
    csv_payload = latest_evaluation_csv(report_type=report_type)
    normalized = "failures" if (report_type or "").lower() == "failures" else "papers"
    filename = f"dse_eval_{normalized}.csv"
    return Response(
        content=csv_payload,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
