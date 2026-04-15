from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import base64
import re

from app.models import ChatRequest, ChatResponse, HealthResponse, EvalRunRequest
from app.graph import rag_graph
from app.services.vectorstore import add_documents
from app.services.eval_service import run_dse_evaluation, read_latest_evaluation, latest_evaluation_csv
from app.services.upload_context import (
    get_latest_image,
    set_latest_image,
    set_latest_text,
)
from app.services.clarify_memory import (
    peek_pending_clarification,
    set_pending_clarification,
    clear_pending_clarification,
)


app = FastAPI(title="Math RAG Backend (LangGraph)")

splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180)

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


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    latest_image = get_latest_image()
    use_image = latest_image and any(x in req.message.lower() for x in ["image", "png", "uploaded"])

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
    latest_image = get_latest_image()
    use_image = latest_image and any(x in req.message.lower() for x in ["image", "png", "uploaded"])

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
    }
    out = rag_graph.invoke(state)

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
        "clarification_memory_used": clarification_memory_used,
        "effective_question": effective_question,
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = file.filename or "uploaded_file"
    mime = file.content_type or "application/octet-stream"
    data = await file.read()

    if mime in ("image/png", "image/x-png"):
        data_url = f"data:{mime};base64,{base64.b64encode(data).decode()}"
        set_latest_image(name, mime, data_url)
        return {
            "success": True,
            "message": "PNG uploaded for direct vision-model question answering",
            "chunks_count": 0,
        }

    text = ""
    if mime in ("text/plain", "text/markdown"):
        text = data.decode("utf-8", errors="ignore")
    elif mime == "application/pdf":
        from io import BytesIO

        reader = PdfReader(BytesIO(data))
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
    else:
        return {"success": False, "error": "Unsupported file format"}

    if not text.strip():
        return {"success": False, "error": "No readable text found in file"}

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
    set_latest_text(name, text[:4000])

    return {
        "success": True,
        "message": "Document processed and ingested successfully",
        "chunks_count": len(enriched_docs),
        "preview": text[:220],
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
