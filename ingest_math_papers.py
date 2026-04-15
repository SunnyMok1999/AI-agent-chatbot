#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import fitz
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pix2text import Pix2Text
from sentence_transformers import SentenceTransformer


@dataclass
class IngestConfig:
    input_dir: Path
    chroma_dir: Path
    collection_name: str
    embedding_model: str
    dpi: int
    chunk_size: int
    chunk_overlap: int
    ocr_backend: str


class SentenceTransformerEmbedding:
    def __init__(self, model_name: str) -> None:
        self._name = model_name
        self.model = SentenceTransformer(model_name)

    def name(self) -> str:
        return self._name

    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors = self.model.encode(input, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self(texts)

    def embed_query(self, input: str | list[str]) -> list[list[float]]:
        if isinstance(input, list):
            query = input[0] if input else ""
        else:
            query = input
        return self([query])


def parse_args() -> IngestConfig:
    parser = argparse.ArgumentParser(description="Ingest scanned DSE Math PDFs into ChromaDB")
    parser.add_argument("--input-dir", default="DSE Math", help="Folder containing Core/M1/M2 PDF files")
    parser.add_argument("--chroma-dir", default="data/chroma_dsemath", help="Persistent Chroma directory")
    parser.add_argument("--collection", default="dsemath_math", help="Chroma collection name")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5", help="SentenceTransformer model")
    parser.add_argument("--dpi", type=int, default=300, help="PDF render DPI for OCR")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap")
    parser.add_argument(
        "--ocr-backend",
        choices=["auto", "pix2text", "tesseract"],
        default="auto",
        help="OCR engine to use (auto falls back to tesseract if pix2text fails)",
    )
    args = parser.parse_args()
    return IngestConfig(
        input_dir=Path(args.input_dir).resolve(),
        chroma_dir=Path(args.chroma_dir).resolve(),
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        dpi=args.dpi,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        ocr_backend=args.ocr_backend,
    )


def resolve_input_dir(cfg: IngestConfig) -> Path:
    candidates = [cfg.input_dir]
    cwd = Path.cwd()
    candidates.extend([
        cwd / "DSE Math",
        cwd / "math",
        cwd / "data" / "seed",
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return cfg.input_dir


def infer_metadata(pdf_path: Path, root: Path) -> dict[str, str]:
    rel_parts = [p.lower() for p in pdf_path.relative_to(root).parts]
    subject = "core"
    doc_type = "unknown"
    type_mapping = {"pastpaper": "pastpaper", "answer": "answer_key", "solutions": "answer_key"}
    for token in rel_parts:
        if token in {"core", "m1", "m2"}:
            subject = token
        if token in type_mapping:
            doc_type = type_mapping[token]
    return {
        "subject": subject,
        "type": doc_type,
        "source": str(pdf_path),
        "filename": pdf_path.name,
    }


def render_pdf_to_images(pdf_path: Path, dpi: int) -> list[bytes]:
    images: list[bytes] = []
    scale = dpi / 72
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            images.append(pix.tobytes("png"))
    return images


def _ocr_result_to_markdown(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("md", "markdown", "text", "result"):
            value = result.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(result, ensure_ascii=False)
    if isinstance(result, list):
        return "\n".join(_ocr_result_to_markdown(item) for item in result)
    return str(result)


def run_math_ocr_pix2text(ocr: Pix2Text, image_png: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        tmp.write(image_png)
        tmp.flush()
        # Pix2Text APIs differ across versions, so we support both common entry points.
        if hasattr(ocr, "recognize_markdown"):
            result = ocr.recognize_markdown(tmp.name)
        else:
            result = ocr.recognize(tmp.name)
    return _ocr_result_to_markdown(result)


def run_math_ocr_tesseract(image_png: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        tmp.write(image_png)
        tmp.flush()
        cmd = ["tesseract", tmp.name, "stdout", "-l", "eng", "--psm", "6"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise RuntimeError(f"Tesseract OCR failed: {stderr or 'unknown error'}")
        return proc.stdout or ""


def run_math_ocr(ocr_backend: str, ocr_engine: Any, image_png: bytes) -> str:
    if ocr_backend == "pix2text":
        return run_math_ocr_pix2text(ocr_engine, image_png)
    if ocr_backend == "tesseract":
        return run_math_ocr_tesseract(image_png)
    raise ValueError(f"Unsupported OCR backend: {ocr_backend}")


def chunk_markdown(markdown_text: str, cfg: IngestConfig) -> list[str]:
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")])
    docs = markdown_splitter.split_text(markdown_text)
    recursive = RecursiveCharacterTextSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
    chunks: list[str] = []
    if docs:
        for doc in docs:
            chunks.extend(recursive.split_text(doc.page_content))
    else:
        chunks.extend(recursive.split_text(markdown_text))
    return [c.strip() for c in chunks if c.strip()]


def ingest_pdf(pdf_path: Path, cfg: IngestConfig, ocr_backend: str, ocr_engine: Any, collection: Any) -> int:
    meta_base = infer_metadata(pdf_path, cfg.input_dir)
    page_pngs = render_pdf_to_images(pdf_path, cfg.dpi)

    total_chunks = 0
    for page_idx, png_data in enumerate(page_pngs, start=1):
        markdown = run_math_ocr(ocr_backend, ocr_engine, png_data)
        chunks = chunk_markdown(markdown, cfg)
        if not chunks:
            continue

        ids = []
        docs = []
        metadatas = []
        for chunk_idx, chunk in enumerate(chunks):
            ids.append(f"{pdf_path.stem}-p{page_idx}-c{chunk_idx}")
            docs.append(chunk)
            metadatas.append({
                **meta_base,
                "page": page_idx,
                "chunk_index": chunk_idx,
            })

        collection.add(ids=ids, documents=docs, metadatas=metadatas)
        total_chunks += len(chunks)

    return total_chunks


def main() -> None:
    cfg = parse_args()
    cfg.input_dir = resolve_input_dir(cfg)
    if not cfg.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {cfg.input_dir}")

    cfg.chroma_dir.mkdir(parents=True, exist_ok=True)
    embedding_fn = SentenceTransformerEmbedding(cfg.embedding_model)
    client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
    collection = client.get_or_create_collection(
        name=cfg.collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ocr_backend = cfg.ocr_backend
    ocr_engine: Any = None

    if ocr_backend in {"auto", "pix2text"}:
        try:
            # "mfd" = math formula detection; this improves OCR quality for equation-heavy scanned papers.
            ocr_engine = Pix2Text(analyzer_config={"model_name": "mfd"})
            ocr_backend = "pix2text"
            print("[INFO] OCR backend: pix2text")
        except Exception as exc:
            if cfg.ocr_backend == "pix2text":
                raise RuntimeError(f"Pix2Text initialization failed: {exc}") from exc
            print(f"[WARN] Pix2Text unavailable, falling back to tesseract: {exc}")
            ocr_backend = "tesseract"

    if ocr_backend == "tesseract":
        check = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
        if check.returncode != 0:
            raise RuntimeError("Tesseract is not installed or not in PATH. Install it and retry.")
        print("[INFO] OCR backend: tesseract")

    pdf_files = sorted(cfg.input_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found under {cfg.input_dir}")
        return

    total = 0
    for pdf_path in pdf_files:
        try:
            chunk_count = ingest_pdf(pdf_path, cfg, ocr_backend, ocr_engine, collection)
            total += chunk_count
            print(f"[OK] {pdf_path} -> {chunk_count} chunks")
        except Exception as exc:
            print(f"[ERROR] {pdf_path}: {exc}")

    print(f"Done. Total chunks ingested: {total}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fatal error: {exc}")
        raise SystemExit(1) from exc
