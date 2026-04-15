from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import re
import csv
import io
import base64

import httpx

from pypdf import PdfReader
from app import config

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover
    fitz = None

from app.graph import (
    _build_context,
    _llm_for_agent,
    _manager_llm,
    _normalize_tutor_format,
    _select_expert_prompts,
    rag_graph,
)
from app.services.retrieval import hybrid_retrieve_with_debug


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass
class PaperJob:
    stream: str
    question_path: Path
    answer_path: Optional[Path]


_AGENT_FOR_EVAL = {
    "NewtonAgent": "Give derivation-first solution with concise equations.",
    "GaussAgent": "Give rigor-first solution and include a verification step.",
    "FeynmanAgent": "Give intuition-first explanation then concise solution.",
    "PolyaAgent": "Use Polya steps and clearly check the result.",
    "GriffithsAgent": "Use quantum intuition-first style for relevant physics math.",
    "SakuraiAgent": "Use formal operator/postulate style for relevant physics math.",
}


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _eval_output_dir() -> Path:
    out = _workspace_root() / "data" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_") or "eval"


def _invoke_vlm_ocr(image_data_url: str, prompt: str) -> str:
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


def _extract_pdf_text_via_vlm(path: Path, max_pages: int) -> str:
    if fitz is None:
        return ""

    try:
        doc = fitz.open(str(path))
    except Exception:
        return ""

    parts: List[str] = []
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
                parts.append(f"[Page {i+1}]\n{ocr_text.strip()}")
    finally:
        doc.close()

    return "\n\n".join(parts)


def _extract_text(path: Path, use_vlm_ocr_for_scanned_pdf: bool = False, ocr_max_pages: int = 3) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(str(path))
        base_text = "\n".join((p.extract_text() or "") for p in reader.pages)
        if len(base_text.strip()) >= config.EVAL_MIN_PDF_TEXT_CHARS_FOR_NATIVE_OCR:
            return base_text

        if use_vlm_ocr_for_scanned_pdf:
            vlm_text = _extract_pdf_text_via_vlm(path, max_pages=ocr_max_pages)
            if len(vlm_text.strip()) > len(base_text.strip()):
                return vlm_text

        return base_text
    return path.read_text(encoding="utf-8", errors="ignore")


def _final_answer(text: str) -> str:
    m = re.search(r"(?i)final answer\s*:\s*(.+)", text or "")
    return (m.group(1).strip() if m else "")[:300]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(_tokenize(a)), set(_tokenize(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _numeric_overlap(answer_text: str, generated_text: str) -> float:
    gold_nums = set(re.findall(r"[-+]?\d+(?:\.\d+)?", answer_text or ""))
    pred_nums = set(re.findall(r"[-+]?\d+(?:\.\d+)?", generated_text or ""))
    if not gold_nums:
        return 0.0
    return len(gold_nums & pred_nums) / len(gold_nums)


def _infer_error_categories(issues: List[str], answer_text: str) -> List[str]:
    text = " ".join(issues or []).lower() + " " + (answer_text or "").lower()
    cats: List[str] = []

    if any(k in text for k in ["algebra", "sign", "expand", "factor", "simplif", "arithmetic"]):
        cats.append("algebra_arithmetic")
    if any(k in text for k in ["derivative", "integral", "limit", "calculus", "differentiat"]):
        cats.append("calculus")
    if any(k in text for k in ["unit", "dimension", "dimensional"]):
        cats.append("units_dimension")
    if any(k in text for k in ["proof", "logic", "assumption", "therefore", "implies", "rigor"]):
        cats.append("logic_proof")
    if any(k in text for k in ["notation", "symbol", "variable", "define", "ambiguous"]):
        cats.append("notation_clarity")
    if any(k in text for k in ["missing", "incomplete", "no final", "unfinished", "uncertain"]):
        cats.append("incomplete_answer")
    if any(k in text for k in ["source", "ground", "unsupported", "hallucinat", "context"]):
        cats.append("grounding")

    if not cats:
        cats.append("other")
    return list(dict.fromkeys(cats))


def _aggregate_error_categories(papers: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for p in papers:
        for c in p.get("error_categories", []):
            out[c] = out.get(c, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def _aggregate_verdicts(papers: List[Dict[str, Any]]) -> Dict[str, int]:
    v: Dict[str, int] = {"pass": 0, "warn": 0, "fail": 0}
    for p in papers:
        verdict = str((p.get("manager_review") or {}).get("verdict", "warn")).lower()
        if verdict not in v:
            v["warn"] += 1
        else:
            v[verdict] += 1
    return v


def _severity_score(row: Dict[str, Any]) -> float:
    scores = row.get("scores", {}) or {}
    collab = float(scores.get("collaboration_score", 0.0) or 0.0)
    consensus = float(scores.get("consensus_score", 0.0) or 0.0)
    overlap = scores.get("answer_overlap_score")
    overlap_penalty = 0.0 if overlap is None else (1.0 - float(overlap or 0.0))
    verdict = str((row.get("manager_review") or {}).get("verdict", "warn")).lower()
    verdict_penalty = {"pass": 0.0, "warn": 0.35, "fail": 0.7}.get(verdict, 0.35)
    return round(0.45 * (1.0 - collab) + 0.25 * (1.0 - consensus) + 0.2 * overlap_penalty + 0.1 * verdict_penalty, 3)


def _mine_top_failures(papers: List[Dict[str, Any]], top_n: int = 20) -> List[Dict[str, Any]]:
    ranked = sorted(papers, key=_severity_score, reverse=True)
    mined: List[Dict[str, Any]] = []
    for row in ranked[: max(1, top_n)]:
        review = row.get("manager_review") or {}
        mined.append(
            {
                "stream": row.get("stream"),
                "paper_name": row.get("paper_name"),
                "question_excerpt": row.get("question_excerpt", ""),
                "question_path": row.get("question_path"),
                "severity": _severity_score(row),
                "error_categories": row.get("error_categories", []),
                "issues": review.get("issues", []),
                "recommended_final": review.get("recommended_final", ""),
                "current_final": _final_answer((row.get("together") or {}).get("answer", "")),
                "collaboration_score": (row.get("scores") or {}).get("collaboration_score"),
            }
        )
    return mined


def _write_failure_jsonl(top_failures: List[Dict[str, Any]], run_tag: str) -> Tuple[str, str]:
    out_dir = _eval_output_dir()
    run_path = out_dir / f"dse_failure_mining_{run_tag}.jsonl"
    latest_path = out_dir / "dse_failure_mining_latest.jsonl"

    lines: List[str] = []
    for row in top_failures:
        rec = {
            "instruction": row.get("question_excerpt", ""),
            "rejected": row.get("current_final", ""),
            "chosen": row.get("recommended_final", ""),
            "metadata": {
                "stream": row.get("stream"),
                "paper_name": row.get("paper_name"),
                "severity": row.get("severity"),
                "error_categories": row.get("error_categories", []),
            },
        }
        lines.append(json.dumps(rec, ensure_ascii=False))

    payload = "\n".join(lines)
    run_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")
    return str(run_path), str(latest_path)


def _balanced_split_records(
    records: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, List[Dict[str, Any]]]:
    total = max(1e-9, train_ratio + val_ratio + test_ratio)
    tr = train_ratio / total
    vr = val_ratio / total
    split_target = {"train": tr, "val": vr, "test": 1.0 - tr - vr}

    ordered = sorted(
        records,
        key=lambda r: len(((r.get("metadata") or {}).get("error_categories") or [])),
        reverse=True,
    )
    split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    def split_score(bucket: str, rec: Dict[str, Any]) -> float:
        cats = ((rec.get("metadata") or {}).get("error_categories") or [])
        size_after = len(split[bucket]) + 1
        total_after = len(ordered)
        ratio_penalty = abs((size_after / max(1, total_after)) - split_target[bucket])

        cat_counts = {
            c: sum(1 for x in split[bucket] if c in (((x.get("metadata") or {}).get("error_categories") or [])))
            for c in cats
        }
        cat_penalty = sum(cat_counts.values()) / max(1, len(cats)) if cats else 0.0
        return ratio_penalty + 0.15 * cat_penalty

    for rec in ordered:
        best = min(["train", "val", "test"], key=lambda b: split_score(b, rec))
        split[best].append(rec)

    return split


def _write_split_jsonl_files(
    preference_rows: List[Dict[str, Any]],
    run_tag: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, str]:
    out_dir = _eval_output_dir()
    split = _balanced_split_records(preference_rows, train_ratio, val_ratio, test_ratio)

    paths: Dict[str, str] = {}
    for bucket in ["train", "val", "test"]:
        run_path = out_dir / f"dse_failure_mining_{bucket}_{run_tag}.jsonl"
        latest_path = out_dir / f"dse_failure_mining_{bucket}_latest.jsonl"
        payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in split[bucket])
        run_path.write_text(payload, encoding="utf-8")
        latest_path.write_text(payload, encoding="utf-8")
        paths[f"{bucket}_run"] = str(run_path)
        paths[f"{bucket}_latest"] = str(latest_path)

    paths["stats"] = json.dumps(
        {
            k: {
                "size": len(v),
                "category_counts": _aggregate_error_categories(
                    [
                        {
                            "error_categories": ((r.get("metadata") or {}).get("error_categories") or []),
                        }
                        for r in v
                    ]
                ),
            }
            for k, v in split.items()
        },
        ensure_ascii=False,
    )
    return paths


def _csv_from_latest(report: Dict[str, Any], report_type: str = "papers") -> str:
    sio = io.StringIO()
    writer = csv.writer(sio)
    papers = report.get("papers", []) or []

    if report_type == "failures":
        failures = (report.get("summary") or {}).get("top_failures", []) or []
        writer.writerow(
            [
                "rank",
                "stream",
                "paper_name",
                "severity",
                "collaboration_score",
                "error_categories",
                "issues",
                "recommended_final",
                "question_path",
            ]
        )
        for i, row in enumerate(failures, start=1):
            writer.writerow(
                [
                    i,
                    row.get("stream", ""),
                    row.get("paper_name", ""),
                    row.get("severity", ""),
                    row.get("collaboration_score", ""),
                    "|".join(row.get("error_categories", []) or []),
                    " | ".join(row.get("issues", []) or []),
                    row.get("recommended_final", ""),
                    row.get("question_path", ""),
                ]
            )
        return sio.getvalue()

    writer.writerow(
        [
            "stream",
            "paper_name",
            "question_path",
            "answer_path",
            "collaboration_score",
            "consensus_score",
            "answer_overlap_score",
            "manager_verdict",
            "manager_confidence",
            "error_categories",
        ]
    )
    for row in papers:
        review = row.get("manager_review") or {}
        scores = row.get("scores") or {}
        writer.writerow(
            [
                row.get("stream", ""),
                row.get("paper_name", ""),
                row.get("question_path", ""),
                row.get("answer_path", ""),
                scores.get("collaboration_score", ""),
                scores.get("consensus_score", ""),
                scores.get("answer_overlap_score", ""),
                review.get("verdict", ""),
                review.get("confidence", ""),
                "|".join(row.get("error_categories", []) or []),
            ]
        )
    return sio.getvalue()


def _load_jobs(dse_root: str, max_papers: int) -> List[PaperJob]:
    root = Path(dse_root).expanduser().resolve()
    jobs: List[PaperJob] = []

    for stream in ["core", "m1", "m2"]:
        pastpaper = root / stream / "pastpaper"
        answer = root / stream / "answer"
        if not pastpaper.exists() or not pastpaper.is_dir():
            continue

        files = sorted([p for p in pastpaper.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS])
        for qf in files:
            af = answer / qf.name
            jobs.append(PaperJob(stream=stream.upper(), question_path=qf, answer_path=af if af.exists() else None))

    return jobs[: max(1, max_papers)]


def _solve_separately(question: str, context: str, expert_names: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in expert_names:
        style = _AGENT_FOR_EVAL.get(name, "Return concise derivation.")
        prompt = (
            f"You are {name}. {style}\n"
            "Return exactly tutor format: Hint, Steps, Final answer.\n"
            f"Context:\n{context[:7000]}\n\nQuestion:\n{question[:2500]}"
        )
        resp = _llm_for_agent(name).invoke(prompt)
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        out[name] = _normalize_tutor_format(content)
    return out


def _manager_review(question: str, separate: Dict[str, str], together: str) -> Dict[str, Any]:
    sections = "\n\n".join(f"[{k}]\n{v}" for k, v in separate.items())
    prompt = (
        "You are ManagerAgent QA reviewer.\n"
        "Given candidate solutions, assess consistency and return strict JSON only:\n"
        '{"verdict":"pass|warn|fail","issues":["..."],"recommended_final":"...","confidence":0.0}\n\n'
        f"Question:\n{question}\n\n"
        f"Separate solutions:\n{sections}\n\n"
        f"Collaborative manager output:\n{together}"
    )
    resp = _manager_llm().invoke(prompt)
    raw = resp.content if isinstance(resp.content, str) else str(resp.content)
    try:
        m = re.search(r"\{[\s\S]*\}", raw)
        return json.loads(m.group(0) if m else raw)
    except Exception:
        return {
            "verdict": "warn",
            "issues": ["Could not parse structured manager review"],
            "recommended_final": _final_answer(together),
            "confidence": 0.3,
        }


def _build_context_for_question(question: str) -> str:
    docs, _ = hybrid_retrieve_with_debug(question, top_k=5)
    return _build_context(docs)


def run_dse_evaluation(
    dse_root: str,
    include_answer_scoring: bool = False,
    max_papers: int = 30,
    use_vlm_ocr_for_scanned_pdf: bool = False,
    ocr_max_pages: int = 3,
    split_train_ratio: float = 0.8,
    split_val_ratio: float = 0.1,
    split_test_ratio: float = 0.1,
) -> Dict[str, Any]:
    jobs = _load_jobs(dse_root=dse_root, max_papers=max_papers)
    if not jobs:
        return {
            "success": False,
            "error": "No papers found. Expected structure: <root>/core|m1|m2/pastpaper",
            "papers": [],
        }

    papers: List[Dict[str, Any]] = []
    agent_totals: Dict[str, List[float]] = {}

    for idx, job in enumerate(jobs, start=1):
        q_text = _extract_text(
            job.question_path,
            use_vlm_ocr_for_scanned_pdf=use_vlm_ocr_for_scanned_pdf,
            ocr_max_pages=ocr_max_pages,
        ).strip()
        if not q_text:
            continue

        selected = list(_select_expert_prompts(q_text).keys())
        context = _build_context_for_question(q_text)

        # Separate solving: each expert works alone.
        separate = _solve_separately(q_text, context, selected)

        # Together solving: graph orchestrates experts + manager.
        together_state = rag_graph.invoke({"question": q_text})
        together_answer = _normalize_tutor_format(str(together_state.get("answer", "")))
        together_agents = together_state.get("agent_outputs", {}) or {}

        review = _manager_review(q_text, separate, together_answer)
        issues = review.get("issues", []) if isinstance(review.get("issues", []), list) else []
        error_categories = _infer_error_categories(issues, together_answer)

        finals = [_final_answer(v) for v in separate.values() if v]
        pairwise = []
        for i in range(len(finals)):
            for j in range(i + 1, len(finals)):
                pairwise.append(_jaccard(finals[i], finals[j]))
        consensus = round(sum(pairwise) / len(pairwise), 3) if pairwise else 0.0

        answer_score = None
        answer_path = str(job.answer_path) if job.answer_path else None
        if include_answer_scoring and job.answer_path and job.answer_path.exists():
            gold = _extract_text(
                job.answer_path,
                use_vlm_ocr_for_scanned_pdf=use_vlm_ocr_for_scanned_pdf,
                ocr_max_pages=ocr_max_pages,
            )
            answer_score = round(_numeric_overlap(gold, together_answer), 3)

        # Score by non-leaking collaboration quality.
        manager_conf = float(review.get("confidence", 0.0) or 0.0)
        verdict = str(review.get("verdict", "warn")).lower()
        verdict_score = {"pass": 1.0, "warn": 0.6, "fail": 0.2}.get(verdict, 0.5)
        final_quality = round(0.5 * consensus + 0.5 * min(1.0, max(0.0, (manager_conf + verdict_score) / 2)), 3)

        for agent_name, ans in separate.items():
            f = _final_answer(ans) or ans
            a_score = _jaccard(f, _final_answer(together_answer) or together_answer)
            agent_totals.setdefault(agent_name, []).append(a_score)

        papers.append(
            {
                "paper_index": idx,
                "stream": job.stream,
                "paper_name": job.question_path.name,
                "question_path": str(job.question_path),
                "answer_path": answer_path,
                "question_excerpt": q_text[:900],
                "separate": separate,
                "together": {
                    "answer": together_answer,
                    "agent_outputs": together_agents,
                },
                "manager_review": review,
                "error_categories": error_categories,
                "scores": {
                    "consensus_score": consensus,
                    "collaboration_score": final_quality,
                    "answer_overlap_score": answer_score,
                },
                "flags": {
                    "answer_scoring_used": bool(include_answer_scoring and answer_score is not None),
                    "no_answer_leak_in_generation": True,
                },
            }
        )

    by_stream: Dict[str, Dict[str, float]] = {}
    for s in ["CORE", "M1", "M2"]:
        stream_rows = [p for p in papers if p.get("stream") == s]
        if stream_rows:
            by_stream[s] = {
                "papers": float(len(stream_rows)),
                "avg_collaboration_score": round(
                    sum(float(r["scores"]["collaboration_score"]) for r in stream_rows) / len(stream_rows),
                    3,
                ),
            }

    by_agent = {
        agent: round(sum(vals) / len(vals), 3)
        for agent, vals in agent_totals.items()
        if vals
    }

    verdict_distribution = _aggregate_verdicts(papers)
    error_category_distribution = _aggregate_error_categories(papers)
    top_failures = _mine_top_failures(papers, top_n=20)

    result = {
        "success": True,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": {
            "dse_root": str(Path(dse_root).expanduser()),
            "include_answer_scoring": include_answer_scoring,
            "max_papers": max_papers,
            "use_vlm_ocr_for_scanned_pdf": use_vlm_ocr_for_scanned_pdf,
            "ocr_max_pages": ocr_max_pages,
            "split_train_ratio": split_train_ratio,
            "split_val_ratio": split_val_ratio,
            "split_test_ratio": split_test_ratio,
        },
        "summary": {
            "paper_count": len(papers),
            "by_stream": by_stream,
            "by_agent": by_agent,
            "verdict_distribution": verdict_distribution,
            "error_category_distribution": error_category_distribution,
            "top_failures": top_failures,
        },
        "papers": papers,
    }

    run_tag = _safe_slug(datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    failure_run_path, failure_latest_path = _write_failure_jsonl(top_failures, run_tag)
    preference_rows = [
        {
            "instruction": row.get("question_excerpt", ""),
            "rejected": row.get("current_final", ""),
            "chosen": row.get("recommended_final", ""),
            "metadata": {
                "stream": row.get("stream"),
                "paper_name": row.get("paper_name"),
                "severity": row.get("severity"),
                "error_categories": row.get("error_categories", []),
            },
        }
        for row in top_failures
    ]
    split_paths = _write_split_jsonl_files(
        preference_rows,
        run_tag=run_tag,
        train_ratio=split_train_ratio,
        val_ratio=split_val_ratio,
        test_ratio=split_test_ratio,
    )

    out_dir = _eval_output_dir()
    run_path = out_dir / f"dse_eval_{run_tag}.json"
    latest_path = out_dir / "dse_eval_latest.json"
    run_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    result["saved_file"] = str(run_path)
    result["latest_file"] = str(latest_path)
    result["fine_tune_failure_jsonl"] = failure_run_path
    result["fine_tune_failure_latest_jsonl"] = failure_latest_path
    result["fine_tune_split_jsonl"] = split_paths
    return result


def read_latest_evaluation() -> Dict[str, Any]:
    latest = _eval_output_dir() / "dse_eval_latest.json"
    if not latest.exists():
        return {
            "success": False,
            "error": "No evaluation report yet. Run /api/eval/run first.",
        }
    return json.loads(latest.read_text(encoding="utf-8"))


def latest_evaluation_csv(report_type: str = "papers") -> str:
    report = read_latest_evaluation()
    if not report.get("success"):
        return "error\nNo evaluation report yet\n"
    normalized_type = "failures" if (report_type or "").lower() == "failures" else "papers"
    return _csv_from_latest(report, report_type=normalized_type)
