from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Dict, Iterable, List, Sequence


def normalize_math_ocr_text(text: str) -> str:
    raw = text or ""
    replacements = {
        "−": "-",
        "–": "-",
        "—": "-",
        "×": "*",
        "·": "*",
        "÷": "/",
        "∕": "/",
        "＝": "=",
        "∑": "sum",
        "∫": "integral",
        "π": "pi",
        "θ": "theta",
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "λ": "lambda",
        "Δ": "Delta",
        "∞": "infinity",
        "√": "sqrt",
        "≤": "<=",
        "≥": ">=",
        "≠": "!=",
        "≈": "~=",
    }
    normalized = raw
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def compute_pdf_fingerprint(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


def compute_page_checksums(page_texts: Sequence[str]) -> Dict[int, str]:
    checksums: Dict[int, str] = {}
    for i, text in enumerate(page_texts):
        normalized = normalize_math_ocr_text(text)
        checksums[i + 1] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return checksums


@dataclass
class QuestionBlock:
    question_no: int
    content: str
    pages: List[int]


def _find_question_anchors(text: str) -> List[tuple[int, int]]:
    anchors: List[tuple[int, int]] = []
    for match in re.finditer(r"(?m)^\s*(\d{1,2})\s*[\.)]\s+", text):
        try:
            q = int(match.group(1))
        except (TypeError, ValueError):
            continue
        anchors.append((q, match.start()))
    return anchors


def segment_hkdse_questions(
    full_text: str,
    page_texts: Sequence[str],
    question_numbers: Iterable[int] = (1, 2, 3),
) -> Dict[int, QuestionBlock]:
    text = normalize_math_ocr_text(full_text)
    normalized_pages = [normalize_math_ocr_text(p).lower() for p in page_texts]
    anchors = _find_question_anchors(text)
    targets = set(question_numbers)
    out: Dict[int, QuestionBlock] = {}

    if not anchors:
        return out

    for idx, (q_no, start) in enumerate(anchors):
        if q_no not in targets:
            continue
        end = anchors[idx + 1][1] if idx + 1 < len(anchors) else len(text)
        body = text[start:end].strip()
        if not body:
            continue

        body_pages: List[int] = []
        snippet = body[:120].lower()
        for p_i, page_text in enumerate(normalized_pages):
            if snippet and snippet in page_text:
                body_pages.append(p_i + 1)
        if not body_pages:
            body_pages = [1]

        out[q_no] = QuestionBlock(question_no=q_no, content=body, pages=body_pages)

    return out


def extract_requested_question_numbers(question: str) -> List[int]:
    q = (question or "").lower()
    requested: List[int] = []

    m = re.search(r"\bquestions?\s*(\d+)\s*(?:-|to)\s*(\d+)\b", q)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        lo, hi = min(a, b), max(a, b)
        return [x for x in range(lo, hi + 1) if 1 <= x <= 3]

    singles = re.findall(r"\b(?:question|q)\s*(\d+)\b", q)
    for s in singles:
        n = int(s)
        if 1 <= n <= 3 and n not in requested:
            requested.append(n)
    return requested


def detect_scan_issues(page_texts: Sequence[str]) -> List[str]:
    issues: List[str] = []
    merged = "\n".join(page_texts)
    compact = re.sub(r"\s+", "", merged)
    if len(compact) < 80:
        issues.append("low_readable_text_density")
    if "�" in merged or "\x00" in merged:
        issues.append("broken_symbol_encoding")
    if re.search(r"[∫∑πθαλβγ]", merged) and not re.search(r"(integral|sum|pi|theta|alpha|beta|gamma|lambda)", merged.lower()):
        issues.append("complex_notation_unstable_ocr")
    if re.search(r"\bhandwritten\b", merged.lower()):
        issues.append("possible_handwritten_annotation")
    return issues
