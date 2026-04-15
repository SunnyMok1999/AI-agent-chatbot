import re
from typing import Iterable


def compute_marking_scheme_alignment(answer: str, marking_points: Iterable[str]) -> float:
    normalized_answer = re.sub(r"\s+", " ", (answer or "").lower()).strip()
    points = [re.sub(r"\s+", " ", (p or "").lower()).strip() for p in marking_points if (p or "").strip()]
    if not points:
        return 0.0
    matched = sum(1 for p in points if p in normalized_answer)
    return matched / len(points)
