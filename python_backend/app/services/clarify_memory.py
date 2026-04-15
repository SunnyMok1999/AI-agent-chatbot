from dataclasses import dataclass
from typing import Optional, Dict, Any
import time


@dataclass
class PendingClarification:
    question: str
    ambiguity_debug: Dict[str, Any]
    created_at: float


_pending: Optional[PendingClarification] = None


def set_pending_clarification(question: str, ambiguity_debug: Dict[str, Any]) -> None:
    global _pending
    _pending = PendingClarification(
        question=(question or "").strip(),
        ambiguity_debug=ambiguity_debug or {},
        created_at=time.time(),
    )


def peek_pending_clarification(max_age_seconds: int = 600) -> Optional[PendingClarification]:
    global _pending
    if _pending is None:
        return None

    if time.time() - _pending.created_at > max_age_seconds:
        _pending = None
        return None

    return _pending


def clear_pending_clarification() -> None:
    global _pending
    _pending = None
