from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional
import json
import time


@dataclass
class UploadTextContext:
    session_id: str
    source: str
    content: str
    uploaded_at: float
    fingerprint: str = ""
    page_checksums: Dict[int, str] | None = None
    question_blocks: Dict[str, str] | None = None
    question_pages: Dict[str, list[int]] | None = None
    extraction_method: str = "native"
    scan_issues: list[str] | None = None


@dataclass
class UploadImageContext:
    session_id: str
    source: str
    mime_type: str
    data_url: str
    uploaded_at: float


_STORE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "session_upload_context.json"
)
_text_by_session: Dict[str, UploadTextContext] = {}
_image_by_session: Dict[str, UploadImageContext] = {}
_latest_session_id: Optional[str] = None


def _serialize() -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "latest_session_id": _latest_session_id,
        "text_by_session": {k: asdict(v) for k, v in _text_by_session.items()},
        "image_by_session": {k: asdict(v) for k, v in _image_by_session.items()},
    }
    _STORE_PATH.write_text(json.dumps(payload), encoding="utf-8")


def _load() -> None:
    global _latest_session_id, _text_by_session, _image_by_session
    if not _STORE_PATH.exists():
        return
    try:
        payload = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
        _latest_session_id = payload.get("latest_session_id")
        raw_text = payload.get("text_by_session", {}) or {}
        raw_image = payload.get("image_by_session", {}) or {}
        _text_by_session = {k: UploadTextContext(**v) for k, v in raw_text.items()}
        _image_by_session = {k: UploadImageContext(**v) for k, v in raw_image.items()}
    except Exception:
        _latest_session_id = None
        _text_by_session = {}
        _image_by_session = {}


def _resolve_session_id(session_id: str | None = None) -> str:
    sid = (session_id or _latest_session_id or "default").strip()
    return sid if sid else "default"


def set_latest_text(
    source: str,
    content: str,
    *,
    session_id: str = "default",
    fingerprint: str = "",
    page_checksums: Optional[Dict[int, str]] = None,
    question_blocks: Optional[Dict[str, str]] = None,
    question_pages: Optional[Dict[str, list[int]]] = None,
    extraction_method: str = "native",
    scan_issues: Optional[list[str]] = None,
) -> None:
    global _latest_session_id
    sid = _resolve_session_id(session_id)
    _text_by_session[sid] = UploadTextContext(
        session_id=sid,
        source=source,
        content=content,
        uploaded_at=time.time(),
        fingerprint=fingerprint,
        page_checksums=page_checksums or {},
        question_blocks=question_blocks or {},
        question_pages=question_pages or {},
        extraction_method=extraction_method,
        scan_issues=scan_issues or [],
    )
    _latest_session_id = sid
    _serialize()


def clear_latest_text(session_id: str | None = None) -> None:
    sid = _resolve_session_id(session_id)
    _text_by_session.pop(sid, None)
    _serialize()


def get_latest_text(
    max_age_seconds: int = 1800,
    session_id: str | None = None,
) -> Optional[UploadTextContext]:
    sid = _resolve_session_id(session_id)
    latest = _text_by_session.get(sid)
    if not latest:
        return None
    if time.time() - latest.uploaded_at > max_age_seconds:
        return None
    return latest


def set_latest_image(source: str, mime_type: str, data_url: str, *, session_id: str = "default") -> None:
    global _latest_session_id
    sid = _resolve_session_id(session_id)
    _image_by_session[sid] = UploadImageContext(
        session_id=sid,
        source=source,
        mime_type=mime_type,
        data_url=data_url,
        uploaded_at=time.time(),
    )
    _latest_session_id = sid
    _serialize()


def clear_latest_image(session_id: str | None = None) -> None:
    sid = _resolve_session_id(session_id)
    _image_by_session.pop(sid, None)
    _serialize()


def get_latest_image(max_age_seconds: int = 1800, session_id: str | None = None) -> Optional[UploadImageContext]:
    sid = _resolve_session_id(session_id)
    latest = _image_by_session.get(sid)
    if not latest:
        return None
    if time.time() - latest.uploaded_at > max_age_seconds:
        return None
    return latest


_load()
