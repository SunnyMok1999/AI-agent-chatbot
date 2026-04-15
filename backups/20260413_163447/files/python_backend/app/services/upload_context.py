from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class UploadTextContext:
    source: str
    content: str
    uploaded_at: float


@dataclass
class UploadImageContext:
    source: str
    mime_type: str
    data_url: str
    uploaded_at: float


_latest_text: Optional[UploadTextContext] = None
_latest_image: Optional[UploadImageContext] = None


def set_latest_text(source: str, content: str) -> None:
    global _latest_text
    _latest_text = UploadTextContext(source=source, content=content, uploaded_at=time.time())


def get_latest_text(max_age_seconds: int = 1800) -> Optional[UploadTextContext]:
    if not _latest_text:
        return None
    if time.time() - _latest_text.uploaded_at > max_age_seconds:
        return None
    return _latest_text


def set_latest_image(source: str, mime_type: str, data_url: str) -> None:
    global _latest_image
    _latest_image = UploadImageContext(
        source=source,
        mime_type=mime_type,
        data_url=data_url,
        uploaded_at=time.time(),
    )


def get_latest_image(max_age_seconds: int = 1800) -> Optional[UploadImageContext]:
    if not _latest_image:
        return None
    if time.time() - _latest_image.uploaded_at > max_age_seconds:
        return None
    return _latest_image
