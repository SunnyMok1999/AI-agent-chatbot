from pydantic import BaseModel
from typing import Optional, List


class ChatRequest(BaseModel):
    message: str
    stream: bool = False
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    success: bool
    content: str
    sources: Optional[List[str]] = None


class HealthResponse(BaseModel):
    success: bool
    message: str


class EvalRunRequest(BaseModel):
    dse_root: str
    include_answer_scoring: bool = False
    max_papers: int = 30
    use_vlm_ocr_for_scanned_pdf: bool = False
    ocr_max_pages: int = 3
    split_train_ratio: float = 0.8
    split_val_ratio: float = 0.1
    split_test_ratio: float = 0.1
