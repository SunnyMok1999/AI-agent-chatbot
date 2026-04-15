from typing import Any, List

from app import config


class CrossEncoderRerankerService:
    def __init__(self) -> None:
        self._model = None
        self._disabled_reason = ""

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not config.ENABLE_CROSS_ENCODER_RERANKER:
            self._disabled_reason = "disabled by config"
            return None
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(config.CROSS_ENCODER_MODEL, max_length=512)
            return self._model
        except Exception as error:
            self._disabled_reason = str(error)
            return None

    def rerank(self, query: str, docs: List[Any], top_k: int) -> List[Any]:
        reranked, _ = self.rerank_with_scores(query=query, docs=docs, top_k=top_k)
        return reranked

    def rerank_with_scores(self, query: str, docs: List[Any], top_k: int) -> tuple[List[Any], List[dict[str, Any]]]:
        if not docs:
            return [], []

        model = self._load_model()
        if model is None:
            # Fallback: no reranking available
            debug_rows = []
            for i, doc in enumerate(docs[:top_k]):
                debug_rows.append(
                    {
                        "rank": i + 1,
                        "score": None,
                        "source": doc.metadata.get("source", f"document_{i+1}"),
                        "chunk_index": doc.metadata.get("chunk_index"),
                        "method": "cross_encoder_unavailable",
                        "reason": self._disabled_reason or "model unavailable",
                    }
                )
            return docs[:top_k], debug_rows

        pairs = [(query, (doc.page_content or "")[:3000]) for doc in docs]
        scores = model.predict(pairs)

        scored = list(zip(scores, docs))
        scored.sort(key=lambda item: float(item[0]), reverse=True)

        top = scored[:top_k]
        reranked_docs = [doc for _, doc in top]
        debug_rows = []
        for i, (score, doc) in enumerate(top):
            debug_rows.append(
                {
                    "rank": i + 1,
                    "score": float(score),
                    "source": doc.metadata.get("source", f"document_{i+1}"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "method": "cross_encoder",
                }
            )

        return reranked_docs, debug_rows


reranker_service = CrossEncoderRerankerService()
