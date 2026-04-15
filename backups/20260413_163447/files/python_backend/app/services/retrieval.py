from dataclasses import dataclass
from typing import Any, List
import re

from app import config
from app.services.vectorstore import semantic_search
from app.services.reranker import reranker_service


@dataclass
class MathIntent:
    topic: str
    mode: str
    tags: List[str]


def classify_math_intent(question: str) -> MathIntent:
    q = question.lower()

    if re.search(r"\b(matrix|determinant|eigen|rank|nullspace)\b", q):
        topic = "linear_algebra"
    elif re.search(r"\b(grad|gradient|divergence|curl|nabla)\b", q):
        topic = "vector_calculus"
    elif re.search(r"\b(derivative|integral|limit|chain rule|partial derivative)\b", q):
        topic = "calculus"
    elif re.search(r"\b(polynomial|quadratic|factor|equation|inequality)\b", q):
        topic = "algebra"
    elif re.search(r"\b(prove|proof|show that)\b", q):
        topic = "proof"
    else:
        topic = "general"

    if re.search(r"\b(prove|proof|show that)\b", q):
        mode = "prove"
    elif re.search(r"\b(solve|roots?|find x)\b", q):
        mode = "solve"
    elif re.search(r"\b(compute|evaluate|calculate|differentiate|integrate)\b", q):
        mode = "compute"
    elif re.search(r"\b(explain|intuition|why|what is)\b", q):
        mode = "explain"
    else:
        mode = "general"

    tags = [x for x in [topic, mode] if x != "general"]
    return MathIntent(topic=topic, mode=mode, tags=tags)


def expand_query(question: str, intent: MathIntent) -> str:
    mapping = {
        "algebra": "equations roots factorization polynomial simplification",
        "calculus": "derivative integral limit chain rule product rule",
        "linear_algebra": "matrix determinant eigenvalue rank nullspace",
        "vector_calculus": "gradient divergence curl nabla line integral",
        "proof": "theorem lemma proof reasoning",
        "solve": "solve steps final answer",
        "compute": "calculation steps",
        "explain": "intuition concept explanation",
        "prove": "assumptions conclusion proof",
    }
    extra = " ".join(mapping.get(tag, "") for tag in intent.tags).strip()
    return f"{question} {extra}".strip() if extra else question


def _keyword_tokens(text: str) -> List[str]:
    stop = {"what", "is", "the", "and", "for", "with", "from", "that", "this"}
    raw = re.findall(r"[a-z0-9_]{2,}", text.lower())
    return sorted(set(x for x in raw if x not in stop))


def _keyword_score(content: str, tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    body = content.lower()
    match = sum(1 for t in tokens if t in body)
    return match / len(tokens)


def _intent_score(doc: Any, intent: MathIntent) -> float:
    signals = " ".join([
        str(doc.metadata.get("domain", "")).lower(),
        str(doc.metadata.get("tags", "")).lower(),
        (doc.page_content or "").lower(),
    ])
    score = 0.0
    for tag in intent.tags:
        normalized = tag.replace("_", " ")
        if tag in signals or normalized in signals:
            score += 0.2
    return min(score, 1.0)


def _domain_filter_for_intent(intent: MathIntent) -> dict[str, str] | None:
    if not config.ENABLE_DOMAIN_METADATA_FILTER:
        return None
    if intent.topic in {"general", "proof"}:
        return None
    return {"domain": intent.topic}


def _doc_matches_filter(doc: Any, metadata_filter: dict[str, str] | None) -> bool:
    if not metadata_filter:
        return True
    for k, v in metadata_filter.items():
        if str(doc.metadata.get(k, "")) != str(v):
            return False
    return True


def hybrid_retrieve(question: str, top_k: int, metadata_filter: dict[str, str] | None = None) -> List[Any]:
    docs, _ = hybrid_retrieve_with_debug(question=question, top_k=top_k, metadata_filter=metadata_filter)
    return docs


def hybrid_retrieve_with_debug(
    question: str,
    top_k: int,
    metadata_filter: dict[str, str] | None = None,
) -> tuple[List[Any], dict[str, Any]]:
    intent = classify_math_intent(question)
    query_a = question
    query_b = expand_query(question, intent)
    domain_filter = _domain_filter_for_intent(intent)
    filter_fallback_used = False
    metadata_filter_fallback_used = False

    # Chroma where-filter compatibility: use one simple filter in query,
    # then apply any secondary filter in-memory.
    primary_filter = metadata_filter or domain_filter
    secondary_filter = domain_filter if metadata_filter else None

    k = max(top_k, config.RETRIEVAL_CANDIDATE_K)
    docs_a = semantic_search(query_a, k=k, metadata_filter=primary_filter)
    docs_b = semantic_search(query_b, k=k, metadata_filter=primary_filter) if config.ENABLE_HYBRID_RETRIEVAL else []

    if secondary_filter:
        docs_a = [d for d in docs_a if _doc_matches_filter(d, secondary_filter)]
        docs_b = [d for d in docs_b if _doc_matches_filter(d, secondary_filter)]

    # Fallback to unfiltered retrieval if domain filter is too strict.
    if (primary_filter or secondary_filter) and len(docs_a) + len(docs_b) < max(2, top_k // 2):
        filter_fallback_used = True
        # First fallback keeps domain signal while dropping agent-specific metadata filter.
        if domain_filter and metadata_filter:
            metadata_filter_fallback_used = True
            docs_a = semantic_search(query_a, k=k, metadata_filter=domain_filter)
            docs_b = semantic_search(query_b, k=k, metadata_filter=domain_filter) if config.ENABLE_HYBRID_RETRIEVAL else []

        # Final fallback to fully unfiltered retrieval.
        if len(docs_a) + len(docs_b) < max(2, top_k // 2):
            docs_a = semantic_search(query_a, k=k)
            docs_b = semantic_search(query_b, k=k) if config.ENABLE_HYBRID_RETRIEVAL else []

    merged: dict[str, Any] = {}
    for doc in [*docs_a, *docs_b]:
        key = f"{(doc.page_content or '')[:300]}::{doc.metadata.get('source', '')}"
        if key not in merged:
            merged[key] = doc

    candidates = list(merged.values())

    debug: dict[str, Any] = {
        "intent": intent.__dict__,
        "query_primary": query_a,
        "query_expanded": query_b,
        "domain_filter": domain_filter,
        "metadata_filter": metadata_filter,
        "domain_filter_fallback_used": filter_fallback_used,
        "metadata_filter_fallback_used": metadata_filter_fallback_used,
        "candidate_count": len(candidates),
        "reranker": {
            "enabled": config.ENABLE_MATH_RERANKER,
            "method": "none",
            "scores": [],
        },
    }

    if not config.ENABLE_MATH_RERANKER:
        docs = candidates[:top_k]
        debug["reranker"]["method"] = "disabled"
        return docs, debug

    # Phase 3: true cross-encoder reranking service.
    reranked, cross_scores = reranker_service.rerank_with_scores(question, candidates, top_k)
    if reranked:
        # If score is None, cross-encoder was unavailable and fallback is needed.
        if cross_scores and cross_scores[0].get("score") is not None:
            debug["reranker"]["method"] = "cross_encoder"
            debug["reranker"]["scores"] = cross_scores
            return reranked, debug

    # Heuristic fallback reranker (used only if cross-encoder unavailable).
    tokens = _keyword_tokens(question)
    scored = []
    for idx, doc in enumerate(candidates):
        keyword_score = _keyword_score(doc.page_content or "", tokens)
        intent_score = _intent_score(doc, intent)
        rank_prior = 1 / (idx + 1)
        total = keyword_score * 0.6 + intent_score * 0.3 + rank_prior * 0.1
        scored.append((total, keyword_score, intent_score, rank_prior, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    docs = [doc for _, _, _, _, doc in top]

    debug_rows = []
    for i, (total, keyword_score, intent_score, rank_prior, doc) in enumerate(top):
        debug_rows.append(
            {
                "rank": i + 1,
                "score": float(total),
                "keyword_score": float(keyword_score),
                "intent_score": float(intent_score),
                "rank_prior": float(rank_prior),
                "source": doc.metadata.get("source", f"document_{i+1}"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "method": "heuristic_fallback",
            }
        )

    debug["reranker"]["method"] = "heuristic_fallback"
    debug["reranker"]["scores"] = debug_rows
    return docs, debug
