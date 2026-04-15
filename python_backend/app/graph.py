from typing import TypedDict, List, Dict, Any
import httpx
import re
import json

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from sympy import Eq, simplify, sympify
from sympy.core.sympify import SympifyError

from app import config
from app.services.retrieval import classify_math_intent, hybrid_retrieve_with_debug


TUTOR_PREFIX = "Teach me like a tutor. Give hints first, then full solution if I ask."


class GraphState(TypedDict, total=False):
    question: str
    guided_question: str
    intent: Dict[str, Any]
    docs: List[Any]
    context: str
    uploaded_text: str
    uploaded_source: str
    answer: str
    sources: List[str]
    image_data_url: str
    agent_outputs: Dict[str, str]
    retrieval_debug: Dict[str, Any]
    strict_validation: Dict[str, Any]
    ambiguity_debug: Dict[str, Any]
    needs_clarification: bool


_llm_cache: Dict[str, ChatOpenAI] = {}


def _build_llm(model_name: str) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=config.NVIDIA_API_KEY,
        base_url=config.NVIDIA_BASE_URL,
        model=model_name,
        temperature=0.1,
        timeout=60,
        max_retries=1,
    )


def _get_llm(model_name: str) -> ChatOpenAI:
    model = (model_name or config.NVIDIA_MODEL).strip()
    cached = _llm_cache.get(model)
    if cached:
        return cached
    client = _build_llm(model)
    _llm_cache[model] = client
    return client


def _llm_for_agent(agent_name: str) -> ChatOpenAI:
    return _get_llm(config.NVIDIA_MATH_MODEL)


def _manager_llm() -> ChatOpenAI:
    return _get_llm(config.NVIDIA_MATH_MODEL)


llm = _get_llm(config.NVIDIA_MATH_MODEL)

EXPERT_SYSTEM_PROMPTS = {
    "NewtonAgent": (
        "You are NewtonAgent. Focus on mechanics-style formal reasoning and clear mathematical derivation. "
        "State assumptions, define variables, and provide structured equations."
    ),
    "GaussAgent": (
        "You are GaussAgent. Focus on rigor, theorem-level precision, and algebraic correctness. "
        "Highlight edge cases and verification checks."
    ),
    "FeynmanAgent": (
        "You are FeynmanAgent. Focus on intuition, conceptual clarity, and beginner-friendly explanation. "
        "Use compact analogies without sacrificing correctness."
    ),
    "PolyaAgent": (
        "You are PolyaAgent. Focus on problem-solving heuristics: understand the problem, devise a plan, "
        "carry out the plan, and check results. Emphasize strategy and common fallback tactics."
    ),
    "GriffithsAgent": (
        "You are GriffithsAgent. Focus on intuition-first quantum mechanics explanations with concrete examples. "
        "Explain physical meaning before formalism and keep notation approachable."
    ),
    "SakuraiAgent": (
        "You are SakuraiAgent. Focus on rigorous quantum mechanics formalism using postulates, operators, "
        "state vectors, and careful mathematical structure."
    ),
}


def _is_quantum_question(question: str) -> bool:
    q = (question or "").lower()
    keywords = [
        "quantum",
        "wavefunction",
        "schrodinger",
        "schrödinger",
        "hilbert",
        "bra-ket",
        "operator",
        "eigenstate",
        "commutator",
        "spin",
        "uncertainty principle",
    ]
    return any(k in q for k in keywords)


def _select_expert_prompts(question: str) -> Dict[str, str]:
    if _is_quantum_question(question):
        return {
            "GriffithsAgent": EXPERT_SYSTEM_PROMPTS["GriffithsAgent"],
            "SakuraiAgent": EXPERT_SYSTEM_PROMPTS["SakuraiAgent"],
        }

    return {
        name: prompt
        for name, prompt in EXPERT_SYSTEM_PROMPTS.items()
        if name not in {"GriffithsAgent", "SakuraiAgent"}
    }


def _agent_profile_for_name(agent_name: str) -> str:
    return agent_name.replace("Agent", "").strip().lower()


def node_classify(state: GraphState) -> GraphState:
    question = state["question"].strip()
    guided = f"{TUTOR_PREFIX}\n\n{question}" if config.ENABLE_TUTOR_MODE else question
    intent = classify_math_intent(question)
    return {"guided_question": guided, "intent": intent.__dict__}


def _detect_ambiguity(
    question: str,
    intent: Dict[str, Any],
    has_image: bool = False,
    has_uploaded_text: bool = False,
) -> Dict[str, Any]:
    q = (question or "").strip()
    q_lower = q.lower()
    reasons: List[str] = []
    score = 0.0

    token_count = len(_tokenize(q))
    if token_count <= 4:
        score += 0.35
        reasons.append("question_too_short")

    if re.search(r"\b(this|that|it|these|those)\b", q_lower):
        score += 0.2
        reasons.append("deictic_reference_without_context")

    if re.search(r"\b(help|explain|solve|compute|calculate)\b", q_lower) and not re.search(r"[0-9=+\-*/^()]", q):
        score += 0.2
        reasons.append("task_verb_without_explicit_expression")

    topic = str(intent.get("topic", "general"))
    mode = str(intent.get("mode", "general"))
    if topic == "general" and mode == "general":
        score += 0.2
        reasons.append("low_intent_confidence")

    if has_image:
        # Uploaded image can supply missing context, so reduce ambiguity pressure.
        score = max(0.0, score - 0.2)

    if has_uploaded_text:
        # A recently uploaded paper gives strong context even when the follow-up is short.
        score = max(0.0, score - 0.35)

    threshold = config.AMBIGUITY_SCORE_THRESHOLD
    needs_clarification = bool(config.ENABLE_AMBIGUITY_GATE and score >= threshold)

    clarification = (
        "Hint: Your question may be under-specified.\n"
        "Steps:\n"
        "1) Tell me the exact topic (e.g., calculus, linear algebra, probability, quantum).\n"
        "2) Provide the full expression/equation and given values/assumptions.\n"
        "3) Tell me what output you want (final value, proof, or explanation).\n"
        "Final answer: Please send these details and I will solve it precisely."
    )

    return {
        "enabled": config.ENABLE_AMBIGUITY_GATE,
        "score": round(score, 3),
        "threshold": threshold,
        "needs_clarification": needs_clarification,
        "reasons": reasons,
        "clarification_prompt": clarification,
    }


def node_disambiguate(state: GraphState) -> GraphState:
    question = state.get("question", "")
    intent = state.get("intent", {}) or {}
    has_image = bool(state.get("image_data_url"))
    has_uploaded_text = bool((state.get("uploaded_text") or "").strip())
    ambiguity = _detect_ambiguity(
        question=question,
        intent=intent,
        has_image=has_image,
        has_uploaded_text=has_uploaded_text,
    )

    if ambiguity.get("needs_clarification"):
        return {
            "needs_clarification": True,
            "answer": ambiguity.get("clarification_prompt", ""),
            "sources": [],
            "retrieval_debug": {"ambiguity_gate": ambiguity},
            "ambiguity_debug": ambiguity,
        }

    return {
        "needs_clarification": False,
        "ambiguity_debug": ambiguity,
    }


def route_after_disambiguate(state: GraphState) -> str:
    return "clarify" if state.get("needs_clarification") else "retrieve"


def _build_context(docs: List[Any]) -> str:
    blocks = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", f"document_{i+1}")
        blocks.append(f"[Source {i+1}] {source}\n{doc.page_content}")
    return "\n\n".join(blocks)


def _normalize_requested_question_range(question: str) -> tuple[int, int] | None:
    q = (question or "").lower()
    m = re.search(r"\bquestions?\s*(\d+)\s*[-to]+\s*(\d+)\b", q)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (min(a, b), max(a, b))

    m = re.search(r"\b(?:q\s*)?(\d+)\s*[-,]\s*(?:q\s*)?(\d+)\b", q)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (min(a, b), max(a, b))

    m = re.search(r"\bquestions?\s*(\d+)\b", q)
    if m:
        n = int(m.group(1))
        return (n, n)

    return None


def _extract_requested_question_blocks(uploaded_text: str, question: str) -> str:
    text = (uploaded_text or "").strip()
    if not text:
        return ""

    question_range = _normalize_requested_question_range(question)
    if not question_range:
        return text

    start_n, end_n = question_range

    # Split common OCR formats like "1.", "1)", "1 " at line starts.
    parts = re.split(r"(?m)^\s*(\d+)\s*[\.)]\s*", text)
    if len(parts) < 3:
        return text

    chunks: list[tuple[int, str]] = []
    # parts layout: [preamble, num1, body1, num2, body2, ...]
    preamble = parts[0].strip()
    for i in range(1, len(parts) - 1, 2):
        try:
            num = int(parts[i])
        except ValueError:
            continue
        body = parts[i + 1].strip()
        if body:
            chunks.append((num, body))

    if not chunks:
        return text

    selected = [body for num, body in chunks if start_n <= num <= end_n]
    if not selected:
        return text

    header = f"Requested questions {start_n}-{end_n}" if start_n != end_n else f"Requested question {start_n}"
    return f"{header}\n\n{preamble}\n\n" + "\n\n".join(selected)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]{2,}", (text or "").lower())


def _keyword_overlap_score(question: str, content: str) -> float:
    q_tokens = [t for t in _tokenize(question) if t not in {"what", "is", "the", "and", "for", "with", "this", "that"}]
    if not q_tokens:
        return 0.0
    body = (content or "").lower()
    matched = sum(1 for t in set(q_tokens) if t in body)
    return matched / max(1, len(set(q_tokens)))


def _apply_source_hygiene(question: str, docs: List[Any], retrieval_debug: Dict[str, Any]) -> List[Any]:
    if not docs:
        return docs

    reranker_scores = retrieval_debug.get("reranker", {}).get("scores", []) if retrieval_debug else []
    score_by_key: Dict[str, float] = {}
    top_score = None
    for row in reranker_scores:
        score = row.get("score")
        if score is None:
            continue
        source = str(row.get("source", ""))
        chunk_index = str(row.get("chunk_index", ""))
        key = f"{source}::{chunk_index}"
        score = float(score)
        score_by_key[key] = score
        if top_score is None or score > top_score:
            top_score = score

    kept: List[Any] = []
    dropped: List[Dict[str, Any]] = []
    dropped_keys: set[str] = set()
    kept_keys: set[str] = set()

    for d in docs:
        source = str(d.metadata.get("source", ""))
        chunk_index = str(d.metadata.get("chunk_index", ""))
        key = f"{source}::{chunk_index}"
        overlap = _keyword_overlap_score(question, d.page_content or "")
        ce_score = score_by_key.get(key)

        overlap_ok = overlap >= config.SOURCE_MIN_KEYWORD_OVERLAP
        score_ok = True
        if top_score is not None and ce_score is not None:
            score_ok = ce_score >= (top_score - config.SOURCE_MAX_SCORE_GAP)

        if overlap_ok and score_ok:
            kept.append(d)
            kept_keys.add(key)
        else:
            dropped_keys.add(key)
            dropped.append(
                {
                    "source": source,
                    "chunk_index": d.metadata.get("chunk_index"),
                    "keyword_overlap": overlap,
                    "cross_encoder_score": ce_score,
                    "reason": "low_relevance",
                }
            )

    fallback_used = False
    if not kept:
        fallback_doc = docs[0]
        fallback_source = str(fallback_doc.metadata.get("source", ""))
        fallback_chunk_index = str(fallback_doc.metadata.get("chunk_index", ""))
        fallback_key = f"{fallback_source}::{fallback_chunk_index}"

        kept = [fallback_doc]
        kept_keys.add(fallback_key)
        fallback_used = True

        if fallback_key in dropped_keys:
            dropped = [
                row
                for row in dropped
                if f"{str(row.get('source', ''))}::{str(row.get('chunk_index', ''))}" != fallback_key
            ]

    # Defensive consistency: never show a chunk in both kept and dropped.
    if dropped:
        dropped = [
            row
            for row in dropped
            if f"{str(row.get('source', ''))}::{str(row.get('chunk_index', ''))}" not in kept_keys
        ]

    if retrieval_debug is not None:
        retrieval_debug["source_hygiene"] = {
            "kept": [
                {
                    "source": d.metadata.get("source"),
                    "chunk_index": d.metadata.get("chunk_index"),
                    "keyword_overlap": _keyword_overlap_score(question, d.page_content or ""),
                }
                for d in kept
            ],
            "dropped": dropped,
            "fallback_used": fallback_used,
        }

    return kept


def _extract_final_answer(answer: str) -> str:
    m = re.search(r"(?i)final answer\s*:\s*(.+)", answer)
    return m.group(1).strip() if m else ""


def _strict_probability_rule_check(question: str) -> Dict[str, Any]:
    q = question.lower().replace(" ", "")
    # Canonical check for common confusion: P(A)P(B)-P(A and B)
    if "p(a)*p(b)-p(aandb)" in q or "p(a)*p(b)-p(a∩b)" in q or "p(a)*p(b)-p(a&b)" in q:
        return {
            "status": "verified",
            "message": (
                "Hint: Use independence as the key condition.\n"
                "Steps:\n"
                "1) In general, P(A∩B) is not equal to P(A)P(B).\n"
                "2) P(A)P(B)-P(A∩B) equals 0 only when A and B are independent.\n"
                "Final answer: The expression is not an identity; it is 0 iff A and B are independent."
            ),
        }
    return {"status": "unknown"}


def _strict_algebra_check(question: str, answer: str) -> Dict[str, Any]:
    # Checks questions like "is expr1 = expr2 correct?"
    q = question.strip().lower()
    if "=" not in q or "is" not in q:
        return {"status": "unknown"}

    try:
        lhs_raw, rhs_raw = question.split("=", 1)
        lhs = sympify(lhs_raw.replace("^", "**"))
        rhs = sympify(re.sub(r"\bis\b.*$", "", rhs_raw, flags=re.I).replace("^", "**"))
        is_equal = simplify(lhs - rhs) == 0
        return {
            "status": "verified",
            "message": (
                "Hint: Compare both sides symbolically.\n"
                f"Steps:\n1) Simplify LHS - RHS.\n2) Result is {'0' if is_equal else 'non-zero'}.\n"
                f"Final answer: {'Correct' if is_equal else 'Not correct'}"
            ),
        }
    except (ValueError, SympifyError, TypeError):
        return {"status": "unknown"}


def _strict_validate(question: str, answer: str) -> Dict[str, Any]:
    if not config.ENABLE_STRICT_CORRECTNESS_MODE:
        return {"status": "disabled"}

    prob = _strict_probability_rule_check(question)
    if prob.get("status") == "verified":
        return prob

    alg = _strict_algebra_check(question, answer)
    if alg.get("status") == "verified":
        return alg

    uncertain_markers = ["i think", "might", "maybe", "not sure", "cannot determine"]
    if any(m in (answer or "").lower() for m in uncertain_markers):
        return {
            "status": "uncertain",
            "message": "I cannot verify this derivation with high confidence from available context.",
        }

    if not _extract_final_answer(answer):
        return {
            "status": "uncertain",
            "message": "I cannot verify this derivation with high confidence from available context.",
        }

    return {"status": "pass"}


def _normalize_tutor_format(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return "Hint: Start by identifying knowns and unknowns.\nSteps:\n1) Define variables.\nFinal answer: Unable to produce an answer."

    # Keep response concise.
    if len(text) > config.MAX_RESPONSE_CHARS:
        text = text[: config.MAX_RESPONSE_CHARS].rsplit(" ", 1)[0].strip() + "..."

    has_hint = re.search(r"(?i)^\s*hint\s*:", text) is not None
    has_steps = re.search(r"(?i)\bsteps\s*:", text) is not None
    has_final = re.search(r"(?i)\bfinal answer\s*:", text) is not None

    if has_hint and has_steps and has_final:
        return text

    # Build strict template from existing content.
    first_sentence = re.split(r"(?<=[.!?])\s+", text)[0].strip()
    bullets = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()][1:5]
    final_answer = _extract_final_answer(text) or "See concise steps above."

    if not bullets:
        bullets = ["Identify the relevant definition or rule.", "Apply it carefully to the expression."]

    formatted_steps = "\n".join(f"{i+1}) {b}" for i, b in enumerate(bullets[:4]))
    return f"Hint: {first_sentence}\nSteps:\n{formatted_steps}\nFinal answer: {final_answer}"


def _extract_tool_call_payload(text: str) -> Dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None

    # Prefer a direct JSON payload.
    candidates: List[str] = []
    if raw.startswith("{") and raw.endswith("}"):
        candidates.append(raw)

    # Fallback: find first JSON object containing "tool_call".
    m = re.search(r"\{[\s\S]*\"tool_call\"[\s\S]*\}", raw)
    if m:
        candidates.append(m.group(0))

    for c in candidates:
        try:
            payload = json.loads(c)
            if isinstance(payload, dict) and isinstance(payload.get("tool_call"), dict):
                return payload
        except json.JSONDecodeError:
            continue

    return None


def _tool_cite_source(query: str, top_k: int, agent_profile: str, default_context: str) -> str:
    q = (query or "").strip()
    if not q:
        return default_context[:1200] if default_context else "No query provided to cite_source."

    docs, _ = hybrid_retrieve_with_debug(
        q,
        top_k=max(1, min(top_k, 5)),
        metadata_filter={"agent_profile": agent_profile} if agent_profile else None,
    )
    if not docs:
        docs, _ = hybrid_retrieve_with_debug(q, top_k=max(1, min(top_k, 5)))

    if not docs:
        return "No relevant sources found."

    lines: List[str] = []
    for i, d in enumerate(docs[:3]):
        src = d.metadata.get("source", f"doc_{i+1}")
        snippet = (d.page_content or "").strip().replace("\n", " ")[:260]
        lines.append(f"[{i+1}] {src}: {snippet}")
    return "\n".join(lines)


def _tool_solve_subproblem(expression: str) -> str:
    expr = (expression or "").strip()
    if not expr:
        return "No expression provided."
    try:
        simplified = simplify(sympify(expr.replace("^", "**")))
        return f"simplified({expr}) = {simplified}"
    except Exception:
        return "Could not symbolically simplify the expression."


def _tool_check_dimensions(equation: str) -> str:
    eq = (equation or "").lower()
    if not eq:
        return "No equation provided."
    # Minimal heuristic checker.
    if "=" not in eq:
        return "No equality sign detected; cannot compare dimensions."
    lhs, rhs = eq.split("=", 1)
    physics_tokens = ["force", "mass", "acceleration", "energy", "momentum", "velocity", "time", "distance"]
    lhs_score = sum(1 for t in physics_tokens if t in lhs)
    rhs_score = sum(1 for t in physics_tokens if t in rhs)
    if lhs_score == rhs_score:
        return "Heuristic dimension check: likely consistent (token-balanced)."
    return "Heuristic dimension check: possible mismatch; verify units explicitly."


def _tool_request_peer_review(
    from_agent: str,
    to_agent: str,
    question: str,
    context: str,
    selected_experts: Dict[str, str],
) -> str:
    target = (to_agent or "").strip()
    if target not in selected_experts:
        return f"Peer agent '{target}' is not active for this question."

    peer_prompt = (
        f"{selected_experts[target]}\n"
        f"Peer review requested by {from_agent}.\n"
        "Return concise review: one correction (if any), one confidence score, one key equation.\n\n"
        f"Context:\n{context[:2000]}\n\n"
        f"Question/issue:\n{question}"
    )
    peer_res = _llm_for_agent(target).invoke(peer_prompt)
    return peer_res.content if isinstance(peer_res.content, str) else str(peer_res.content)


def _execute_tool_call(
    payload: Dict[str, Any],
    *,
    agent_name: str,
    agent_profile: str,
    context: str,
    question: str,
    selected_experts: Dict[str, str],
) -> str:
    tool_call = payload.get("tool_call") or {}
    name = str(tool_call.get("name", "")).strip()
    args = tool_call.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}

    if name == "cite_source":
        return _tool_cite_source(
            query=str(args.get("query", question)),
            top_k=int(args.get("top_k", 3)),
            agent_profile=agent_profile,
            default_context=context,
        )
    if name == "solve_subproblem":
        return _tool_solve_subproblem(str(args.get("expression", "")))
    if name == "check_dimensions":
        return _tool_check_dimensions(str(args.get("equation", "")))
    if name == "request_peer_review":
        return _tool_request_peer_review(
            from_agent=agent_name,
            to_agent=str(args.get("to_agent", "")),
            question=str(args.get("question", question)),
            context=context,
            selected_experts=selected_experts,
        )

    return f"Unknown tool: {name}"


def node_retrieve(state: GraphState) -> GraphState:
    docs, retrieval_debug = hybrid_retrieve_with_debug(state["question"], top_k=config.TOP_K_RETRIEVAL)
    ambiguity = state.get("ambiguity_debug")
    if ambiguity is not None and retrieval_debug is not None:
        retrieval_debug["ambiguity_gate"] = ambiguity
    docs = _apply_source_hygiene(state["question"], docs, retrieval_debug)
    context = _build_context(docs)
    sources = [doc.metadata.get("source", f"document_{i+1}") for i, doc in enumerate(docs)]
    return {
        "docs": docs,
        "context": context,
        "sources": list(dict.fromkeys(sources)),
        "retrieval_debug": retrieval_debug,
    }


def _invoke_vlm(question: str, image_data_url: str) -> str:
    if not (config.NVIDIA_API_KEY and config.NVIDIA_VLM_MODEL):
        return "A vision model is not configured yet. Set NVIDIA_VLM_MODEL in .env."

    payload = {
        "model": config.NVIDIA_VLM_MODEL,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
    }

    endpoint = f"{config.NVIDIA_BASE_URL.rstrip('/')}/chat/completions"
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
            return f"Vision model request failed: {res.text}"
        data = res.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def _vlm_preprocess_for_agents(question: str, image_data_url: str) -> str:
    preprocess_prompt = (
        "You are an OCR + math extraction assistant.\n"
        "Extract relevant text, equations, symbols, and given values from the image.\n"
        "Return concise structured output with:\n"
        "1) Problem statement\n"
        "2) Given values\n"
        "3) Target asked\n"
        "4) Clean equations\n"
        "Do not solve the problem.\n\n"
        f"User question: {question}"
    )
    extracted = _invoke_vlm(preprocess_prompt, image_data_url)
    if len(extracted) > config.MAX_VLM_PREPROCESS_CHARS:
        extracted = extracted[: config.MAX_VLM_PREPROCESS_CHARS].rsplit(" ", 1)[0].strip() + "..."
    return extracted


def node_generate(state: GraphState) -> GraphState:
    guided_question = state["guided_question"]
    context = state.get("context", "")

    uploaded_text = (state.get("uploaded_text") or "").strip()
    uploaded_source = (state.get("uploaded_source") or "").strip()
    if uploaded_text:
        upload_header = f"[Latest uploaded document{f': {uploaded_source}' if uploaded_source else ''}]"
        upload_context = f"{upload_header}\n{_extract_requested_question_blocks(uploaded_text, state.get('question', guided_question))}"
        if context:
            context = f"{upload_context}\n\n{context}"
        else:
            context = upload_context

    if state.get("image_data_url"):
        if config.ENABLE_VLM_PREPROCESS_FOR_AGENTS:
            extracted = _vlm_preprocess_for_agents(state.get("question", guided_question), state["image_data_url"])
            guided_question = (
                f"{guided_question}\n\n"
                "Image-derived structured context:\n"
                f"{extracted}"
            )
            if context:
                context = f"{context}\n\n[Image context]\n{extracted}"
            else:
                context = f"[Image context]\n{extracted}"
        else:
            answer = _invoke_vlm(guided_question, state["image_data_url"])
            return {"answer": answer}

    if len(context) > config.MULTI_AGENT_MAX_CONTEXT_CHARS:
        context = context[: config.MULTI_AGENT_MAX_CONTEXT_CHARS]

    if config.ENABLE_MULTI_AGENT_TUTOR:
        agent_outputs: Dict[str, str] = {}
        agent_tool_traces: Dict[str, Any] = {}
        selected_experts = _select_expert_prompts(state.get("question", guided_question))

        for agent_name, agent_prompt in selected_experts.items():
            agent_profile = _agent_profile_for_name(agent_name)
            agent_docs, _ = hybrid_retrieve_with_debug(
                state["question"],
                top_k=config.TOP_K_RETRIEVAL,
                metadata_filter={"agent_profile": agent_profile},
            )
            agent_retrieved_context = _build_context(agent_docs) if agent_docs else ""
            if uploaded_text:
                agent_context = context
                if agent_retrieved_context:
                    agent_context = f"{context}\n\n[Retrieved reference context]\n{agent_retrieved_context}"
            else:
                agent_context = agent_retrieved_context if agent_retrieved_context else context
            if len(agent_context) > config.MULTI_AGENT_MAX_CONTEXT_CHARS:
                agent_context = agent_context[: config.MULTI_AGENT_MAX_CONTEXT_CHARS]

            expert_prompt = (
                f"{agent_prompt}\n"
                "You can optionally call one tool by returning ONLY JSON in this exact schema:\n"
                '{"tool_call":{"name":"cite_source|solve_subproblem|check_dimensions|request_peer_review","arguments":{...}}}\n'
                "If no tool needed, return concise output only: one hint sentence, 2-4 numbered steps, one pitfall, confidence 0-1.\n\n"
                f"Context:\n{agent_context}\n\nQuestion: {guided_question}"
            )
            expert_result = _llm_for_agent(agent_name).invoke(expert_prompt)
            expert_text = expert_result.content if isinstance(expert_result.content, str) else str(expert_result.content)

            tool_trace: List[Dict[str, Any]] = []
            max_tool_calls = 2
            tool_calls = 0
            while tool_calls < max_tool_calls:
                payload = _extract_tool_call_payload(expert_text)
                if not payload:
                    break

                tool_calls += 1
                tool_result = _execute_tool_call(
                    payload,
                    agent_name=agent_name,
                    agent_profile=agent_profile,
                    context=agent_context,
                    question=guided_question,
                    selected_experts=selected_experts,
                )
                tool_trace.append({
                    "call": payload.get("tool_call", {}),
                    "result_preview": (tool_result or "")[:400],
                })

                follow_up_prompt = (
                    f"{agent_prompt}\n"
                    "You executed a tool. Now continue reasoning and either call another tool (same JSON schema) or produce final concise output.\n\n"
                    f"Question: {guided_question}\n\n"
                    f"Tool result:\n{tool_result}\n"
                )
                follow_up = _llm_for_agent(agent_name).invoke(follow_up_prompt)
                expert_text = follow_up.content if isinstance(follow_up.content, str) else str(follow_up.content)

            agent_outputs[agent_name] = expert_text
            if tool_trace:
                agent_tool_traces[agent_name] = tool_trace

        expert_sections = "\n\n".join(
            f"[{agent_name}]\n{agent_outputs.get(agent_name, '')}"
            for agent_name in selected_experts.keys()
        )

        synthesis_prompt = (
            "You are the ManagerAgent for a mathematics tutoring system.\n"
            "You received expert outputs from multiple specialist tutors.\n"
            "Strictly output this exact format and be concise:\n"
            "Hint: ...\n"
            "Steps:\n1) ...\n2) ...\n3) ...\n"
            "Final answer: ...\n"
            "Do not add extra sections. Do not use filler text. Refuse uncertain derivations.\n"
            "Be faithful to context and do not invent facts.\n\n"
            f"Question: {guided_question}\n\n"
            f"Context:\n{context}\n\n"
            "Expert Outputs:\n"
            f"{expert_sections}\n"
        )

        synth_result = _manager_llm().invoke(synthesis_prompt)
        answer = synth_result.content if isinstance(synth_result.content, str) else str(synth_result.content)
        answer = _normalize_tutor_format(answer)
        if agent_tool_traces:
            agent_outputs["_tool_traces"] = agent_tool_traces
        return {"answer": answer, "agent_outputs": agent_outputs}

    if context:
        prompt = (
            "You are a patient Mathematics tutor for university students. "
            "Use ONLY the retrieved context. Start with a hint, then numbered steps, and end with 'Final answer: ...'.\n\n"
            f"Context:\n{context}\n\nQuestion: {guided_question}"
        )
    else:
        prompt = (
            "You are a patient Mathematics tutor for university students. "
            "Start with a hint, then numbered steps, and end with 'Final answer: ...'.\n\n"
            f"Question: {guided_question}"
        )

    result = _manager_llm().invoke(prompt)
    answer = result.content if isinstance(result.content, str) else str(result.content)
    answer = _normalize_tutor_format(answer)
    return {"answer": answer}


def node_validate(state: GraphState) -> GraphState:
    answer = state.get("answer", "")
    question = state.get("question", "")

    answer = _normalize_tutor_format(answer)
    strict = _strict_validate(question=question, answer=answer)

    if strict.get("status") == "verified":
        answer = _normalize_tutor_format(strict.get("message", answer))
    elif strict.get("status") == "uncertain" and config.REFUSE_UNCERTAIN_DERIVATIONS:
        answer = (
            "Hint: I need verifiable conditions before deriving.\n"
            "Steps:\n"
            "1) State exact assumptions (e.g., independence, domain, constraints).\n"
            "2) Provide the precise expression/equation to verify.\n"
            "Final answer: I cannot provide a reliable derivation with current information."
        )

    if len(answer) > config.MAX_RESPONSE_CHARS:
        answer = answer[: config.MAX_RESPONSE_CHARS].rsplit(" ", 1)[0].strip() + "..."

    return {"answer": answer, "strict_validation": strict}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify", node_classify)
    graph.add_node("disambiguate", node_disambiguate)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("generate", node_generate)
    graph.add_node("validate", node_validate)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "disambiguate")
    graph.add_conditional_edges(
        "disambiguate",
        route_after_disambiguate,
        {
            "clarify": END,
            "retrieve": "retrieve",
        },
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", END)
    return graph.compile()


rag_graph = build_graph()
