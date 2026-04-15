from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any

import chromadb
import fitz
import httpx
import streamlit as st
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer

load_dotenv()


NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_VLM_MODEL = os.getenv("NVIDIA_VLM_MODEL", "microsoft/phi-3-vision-128k-instruct")
NVIDIA_TEXT_MODEL = os.getenv("NVIDIA_TEXT_MODEL", "meta/llama-3.1-8b-instruct")
CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chroma_dsemath")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "dsemath_math")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
MAX_PDF_PAGES_FOR_CONTEXT = int(os.getenv("MAX_PDF_PAGES_FOR_CONTEXT", "4"))


@dataclass
class AgentConfig:
    key: str
    label: str
    system_prompt: str


AGENT_REGISTRY: dict[str, AgentConfig] = {
    "dsemath": AgentConfig(
        key="dsemath",
        label="DSEMATH AI Agent",
        system_prompt=(
            "You are DSEMATH AI Agent. You help with HKDSE Math Core, M1, and M2. "
            "Use tools when needed: `web_search` for latest references and `rag_retrieve` for local past-paper knowledge. "
            "If a user uploaded a session file, treat it as temporary conversation context only."
        ),
    )
}


class SentenceTransformerEmbedding:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors = self.model.encode(input, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]


@st.cache_resource(show_spinner=False)
def get_collection() -> Any:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding = SentenceTransformerEmbedding(EMBEDDING_MODEL)
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embedding,
        metadata={"hnsw:space": "cosine"},
    )


def to_data_url(mime_type: str, data: bytes) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('utf-8')}"


def pdf_to_page_images(pdf_bytes: bytes, max_pages: int = MAX_PDF_PAGES_FOR_CONTEXT) -> list[str]:
    images: list[str] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_count = len(doc) if max_pages <= 0 else min(len(doc), max_pages)
        for page_index in range(page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            images.append(to_data_url("image/png", pix.tobytes("png")))
    return images


def run_web_search(query: str) -> str:
    rows = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=5):
            rows.append({
                "title": item.get("title", ""),
                "url": item.get("href", ""),
                "snippet": item.get("body", ""),
            })
    return json.dumps(rows, ensure_ascii=False)


def run_rag_retrieve(query: str) -> str:
    try:
        collection = get_collection()
        result = collection.query(query_texts=[query], n_results=RAG_TOP_K)
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        payload = [{"content": d, "metadata": m} for d, m in zip(docs, metas)]
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def call_nvidia_chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]], model: str) -> dict[str, Any]:
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is not configured")

    endpoint = f"{NVIDIA_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.2,
    }

    with httpx.Client(timeout=60) as client:
        res = client.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        res.raise_for_status()
        return res.json()


def build_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current references or definitions.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rag_retrieve",
                "description": "Retrieve relevant chunks from local DSE Math vector store.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
    ]


def execute_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    if tool_name == "web_search":
        return run_web_search(arguments.get("query", ""))
    if tool_name == "rag_retrieve":
        return run_rag_retrieve(arguments.get("query", ""))
    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def run_agent(agent: AgentConfig, user_text: str, session_upload_images: list[str], chat_history: list[dict[str, str]]) -> str:
    tools = build_tools()
    model = NVIDIA_VLM_MODEL if session_upload_images else NVIDIA_TEXT_MODEL

    messages: list[dict[str, Any]] = [{"role": "system", "content": agent.system_prompt}]
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for image_data_url in session_upload_images:
        user_content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    messages.append({"role": "user", "content": user_content})

    for _ in range(4):
        response = call_nvidia_chat(messages=messages, tools=tools, model=model)
        choice = response.get("choices", [{}])[0].get("message", {})
        tool_calls = choice.get("tool_calls") or []

        if not tool_calls:
            content = choice.get("content", "")
            return content if isinstance(content, str) else json.dumps(content)

        messages.append(
            {
                "role": "assistant",
                "content": choice.get("content", ""),
                "tool_calls": tool_calls,
            }
        )

        for call in tool_calls:
            fn = call.get("function", {})
            tool_name = fn.get("name", "")
            raw_arguments = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_arguments)
            except json.JSONDecodeError:
                args = {}
            tool_output = execute_tool(tool_name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "name": tool_name,
                    "content": tool_output,
                }
            )

    return "I could not complete the tool workflow within the allowed steps. Please try again."


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("agent_key", "dsemath")
    st.session_state.setdefault("session_upload_images", [])
    st.session_state.setdefault("session_upload_names", [])


def render_ui() -> None:
    st.set_page_config(page_title="DSEMATH AI Agent", page_icon="🧠", layout="wide")
    st.title("DSEMATH AI Agent")
    st.caption("Unified multimodal assistant for DSE Math Core / M1 / M2")

    init_state()

    selected_label = st.selectbox(
        "Select agent",
        options=[cfg.label for cfg in AGENT_REGISTRY.values()],
        index=0,
    )
    selected = next(cfg for cfg in AGENT_REGISTRY.values() if cfg.label == selected_label)
    st.session_state.agent_key = selected.key

    uploads = st.file_uploader(
        "Session file uploads (.pdf, .png) - temporary context only",
        type=["pdf", "png"],
        accept_multiple_files=True,
    )

    if st.button("Update session uploads"):
        st.session_state.session_upload_images = []
        st.session_state.session_upload_names = []
        for upload in uploads or []:
            payload = upload.read()
            suffix = upload.name.lower().rsplit(".", maxsplit=1)[-1]
            if suffix == "png":
                st.session_state.session_upload_images.append(to_data_url("image/png", payload))
                st.session_state.session_upload_names.append(upload.name)
            elif suffix == "pdf":
                st.session_state.session_upload_images.extend(pdf_to_page_images(payload))
                st.session_state.session_upload_names.append(upload.name)

    if st.session_state.session_upload_names:
        st.info(f"Session-only multimodal files: {', '.join(st.session_state.session_upload_names)}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask DSE Math (Core/M1/M2) question...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = run_agent(
                agent=selected,
                user_text=prompt,
                session_upload_images=st.session_state.session_upload_images,
                chat_history=st.session_state.messages[:-1],
            )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    render_ui()
