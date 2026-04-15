from typing import Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import re
import math

from app import config


class LocalHashEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def _to_vector(self, text: str) -> List[float]:
        vec = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9_]+", (text or "").lower())

        for token in tokens:
            h = 2166136261
            for ch in token:
                h ^= ord(ch)
                h = (h + (h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24)) & 0xFFFFFFFF
            idx = abs(h) % self.dimensions
            vec[idx] += 1.0

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._to_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._to_vector(text)


def _create_embeddings() -> Embeddings:
    model_name = config.EMBEDDING_MODEL
    try:
        # Xenova/* models are for JS transformers; fallback to a Python-compatible model.
        if model_name.lower().startswith("xenova/"):
            model_name = "BAAI/bge-small-en-v1.5"

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as error:
        print(f"[python_backend] Embedding model load failed ({model_name}). Falling back to LocalHashEmbeddings. Error: {error}")
        return LocalHashEmbeddings()


_embeddings = _create_embeddings()

_vectorstore = Chroma(
    collection_name=config.CHROMA_COLLECTION_NAME,
    embedding_function=_embeddings,
    persist_directory="./data/chromadb_py",
)


def get_vectorstore() -> Chroma:
    return _vectorstore


def add_documents(docs: List[Document]) -> None:
    if not docs:
        return
    _vectorstore.add_documents(docs)


def semantic_search(query: str, k: int, metadata_filter: Optional[Dict[str, str]] = None) -> List[Document]:
    if metadata_filter:
        return _vectorstore.similarity_search(query, k=k, filter=metadata_filter)
    return _vectorstore.similarity_search(query, k=k)
