"""Sentence-transformer embeddings for memory semantic search.

Separate from RAG: KAI's memory system must not import from rag/. The same
model family is used, but the pipeline is independent (memory retrieval is
not document retrieval).
"""
from __future__ import annotations

import threading
from typing import List, Optional, Sequence

import numpy as np

from .exceptions import MemoryConfigError

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_model_name: Optional[str] = None
_load_lock = threading.Lock()


def _get_model(model_name: str):
    global _model, _model_name
    with _load_lock:
        if _model is not None and _model_name == model_name:
            return _model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise MemoryConfigError(
                "sentence-transformers is required for memory embeddings. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        _model = SentenceTransformer(model_name)
        _model_name = model_name
        return _model


class MemoryEmbedder:
    """Wraps a sentence-transformer with lazy load and L2-normalized vectors."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            self._model = _get_model(self.model_name)
        return self._model

    def embed(self, text: str) -> List[float]:
        if not isinstance(text, str) or not text.strip():
            raise MemoryConfigError("Cannot embed empty text")
        model = self._ensure_model()
        vec = model.encode(
            [text.strip()],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return [float(x) for x in vec]

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        cleaned = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
        if not cleaned:
            return []
        model = self._ensure_model()
        vecs = model.encode(
            cleaned,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [[float(x) for x in v] for v in vecs]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)
