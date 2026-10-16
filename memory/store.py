"""Memory store interface and local in-memory implementation.

Provides a common interface so MemoryService works identically against
Supabase (production) and the in-memory store (tests, offline local mode).
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from .embeddings import cosine_similarity


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore(ABC):
    @abstractmethod
    def create(
        self,
        user_id: str,
        content: str,
        embedding: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        ...

    @abstractmethod
    def search(
        self,
        user_id: str,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def list_user(
        self,
        user_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def delete(self, user_id: str, memory_id: str) -> bool:
        ...


class InMemoryStore(MemoryStore):
    """Non-persistent store with ownership enforcement and client-side cosine."""

    def __init__(self):
        self._rows: Dict[str, Dict[str, Any]] = {}

    def create(
        self,
        user_id: str,
        content: str,
        embedding: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        row = {
            "id": _new_id(),
            "user_id": user_id,
            "content": content,
            "embedding": list(embedding) if embedding is not None else None,
            "created_at": _utcnow_iso(),
        }
        self._rows[row["id"]] = row
        return {k: v for k, v in row.items() if k != "embedding"}

    def search(
        self,
        user_id: str,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        candidates = [
            r
            for r in self._rows.values()
            if r["user_id"] == user_id and r["embedding"] is not None
        ]
        scored = []
        for r in candidates:
            sim = cosine_similarity(query_embedding, r["embedding"])
            scored.append((sim, r))
        scored.sort(key=lambda t: t[0], reverse=True)
        out = []
        for sim, r in scored[: max(0, top_k)]:
            item = {k: v for k, v in r.items() if k != "embedding"}
            item["similarity"] = sim
            out.append(item)
        return out

    def list_user(
        self,
        user_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        rows = [r for r in self._rows.values() if r["user_id"] == user_id]
        rows.sort(key=lambda r: r["created_at"], reverse=True)
        return [
            {k: v for k, v in r.items() if k != "embedding"}
            for r in rows[: max(0, limit)]
        ]

    def delete(self, user_id: str, memory_id: str) -> bool:
        row = self._rows.get(memory_id)
        if row is None or row["user_id"] != user_id:
            return False
        del self._rows[memory_id]
        return True
