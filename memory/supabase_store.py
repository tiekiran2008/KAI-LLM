"""Supabase/PostgreSQL-backed memory store.

Uses the match_memories RPC (pgvector, cosine) when available; otherwise
falls back to fetching recent rows for the user and scoring client-side
with numpy. The pgvector Python package is NOT required.

Security:
- Credentials come only from environment variables (MemoryConfig.from_env).
- The anon key is the default credential; RLS enforces ownership at the DB
  layer (policies created by sql/001_create_memories.sql).
- The service-role key must never reach a frontend; it is not used here.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Sequence

from .config import MemoryConfig
from .embeddings import cosine_similarity
from .exceptions import MemoryConfigError
from .store import MemoryStore


class SupabaseMemoryStore(MemoryStore):
    requires_verified_identity = True

    def __init__(self, config: Optional[MemoryConfig] = None, client=None):
        self.config = config or MemoryConfig.from_env()
        if client is None:
            self.config.require_credentials()
            try:
                from supabase import create_client
            except ImportError as exc:
                raise MemoryConfigError(
                    "supabase is required for SupabaseMemoryStore. "
                    "Install it with: pip install supabase"
                ) from exc
            client = create_client(
                self.config.supabase_url,
                self.config.supabase_anon_key,
            )
        self.client = client

    def _table(self):
        return self.client.table(self.config.table_name)

    def _normalize_row(self, row: Dict[str, Any], similarity: Optional[float] = None) -> Dict[str, Any]:
        item = {
            "id": row.get("id"),
            "user_id": row.get("user_id"),
            "content": row.get("content"),
            "created_at": row.get("created_at"),
        }
        sim = similarity if similarity is not None else row.get("similarity")
        if sim is not None:
            item["similarity"] = float(sim)
        return item

    def _parse_embedding(self, raw: Any) -> Optional[List[float]]:
        if raw is None:
            return None
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return None
        try:
            return [float(x) for x in raw]
        except (TypeError, ValueError):
            return None

    def create(
        self,
        user_id: str,
        content: str,
        embedding: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        row_id = str(uuid.uuid4())
        record = {
            "id": row_id,
            "user_id": user_id,
            "content": content,
        }
        if embedding is not None:
            record["embedding"] = [float(x) for x in embedding]
        try:
            resp = self._table().insert(record).execute()
        except Exception:
            if "embedding" not in record:
                raise
            record.pop("embedding", None)
            resp = self._table().insert(record).execute()
        data = getattr(resp, "data", None) or []
        if data:
            return self._normalize_row(data[0])
        record.pop("embedding", None)
        return record

    def search(
        self,
        user_id: str,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        match_count = max(1, min(int(top_k), self.config.max_top_k))
        try:
            resp = self.client.rpc(
                self.config.match_function,
                {
                    "query_user_id": user_id,
                    "query_embedding": [float(x) for x in query_embedding],
                    "match_count": match_count,
                },
            ).execute()
            rows = getattr(resp, "data", None) or []
            out = []
            for r in rows:
                item = self._normalize_row(r)
                item.setdefault("user_id", user_id)
                out.append(item)
            return out[:match_count]
        except Exception:
            return self._fallback_search(user_id, query_embedding, match_count)

    def _fallback_search(
        self,
        user_id: str,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        resp = (
            self._table()
            .select("id, user_id, content, embedding, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(self.config.fallback_scan_limit)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        scored = []
        for r in rows:
            emb = self._parse_embedding(r.get("embedding"))
            if emb is None:
                continue
            sim = cosine_similarity(query_embedding, emb)
            scored.append((sim, r))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            self._normalize_row(r, similarity=sim) for sim, r in scored[:top_k]
        ]

    def list_user(
        self,
        user_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        resp = (
            self._table()
            .select("id, user_id, content, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(max(1, min(int(limit), self.config.max_list_limit)))
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        return [self._normalize_row(r) for r in rows]

    def delete(self, user_id: str, memory_id: str) -> bool:
        resp = (
            self._table()
            .select("id")
            .eq("id", memory_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        if not rows:
            return False
        self._table().delete().eq("id", memory_id).eq("user_id", user_id).execute()
        return True
