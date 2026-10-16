"""MemoryService: high-level API for KAI's long-term memory.

Key properties (project requirements):
- save_memory requires explicit approval (approved=True) — never store
  every conversation automatically.
- Identity: every operation is authorized against a verified authenticated
  user id (obtained from a Supabase session, not caller-supplied alone).
- Retrieval limits: top_k is capped; context is character-capped so entire
  conversation histories are never sent to the LLM.
- build_memory_context() produces a compact, LLM-ready memory block that is
  separate from the RAG pipeline (memory ≠ documents).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from .config import MemoryConfig
from .embeddings import MemoryEmbedder
from .exceptions import (
    ApprovalRequiredError,
    IdentityVerificationError,
    MemoryNotFoundError,
)
from .store import InMemoryStore, MemoryStore


class MemoryService:
    def __init__(
        self,
        store: MemoryStore,
        embedder: Optional[MemoryEmbedder] = None,
        authenticated_user_id: Optional[str] = None,
        config: Optional[MemoryConfig] = None,
        identity_verifier: Optional[Callable[[], str]] = None,
    ):
        if authenticated_user_id is None and identity_verifier is None:
            raise IdentityVerificationError(
                "MemoryService requires a verified authenticated user id "
                "or an identity_verifier callable"
            )
        self.store = store
        self.config = config or MemoryConfig()
        self.embedder = embedder or MemoryEmbedder(self.config.embedding_model)
        self._authenticated_user_id = authenticated_user_id
        self._identity_verifier = identity_verifier

    @classmethod
    def local(
        cls,
        authenticated_user_id: str,
        config: Optional[MemoryConfig] = None,
        embedder: Optional[MemoryEmbedder] = None,
    ) -> "MemoryService":
        """Local/test mode: identity is fixed and trusted (e.g. single-user
        laptop app, unit tests). Uses InMemoryStore — nothing is persisted."""
        return cls(
            store=InMemoryStore(),
            embedder=embedder,
            authenticated_user_id=authenticated_user_id,
            config=config,
        )

    @classmethod
    def from_supabase(
        cls,
        config: Optional[MemoryConfig] = None,
        client=None,
        embedder: Optional[MemoryEmbedder] = None,
    ) -> "MemoryService":
        from .auth import verify_authenticated_user
        from .supabase_store import SupabaseMemoryStore

        config = config or MemoryConfig.from_env()
        store = SupabaseMemoryStore(config=config, client=client)
        user_id = verify_authenticated_user(store.client)
        return cls(
            store=store,
            embedder=embedder,
            authenticated_user_id=user_id,
            config=config,
        )

    def _authorize(self, user_id: str) -> None:
        if not isinstance(user_id, str) or not user_id.strip():
            raise IdentityVerificationError("user_id is missing")
        if self._identity_verifier is not None:
            verified = self._identity_verifier()
            if verified != user_id:
                raise IdentityVerificationError(
                    "Authenticated identity does not match requested user_id"
                )
        if self._authenticated_user_id is not None and user_id != self._authenticated_user_id:
            raise IdentityVerificationError(
                "Authenticated identity does not match requested user_id"
            )

    def save_memory(
        self,
        user_id: str,
        content: str,
        approved: bool = False,
    ) -> Dict[str, Any]:
        self._authorize(user_id)
        if not approved:
            raise ApprovalRequiredError(
                "Saving memories requires explicit user approval "
                "(call with approved=True after the user confirms)"
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        content = content.strip()
        if len(content) > self.config.max_content_chars:
            raise ValueError(
                f"content exceeds maximum length of {self.config.max_content_chars} characters"
            )
        embedding = self.embedder.embed(content)
        return self.store.create(user_id, content, embedding)

    def retrieve_memories(
        self,
        user_id: str,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        self._authorize(user_id)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        k = top_k if top_k is not None else self.config.top_k
        k = max(1, min(int(k), self.config.max_top_k))
        query_embedding = self.embedder.embed(query)
        return self.store.search(user_id, query_embedding, k)

    def list_memories(
        self,
        user_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        self._authorize(user_id)
        n = limit if limit is not None else self.config.list_limit
        n = max(1, min(int(n), self.config.max_list_limit))
        return self.store.list_user(user_id, n)

    def delete_memory(self, user_id: str, memory_id: str) -> None:
        self._authorize(user_id)
        if not self.store.delete(user_id, memory_id):
            raise MemoryNotFoundError(
                f"Memory {memory_id!r} not found for this user"
            )

    def build_memory_context(
        self,
        user_id: str,
        query: str,
        top_k: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        memories = self.retrieve_memories(user_id, query, top_k=top_k)
        if not memories:
            return ""
        limit = max_chars if max_chars is not None else self.config.context_max_chars
        limit = max(100, int(limit))
        per_cap = self.config.per_memory_max_chars
        lines = []
        for m in memories:
            content = str(m.get("content", "")).strip()
            if len(content) > per_cap:
                content = content[: per_cap - 3].rstrip() + "..."
            created = str(m.get("created_at", "")).strip()[:10]
            sim = m.get("similarity")
            header = f"[memory {created}]" if created else "[memory]"
            if sim is not None:
                header += f" (similarity={float(sim):.2f})"
            lines.append(f"{header} {content}")
        text = "\n".join(lines)
        if len(text) > limit:
            text = text[: limit - 3].rstrip() + "..."
        return text
