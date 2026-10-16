"""Supabase authentication bridge for the API.

Memory endpoints require a verified Supabase JWT. The user id is NEVER
taken from the request body — it comes only from client.auth.get_user().
Without SUPABASE_URL/SUPABASE_ANON_KEY configured, memory endpoints
return 503 (not silently disabled with fake data).
"""
from __future__ import annotations

from typing import Optional

from .config import APIConfig


class AuthUnavailableError(Exception):
    pass


class AuthInvalidError(Exception):
    pass


class SupabaseAuthBridge:
    """Verifies bearer tokens against Supabase Auth; builds per-user
    MemoryService instances bound to the verified identity."""

    def __init__(self, config: APIConfig):
        if not config.memory_configured:
            raise AuthUnavailableError(
                "Supabase is not configured (set SUPABASE_URL and "
                "SUPABASE_ANON_KEY). Memory endpoints are unavailable."
            )
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from supabase import create_client
            except ImportError as exc:
                raise AuthUnavailableError(f"supabase package not installed: {exc}") from exc
            self._client = create_client(
                self.config.supabase_url, self.config.supabase_anon_key
            )
        return self._client

    def verify_token(self, bearer_token: str) -> str:
        """Return the verified user id for a JWT, or raise AuthInvalidError."""
        if not bearer_token or not bearer_token.strip():
            raise AuthInvalidError("missing bearer token")
        try:
            resp = self.client.auth.get_user(bearer_token.strip())
        except Exception as exc:
            raise AuthInvalidError(f"token verification failed: {exc}") from exc
        user = getattr(resp, "user", resp)
        user_id = getattr(user, "id", None) if user is not None else None
        if not user_id or not isinstance(user_id, str):
            raise AuthInvalidError("could not verify token identity")
        return user_id

    def memory_service_for(self, user_id: str):
        """Build a MemoryService bound to the verified user (RLS +
        app-level identity check both enforce ownership)."""
        from memory.config import MemoryConfig
        from memory.service import MemoryService
        from memory.supabase_store import SupabaseMemoryStore

        mem_cfg = MemoryConfig(
            supabase_url=self.config.supabase_url,
            supabase_anon_key=self.config.supabase_anon_key,
        )
        store = SupabaseMemoryStore(config=mem_cfg, client=self.client)
        # Identity is already verified; pass it as the authenticated id so
        # MemoryService._authorize rejects any mismatched user_id.
        return MemoryService(store=store, authenticated_user_id=user_id, config=mem_cfg)
