"""Identity verification for memory operations.

Never trust a user ID passed by the caller alone for authorization. Identity
must come from an authenticated Supabase session (the JWT is verified server-
side by Supabase Auth). Application-level checks plus database RLS both
enforce ownership.
"""
from __future__ import annotations

from .exceptions import IdentityVerificationError


def verify_authenticated_user(client) -> str:
    """Return the user id of the currently authenticated Supabase session.

    Raises IdentityVerificationError if there is no valid session.
    """
    if client is None or getattr(client, "auth", None) is None:
        raise IdentityVerificationError("Supabase client is missing an auth object")
    try:
        resp = client.auth.get_user()
    except Exception as exc:
        raise IdentityVerificationError(
            "No authenticated Supabase session. Sign in before using memory."
        ) from exc
    user = getattr(resp, "user", resp)
    user_id = None
    if user is not None:
        user_id = getattr(user, "id", None)
        if user_id is None and isinstance(user, dict):
            user_id = user.get("id")
    if not user_id or not isinstance(user_id, str):
        raise IdentityVerificationError(
            "Could not verify authenticated user identity"
        )
    return user_id
