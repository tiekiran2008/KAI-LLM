"""KAI-LLM long-term memory (Phase 2).

Separate from RAG: memory stores user facts with explicit approval and
retrieves them with semantic search (sentence-transformers + pgvector RPC
when available, client-side cosine fallback otherwise).
"""
from .config import MemoryConfig
from .embeddings import MemoryEmbedder, cosine_similarity
from .exceptions import (
    ApprovalRequiredError,
    IdentityVerificationError,
    MemoryConfigError,
    MemoryError,
    MemoryNotFoundError,
)
from .service import MemoryService
from .store import InMemoryStore, MemoryStore
from .supabase_store import SupabaseMemoryStore

__all__ = [
    "MemoryConfig",
    "MemoryEmbedder",
    "cosine_similarity",
    "ApprovalRequiredError",
    "IdentityVerificationError",
    "MemoryConfigError",
    "MemoryError",
    "MemoryNotFoundError",
    "MemoryService",
    "InMemoryStore",
    "MemoryStore",
    "SupabaseMemoryStore",
]
