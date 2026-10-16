import os
from dataclasses import dataclass, field
from pathlib import Path

from .exceptions import MemoryConfigError


def _load_dotenv():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


@dataclass
class MemoryConfig:
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = field(default="", repr=False)

    table_name: str = "memories"
    match_function: str = "match_memories"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    top_k: int = 5
    max_top_k: int = 10
    list_limit: int = 50
    max_list_limit: int = 100
    context_max_chars: int = 1200
    per_memory_max_chars: int = 300
    max_content_chars: int = 2000
    fallback_scan_limit: int = 200

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        _load_dotenv()
        return cls(
            supabase_url=_env("SUPABASE_URL"),
            supabase_anon_key=_env("SUPABASE_ANON_KEY"),
            supabase_service_role_key=_env("SUPABASE_SERVICE_ROLE_KEY"),
        )

    def __post_init__(self):
        self.top_k = max(1, min(self.top_k, self.max_top_k))
        self.list_limit = max(1, min(self.list_limit, self.max_list_limit))
        if self.context_max_chars < 100:
            self.context_max_chars = 100
        if self.per_memory_max_chars < 50:
            self.per_memory_max_chars = 50

    @property
    def has_credentials(self) -> bool:
        return bool(self.supabase_url and self.supabase_anon_key)

    def require_credentials(self):
        if not self.has_credentials:
            raise MemoryConfigError(
                "Missing Supabase credentials. Set SUPABASE_URL and "
                "SUPABASE_ANON_KEY as environment variables (e.g. in a "
                ".env file). The service-role key must never be required "
                "for user-scoped memory operations."
            )

    def __repr__(self) -> str:
        return (
            "MemoryConfig("
            f"supabase_url={self.supabase_url!r}, "
            f"supabase_anon_key={'<set>' if self.supabase_anon_key else '<unset>'}, "
            f"supabase_service_role_key={'<set>' if self.supabase_service_role_key else '<unset>'}, "
            f"table_name={self.table_name!r}, top_k={self.top_k}, "
            f"max_top_k={self.max_top_k}, context_max_chars={self.context_max_chars})"
        )
