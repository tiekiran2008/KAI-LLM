"""API configuration.

Verified against training/train_qlora.py (Phase 4 inspection):
- BASE_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
- OUTPUT_DIR  = "./qlora_output_kai"   (LoRA adapter; trained on Colab)
- LoRA        = r 64, alpha 16, dropout 0.1,
                targets q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj

The adapter directory is NOT present on this machine. Real V3 inference
stays disabled until the Colab-trained qlora_output_kai folder is copied
here — the API NEVER silently substitutes the base Qwen model (req. 6).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Verified from training/train_qlora.py — do not change without retraining.
KAI_V3_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
KAI_V3_ADAPTER_PATH = str(PROJECT_ROOT / "qlora_output_kai")


def _load_dotenv():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass
class APIConfig:
    # "mock" | "openai_endpoint" | "local_adapter"
    llm_mode: str = "mock"

    # openai_endpoint settings (any OpenAI-compatible server: Ollama, vLLM, ...)
    endpoint_base_url: str = ""
    endpoint_model: str = ""
    endpoint_api_key: str = field(default="", repr=False)
    endpoint_timeout: float = 30.0

    # local_adapter (KAI-LLM V3) — strict, no silent fallback
    base_model: str = KAI_V3_BASE_MODEL
    adapter_path: str = KAI_V3_ADAPTER_PATH

    # RAG
    rag_enabled: bool = True
    rag_persist_dir: str = field(
        default_factory=lambda: str(PROJECT_ROOT / "rag" / "chroma_db")
    )

    # File inspection workspace (None = tool disabled)
    workspace_dir: Optional[str] = None

    # Supabase memory (optional; memory endpoints 503 when unset)
    supabase_url: str = ""
    supabase_anon_key: str = field(default="", repr=False)

    # Agent limits
    max_tool_calls: int = 3
    max_upload_bytes: int = 5 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "APIConfig":
        _load_dotenv()
        mode = _env("KAI_API_MODE", "mock").lower()
        if mode not in ("mock", "openai_endpoint", "local_adapter"):
            mode = "mock"
        rag = _env("KAI_RAG_ENABLED", "1").lower()
        return cls(
            llm_mode=mode,
            endpoint_base_url=_env("KAI_ENDPOINT_BASE_URL"),
            endpoint_model=_env("KAI_ENDPOINT_MODEL"),
            endpoint_api_key=_env("KAI_ENDPOINT_API_KEY"),
            endpoint_timeout=float(_env("KAI_ENDPOINT_TIMEOUT", "30") or 30),
            adapter_path=_env("KAI_ADAPTER_PATH", KAI_V3_ADAPTER_PATH),
            rag_enabled=rag not in ("0", "false", "no"),
            rag_persist_dir=_env("KAI_RAG_PERSIST_DIR", str(PROJECT_ROOT / "rag" / "chroma_db")),
            workspace_dir=_env("KAI_WORKSPACE_DIR") or None,
            supabase_url=_env("SUPABASE_URL"),
            supabase_anon_key=_env("SUPABASE_ANON_KEY"),
        )

    def adapter_available(self) -> bool:
        p = Path(self.adapter_path)
        return p.is_dir() and (p / "adapter_config.json").is_file()

    @property
    def memory_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_anon_key)
