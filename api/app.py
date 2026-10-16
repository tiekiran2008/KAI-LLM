"""KAI-LLM V3 FastAPI backend (Phase 4).

Connects existing components without rewriting them:
- agents/  : LangGraph workflow, tools, LLMProvider interface
- rag/     : Retriever (ChromaDB + MiniLM), loaded lazily
- memory/  : MemoryService (Supabase), only after JWT verification

Endpoints:
  GET  /health            — service + model status (no model load)
  POST /chat              — agent chat (tools: calculator, doc search,
                            memory search if authenticated, file inspect)
  POST /documents/ingest  — upload .txt/.pdf, ingest into ChromaDB
  GET  /memories          — list verified user's memories (auth required)
  POST /memories          — save a memory (auth + explicit approved=true)
  DELETE /memories/{id}   — delete a memory (auth required)

Security:
- User id NEVER comes from the request body for memory ops; it comes from
  the Supabase JWT via SupabaseAuthBridge.
- Saving memories requires approved=true in the request (explicit user
  approval, requirement 10).
- Tool and document outputs are wrapped as untrusted data by agents/
  (requirement 11); this app never eval/exec's anything from them.
- KAI-LLM V3 is never claimed connected until LocalV3LLM successfully
  verifies the trained adapter (requirement 6: no silent base-model swap).

Run locally:
  uvicorn api.app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from agents import (
    AgentConfig,
    CalculatorTool,
    DocumentSearchTool,
    FileInspectTool,
    MemorySearchTool,
    ToolRegistry,
    run_agent,
)

from .config import APIConfig
from .llm_providers import V3AdapterError, build_llm
from .memory_bridge import AuthInvalidError, AuthUnavailableError, SupabaseAuthBridge


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    top_k: Optional[int] = Field(None, ge=1, le=10)


class ChatResponse(BaseModel):
    reply: str
    success: bool
    llm_mode: str
    steps: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    sources: List[str] = []
    memories_used: int = 0
    error: str = ""


class IngestResponse(BaseModel):
    chunks: int
    filename: str


class MemoryCreateRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=4000)
    approved: bool = False


class MemoryItem(BaseModel):
    id: Optional[str] = None
    content: str
    created_at: Optional[str] = None
    similarity: Optional[float] = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app(
    config: Optional[APIConfig] = None,
    llm: Any = None,
    retriever: Any = None,
    auth_bridge: Any = None,
) -> FastAPI:
    config = config or APIConfig.from_env()

    # LLM: injected (tests) or built from config. Build-time verification
    # only; the model itself loads lazily on first use.
    llm_info: Dict[str, Any] = {"mode": config.llm_mode, "verified": False, "detail": ""}
    if llm is None:
        try:
            llm = build_llm(config)
            if config.llm_mode == "local_adapter":
                info = llm.verify_adapter()
                llm_info["verified"] = True
                llm_info["detail"] = (
                    f"V3 adapter verified at {info['adapter_path']} "
                    f"(base={info['base_model']}, r={info['r']})"
                )
            else:
                llm_info["detail"] = f"{config.llm_mode} provider ready"
        except V3AdapterError as exc:
            # Do NOT fall back to the base model. Report clearly and use
            # mock ONLY if explicitly configured; otherwise fail fast.
            if config.llm_mode == "local_adapter":
                raise
            llm_info["detail"] = str(exc)
    else:
        llm_info["detail"] = "injected provider"

    app = FastAPI(title="KAI-LLM V3 API", version="0.4.0")
    app.state.config = config
    app.state.llm = llm
    app.state.llm_info = llm_info
    app.state.retriever = retriever  # None = lazy real, or injected fake
    app.state.auth_bridge = auth_bridge

    # Lazy real RAG retriever (ChromaDB + MiniLM) — created on first use.
    def get_retriever():
        if app.state.retriever is None:
            if not config.rag_enabled:
                raise HTTPException(status_code=503, detail="RAG is disabled")
            from rag import RAGConfig, Retriever

            rag_cfg = RAGConfig(persist_dir=config.rag_persist_dir)
            app.state.retriever = Retriever(rag_cfg)
        return app.state.retriever

    def get_auth_bridge():
        if app.state.auth_bridge is None:
            if not config.memory_configured:
                raise HTTPException(
                    status_code=503,
                    detail="Memory is not configured (set SUPABASE_URL and SUPABASE_ANON_KEY)",
                )
            app.state.auth_bridge = SupabaseAuthBridge(config)
        return app.state.auth_bridge

    def require_user(authorization: Optional[str] = Header(None)) -> str:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Bearer token required")
        token = authorization[7:]
        try:
            return get_auth_bridge().verify_token(token)
        except AuthInvalidError as exc:
            raise HTTPException(status_code=401, detail=str(exc))
        except AuthUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    # ------------------------------------------------------------------
    # /health
    # ------------------------------------------------------------------
    @app.get("/health")
    def health() -> Dict[str, Any]:
        doc_count = None
        rag_error = ""
        if app.state.retriever is not None:
            try:
                doc_count = app.state.retriever.count()
            except Exception as exc:
                rag_error = str(exc)
        # V3 may only be claimed "connected" after a REAL inference request
        # has succeeded with the trained adapter — verification alone is
        # not enough (Phase 4 honesty rule).
        v3_inferred = bool(getattr(llm, "inference_succeeded", False))
        return {
            "status": "ok",
            "llm_mode": llm_info["mode"],
            "llm_detail": llm_info["detail"],
            "v3_adapter_verified": llm_info["verified"],
            "v3_claimed_connected": bool(
                llm_info["verified"]
                and config.llm_mode == "local_adapter"
                and v3_inferred
            ),
            "rag_enabled": config.rag_enabled,
            "rag_documents": doc_count,
            "rag_error": rag_error,
            "memory_configured": config.memory_configured,
            "workspace_dir": config.workspace_dir,
        }

    # ------------------------------------------------------------------
    # /chat
    # ------------------------------------------------------------------
    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest, authorization: Optional[str] = Header(None)) -> ChatResponse:
        # Build a per-request tool registry (lightweight objects only).
        registry = ToolRegistry()
        registry.register(CalculatorTool())

        user_id: Optional[str] = None
        memories_used = 0
        if authorization and authorization.lower().startswith("bearer "):
            try:
                user_id = get_auth_bridge().verify_token(authorization[7:])
            except (AuthInvalidError, AuthUnavailableError):
                user_id = None  # chat still works; memory tool just omitted

        # Register document search when RAG is enabled OR a retriever was
        # injected (tests / custom wiring).
        if config.rag_enabled or app.state.retriever is not None:
            try:
                registry.register(DocumentSearchTool(retriever=get_retriever()))
            except HTTPException:
                pass

        if user_id and config.memory_configured:
            try:
                svc = get_auth_bridge().memory_service_for(user_id)
                registry.register(MemorySearchTool(memory_service=svc, user_id=user_id))
            except Exception:
                pass

        if config.workspace_dir:
            registry.register(
                FileInspectTool(AgentConfig(workspace_dir=config.workspace_dir))
            )

        agent_cfg = AgentConfig(
            max_tool_calls=config.max_tool_calls,
            workspace_dir=config.workspace_dir,
        )

        result = run_agent(
            req.message,
            llm=app.state.llm,
            registry=registry,
            config=agent_cfg,
            user_id=user_id,
            approval_callback=None,  # no mutating tools registered; deny-all
        )

        # Collect light metadata from the tool call log. Tool outputs
        # themselves are untrusted data and never interpreted here.
        sources: List[str] = []
        memories_used = sum(1 for c in result.tool_calls if c.get("tool") == "memory_search")

        return ChatResponse(
            reply=result.response,
            success=result.success,
            llm_mode=llm_info["mode"],
            steps=result.steps,
            tool_calls=result.tool_calls,
            sources=sources,
            memories_used=memories_used,
            error=result.error,
        )

    # ------------------------------------------------------------------
    # /documents/ingest
    # ------------------------------------------------------------------
    @app.post("/documents/ingest", response_model=IngestResponse)
    async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
        if not config.rag_enabled:
            raise HTTPException(status_code=503, detail="RAG is disabled")
        suffix = Path(file.filename or "upload.txt").suffix.lower()
        if suffix not in (".txt", ".pdf"):
            raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported")
        raw = await file.read()
        if len(raw) > config.max_upload_bytes:
            raise HTTPException(status_code=413, detail="File too large (max 5 MB)")
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / f"upload{suffix}"
            tmp_path.write_bytes(raw)
            retriever = get_retriever()
            try:
                chunks = retriever.ingest(str(tmp_path))
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"ingestion failed: {exc}") from exc
        return IngestResponse(chunks=chunks, filename=file.filename or f"upload{suffix}")

    # ------------------------------------------------------------------
    # /memories (auth required; explicit approval to save)
    # ------------------------------------------------------------------
    @app.get("/memories", response_model=List[MemoryItem])
    def list_memories(user_id: str = Depends(require_user)):
        svc = get_auth_bridge().memory_service_for(user_id)
        rows = svc.list_memories(user_id)
        return [MemoryItem(id=r.get("id"), content=r.get("content", ""), created_at=r.get("created_at")) for r in rows]

    @app.post("/memories", response_model=MemoryItem, status_code=201)
    def create_memory(req: MemoryCreateRequest, user_id: str = Depends(require_user)):
        if not req.approved:
            raise HTTPException(
                status_code=403,
                detail="Saving a memory requires explicit user approval (approved=true)",
            )
        svc = get_auth_bridge().memory_service_for(user_id)
        try:
            rec = svc.save_memory(user_id, req.content, approved=True)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return MemoryItem(id=rec.get("id"), content=rec.get("content", ""), created_at=rec.get("created_at"))

    @app.delete("/memories/{memory_id}", status_code=204)
    def delete_memory(memory_id: str, user_id: str = Depends(require_user)):
        from memory.exceptions import MemoryNotFoundError

        svc = get_auth_bridge().memory_service_for(user_id)
        try:
            svc.delete_memory(user_id, memory_id)
        except MemoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


# Default ASGI entrypoint (reads env at import time).
app = create_app()
