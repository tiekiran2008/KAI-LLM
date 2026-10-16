"""Agent tools: calculator, document search, memory search, file inspect.

Hard rules enforced here (requirements 7-9):
- Calculator uses a whitelisted AST evaluator — never eval/exec.
- File inspection is READ-ONLY and confined to an approved workspace
  directory (no writes, no deletes, no shell, no absolute paths outside).
- There is NO tool for arbitrary Python, shell, or system commands, and
  none can be registered without an explicit approval gate.
- Document and memory outputs are returned as plain text; the agent
  wraps them as untrusted data (see safety.py).
"""
from __future__ import annotations

import ast
import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import AgentConfig
from .exceptions import ToolExecutionError, ToolValidationError


@dataclass
class ToolResult:
    success: bool
    data: str = ""
    error: str = ""
    source: str = ""
    raw: Any = None


class Tool(ABC):
    name: str = ""
    description: str = ""
    # JSON-ish schema used by the LLM prompt and by validate_args.
    args_schema: Dict[str, str] = {}
    # True for any tool that writes files, runs commands, or performs
    # external side effects. Such tools MUST set requires_approval=True
    # and will be refused by ToolRegistry otherwise.
    mutates_state: bool = False
    requires_approval: bool = False

    @abstractmethod
    def validate_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @abstractmethod
    def run(self, args: Dict[str, Any]) -> ToolResult:
        ...


# --------------------------------------------------------------------------
# Calculator (whitelisted AST; no eval/exec)
# --------------------------------------------------------------------------
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval_arithmetic(expression: str) -> float:
    """Evaluate a pure arithmetic expression via a whitelisted AST.

    Allowed: numbers, + - * / // % **, parentheses, unary +/-.
    Everything else (names, calls, attributes, comprehensions, ...)
    raises ToolValidationError. Exponent magnitude is capped.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ToolValidationError("expression must be a non-empty string")
    if len(expression) > 200:
        raise ToolValidationError("expression too long (max 200 chars)")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolValidationError(f"invalid arithmetic expression: {exc}") from exc

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ToolValidationError("exponent too large (max 100)")
            try:
                return _BIN_OPS[type(node.op)](left, right)
            except ZeroDivisionError as exc:
                raise ToolExecutionError("division by zero") from exc
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ToolValidationError(
            f"disallowed expression element: {type(node).__name__}"
        )

    return _eval(tree)


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate a basic arithmetic expression (numbers and + - * / // % ** only)."
    args_schema = {"expression": "str — arithmetic expression, e.g. '(2 + 3) * 4'"}

    def validate_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(args, dict):
            raise ToolValidationError("args must be an object")
        expr = args.get("expression")
        if not isinstance(expr, str) or not expr.strip():
            raise ToolValidationError("missing required argument: expression (str)")
        return {"expression": expr.strip()}

    def run(self, args: Dict[str, Any]) -> ToolResult:
        try:
            value = safe_eval_arithmetic(args["expression"])
        except (ToolValidationError, ToolExecutionError) as exc:
            return ToolResult(success=False, error=str(exc), source=self.name)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return ToolResult(success=True, data=str(value), source=self.name, raw=value)


# --------------------------------------------------------------------------
# Document search (wraps RAG retriever — injected, lazy)
# --------------------------------------------------------------------------
class DocumentSearchTool(Tool):
    name = "document_search"
    description = "Search the ingested project documents (RAG) and return the most relevant text chunks."
    args_schema = {"query": "str — what to search for", "top_k": "int (optional, default 4)"}

    def __init__(self, retriever: Any = None, top_k: int = 4):
        self._retriever = retriever
        self.top_k = top_k

    def _get_retriever(self) -> Any:
        if self._retriever is None:
            from rag import Retriever  # lazy: avoids loading embedding model at import

            self._retriever = Retriever()
        return self._retriever

    def validate_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(args, dict):
            raise ToolValidationError("args must be an object")
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolValidationError("missing required argument: query (str)")
        top_k = args.get("top_k", self.top_k)
        if not isinstance(top_k, int) or top_k < 1 or top_k > 10:
            raise ToolValidationError("top_k must be an int between 1 and 10")
        return {"query": query.strip(), "top_k": top_k}

    def run(self, args: Dict[str, Any]) -> ToolResult:
        try:
            results = self._get_retriever().retrieve(args["query"], top_k=args["top_k"])
        except Exception as exc:
            return ToolResult(success=False, error=f"document search failed: {exc}", source=self.name)
        if not results:
            return ToolResult(success=True, data="No matching documents found.", source=self.name, raw=[])
        lines = []
        for i, r in enumerate(results, 1):
            src = r.get("source", "unknown")
            text = (r.get("text") or "").strip()
            lines.append(f"[{i}] ({src}) {text}")
        return ToolResult(success=True, data="\n".join(lines), source=self.name, raw=results)


# --------------------------------------------------------------------------
# Memory search (wraps MemoryService — injected, lazy)
# --------------------------------------------------------------------------
class MemorySearchTool(Tool):
    name = "memory_search"
    description = "Search the user's long-term memories for facts saved in earlier conversations."
    args_schema = {"query": "str — what to look up in memory", "top_k": "int (optional, default 5)"}

    def __init__(self, memory_service: Any = None, user_id: Optional[str] = None, top_k: int = 5):
        self._memory_service = memory_service
        self.user_id = user_id
        self.top_k = top_k

    def _get_service(self) -> Any:
        if self._memory_service is None:
            raise ToolExecutionError("memory service not configured")
        return self._memory_service

    def validate_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(args, dict):
            raise ToolValidationError("args must be an object")
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolValidationError("missing required argument: query (str)")
        top_k = args.get("top_k", self.top_k)
        if not isinstance(top_k, int) or top_k < 1 or top_k > 10:
            raise ToolValidationError("top_k must be an int between 1 and 10")
        return {"query": query.strip(), "top_k": top_k}

    def run(self, args: Dict[str, Any]) -> ToolResult:
        if not self.user_id:
            return ToolResult(
                success=False,
                error="memory search requires an authenticated user id",
                source=self.name,
            )
        top_k = args.get("top_k", self.top_k)
        try:
            memories = self._get_service().retrieve_memories(
                self.user_id, args["query"], top_k=top_k
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"memory search failed: {exc}", source=self.name)
        if not memories:
            return ToolResult(success=True, data="No relevant memories found.", source=self.name, raw=[])
        lines = []
        for i, m in enumerate(memories, 1):
            created = str(m.get("created_at", ""))[:10]
            content = (m.get("content") or "").strip()
            sim = m.get("similarity")
            sim_s = f" sim={float(sim):.2f}" if sim is not None else ""
            lines.append(f"[{i}] ({created}{sim_s}) {content}")
        return ToolResult(success=True, data="\n".join(lines), source=self.name, raw=memories)


# --------------------------------------------------------------------------
# File inspection (read-only, workspace-confined)
# --------------------------------------------------------------------------
class FileInspectTool(Tool):
    name = "file_inspect"
    description = (
        "Inspect files inside the approved workspace directory. "
        "Actions: 'list' a directory or 'read' a text file. Read-only."
    )
    args_schema = {
        "action": "'list' or 'read'",
        "path": "str — relative path inside the workspace (default '.')",
    }

    def __init__(self, config: AgentConfig):
        self.config = config

    def _resolve(self, rel_path: str) -> Path:
        workspace = self.config.resolved_workspace()
        if workspace is None:
            raise ToolExecutionError("file inspection is disabled (no workspace_dir configured)")
        if not isinstance(rel_path, str) or not rel_path.strip():
            rel_path = "."
        rel = rel_path.strip()
        if Path(rel).is_absolute():
            raise ToolValidationError("path must be relative to the workspace")
        candidate = (workspace / rel).resolve()
        try:
            candidate.relative_to(workspace)
        except ValueError as exc:
            raise ToolValidationError("path escapes the approved workspace directory") from exc
        return candidate

    def validate_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(args, dict):
            raise ToolValidationError("args must be an object")
        action = args.get("action", "list")
        if action not in ("list", "read"):
            raise ToolValidationError("action must be 'list' or 'read'")
        path = args.get("path", ".")
        if not isinstance(path, str):
            raise ToolValidationError("path must be a str")
        return {"action": action, "path": path}

    def run(self, args: Dict[str, Any]) -> ToolResult:
        try:
            target = self._resolve(args.get("path", "."))
        except (ToolValidationError, ToolExecutionError) as exc:
            return ToolResult(success=False, error=str(exc), source=self.name)
        action = args["action"]
        if action == "list":
            return self._list(target)
        return self._read(target)

    def _list(self, target: Path) -> ToolResult:
        if not target.exists():
            return ToolResult(success=False, error="path does not exist", source=self.name)
        if not target.is_dir():
            return ToolResult(success=False, error="path is not a directory", source=self.name)
        entries = sorted(target.iterdir())[: self.config.max_dir_entries]
        lines = []
        for e in entries:
            suffix = "/" if e.is_dir() else ""
            lines.append(e.name + suffix)
        if not lines:
            return ToolResult(success=True, data="(empty directory)", source=self.name)
        return ToolResult(success=True, data="\n".join(lines), source=self.name, raw=lines)

    def _read(self, target: Path) -> ToolResult:
        if not target.exists():
            return ToolResult(success=False, error="file does not exist", source=self.name)
        if not target.is_file():
            return ToolResult(success=False, error="path is not a file", source=self.name)
        try:
            data = target.read_bytes()[: self.config.max_file_read_bytes]
            text = data.decode("utf-8", errors="replace")
        except OSError as exc:
            return ToolResult(success=False, error=f"could not read file: {exc}", source=self.name)
        return ToolResult(success=True, data=text, source=self.name)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
class ToolRegistry:
    def __init__(self, tools: Optional[List[Tool]] = None):
        self._tools: Dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if not getattr(tool, "name", ""):
            raise ToolValidationError("tool must have a name")
        # Refuse tools that claim write/shell capability unless they
        # explicitly declare an approval requirement (defense in depth:
        # no such tools exist in this package, and none should be added
        # without an approval workflow).
        if getattr(tool, "mutates_state", False) and not getattr(tool, "requires_approval", False):
            raise ToolValidationError(
                f"tool '{tool.name}' mutates state but does not require approval"
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return sorted(self._tools)

    def describe(self) -> str:
        lines = []
        for name in self.names():
            t = self._tools[name]
            args = ", ".join(f"{k}: {v}" for k, v in t.args_schema.items())
            lines.append(f"- {name}: {t.description} Args: {{{args}}}")
        return "\n".join(lines)
