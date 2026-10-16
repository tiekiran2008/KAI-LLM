"""Agent configuration with hard safety limits."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AgentConfig:
    # Loop / budget limits (requirement: prevent infinite agent loops)
    max_tool_calls: int = 3
    max_retries: int = 1
    tool_timeout_seconds: float = 15.0
    llm_timeout_seconds: float = 30.0

    # Tool output limits
    max_tool_output_chars: int = 4000
    max_file_read_bytes: int = 64 * 1024
    max_dir_entries: int = 200

    # Read-only file inspection restricted to this directory.
    # None means the file tool is disabled.
    workspace_dir: Optional[str] = None

    # Explicit approval required for ANY action that modifies files,
    # runs terminal commands, or performs external side effects.
    # The agent never grants these by default.
    approved_actions: tuple = ()

    # Deterministic fallback when the LLM cannot be reached or
    # returns unparseable structured output.
    enable_deterministic_fallback: bool = True

    # Maximum characters of tool output passed to the LLM in one turn
    max_context_chars: int = 6000

    def resolved_workspace(self) -> Optional[Path]:
        if not self.workspace_dir:
            return None
        return Path(self.workspace_dir).resolve()
