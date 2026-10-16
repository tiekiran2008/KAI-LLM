"""KAI-LLM agents (Phase 3): LangGraph workflow + validated tools."""
from .agent import AgentResult, KAIWorkflow, run_agent
from .config import AgentConfig
from .exceptions import (
    AgentError,
    ApprovalRequiredError,
    LLMError,
    LoopLimitExceededError,
    ToolExecutionError,
    ToolTimeoutError,
    ToolValidationError,
)
from .llm import LLMProvider, LLMUnavailableError, MockLLM, OpenAICompatibleLLM
from .safety import looks_like_injection, wrap_untrusted
from .tools import (
    CalculatorTool,
    DocumentSearchTool,
    FileInspectTool,
    MemorySearchTool,
    Tool,
    ToolRegistry,
    ToolResult,
    safe_eval_arithmetic,
)

__all__ = [
    "AgentResult",
    "KAIWorkflow",
    "run_agent",
    "AgentConfig",
    "AgentError",
    "ApprovalRequiredError",
    "LLMError",
    "LoopLimitExceededError",
    "ToolExecutionError",
    "ToolTimeoutError",
    "ToolValidationError",
    "LLMProvider",
    "LLMUnavailableError",
    "MockLLM",
    "OpenAICompatibleLLM",
    "looks_like_injection",
    "wrap_untrusted",
    "CalculatorTool",
    "DocumentSearchTool",
    "FileInspectTool",
    "MemorySearchTool",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "safe_eval_arithmetic",
]
