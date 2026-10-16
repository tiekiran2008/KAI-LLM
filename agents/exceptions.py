class AgentError(Exception):
    pass


class ToolValidationError(AgentError):
    pass


class ToolExecutionError(AgentError):
    pass


class ToolTimeoutError(ToolExecutionError):
    pass


class ApprovalRequiredError(AgentError):
    pass


class LoopLimitExceededError(AgentError):
    pass


class LLMError(AgentError):
    pass
