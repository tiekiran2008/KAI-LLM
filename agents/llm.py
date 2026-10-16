"""Provider-independent LLM interface for the KAI agent.

Design goals:
- The agent never calls a model-specific API directly; it only uses
  LLMProvider.complete(prompt) -> str.
- Native tool-calling is NOT required. Tool selection is done with a
  structured JSON protocol (see agent.py) that any chat model can follow,
  plus a deterministic fallback when the model is unavailable.
- KAI-LLM V3 (Qwen2.5-1.5B-Instruct + LoRA) can be used through ANY
  OpenAI-compatible inference endpoint (Ollama, vLLM, llama.cpp server,
  LM Studio, a remote GPU box, etc.) via OpenAICompatibleLLM.
- MockLLM enables fully offline unit tests — KAI-LLM V3 is never loaded
  during tests (requirement).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence, Union


class LLMProvider(ABC):
    """Minimal chat-completion interface (no tool-calling needed)."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        ...


class MockLLM(LLMProvider):
    """Scripted LLM for tests and offline demos.

    responses: list of strings returned in order (last one repeats), or a
    callable(prompt, system) -> str for dynamic behavior.
    """

    def __init__(
        self,
        responses: Optional[Union[Sequence[str], Callable[[str, Optional[str]], str]]] = None,
    ):
        self._responses: List[str] = list(responses) if isinstance(responses, (list, tuple)) else []
        self._callable = responses if callable(responses) else None
        self.calls: List[dict] = []

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        self.calls.append({"prompt": prompt, "system": system})
        if self._callable is not None:
            return str(self._callable(prompt, system))
        if not self._responses:
            raise LLMUnavailableError("MockLLM has no scripted responses left")
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)


class LLMUnavailableError(Exception):
    pass


class OpenAICompatibleLLM(LLMProvider):
    """Talks to any OpenAI-compatible /chat/completions endpoint.

    Works with Ollama (http://localhost:11434/v1), vLLM, llama.cpp server,
    LM Studio, or a remote inference host serving KAI-LLM V3.
    Uses only the standard library (no openai package required).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        if not base_url:
            raise ValueError("base_url is required")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMUnavailableError(f"LLM endpoint unavailable: {exc}") from exc
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailableError(f"Unexpected endpoint response shape: {exc}") from exc
