"""LLM providers for the KAI API.

Uses the existing agents.llm.LLMProvider interface (requirement 4).

KAI-LLM V3 = Qwen/Qwen2.5-1.5B-Instruct + LoRA adapter at ./qlora_output_kai
(verified from training/train_qlora.py). The adapter is NOT present on this
machine (trained on Colab). LocalV3LLM therefore:
  - refuses to load when the adapter is missing (NO silent base-model
    substitution, requirement 6),
  - reports adapter verification (base_model in adapter_config.json must
    match the training base model),
  - loads the model lazily on first complete() call.

For unit/integration tests use MockLLM (from agents.llm) or
KAIMockInferenceLLM below — KAI-LLM V3 is never loaded during tests.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from agents.llm import LLMProvider, LLMUnavailableError, MockLLM, OpenAICompatibleLLM

from .config import APIConfig, KAI_V3_BASE_MODEL


class V3AdapterError(Exception):
    """Raised when the KAI-LLM V3 adapter cannot be verified/loaded."""


class LocalV3LLM(LLMProvider):
    """Local inference with KAI-LLM V3 (Qwen2.5-1.5B-Instruct + QLoRA adapter).

    Strict: if the adapter directory or adapter_config.json is missing, or
    the adapter's base model does not match the training base model, this
    raises V3AdapterError. It never silently loads the base model alone.
    """

    def __init__(self, config: APIConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self.verified = False
        # Becomes True ONLY after a real inference request succeeds with
        # the trained adapter (Phase 4 honesty rule): verification alone
        # must never be reported as "connected".
        self.inference_succeeded = False
        self.verify_adapter()

    def verify_adapter(self) -> dict:
        """Verify adapter presence and base-model match (no model load)."""
        adapter_path = Path(self.config.adapter_path)
        if not adapter_path.is_dir():
            raise V3AdapterError(
                f"KAI-LLM V3 LoRA adapter not found at '{adapter_path}'. "
                "Copy the Colab-trained qlora_output_kai folder to the project "
                "root. The base Qwen model will NOT be substituted silently."
            )
        cfg_file = adapter_path / "adapter_config.json"
        if not cfg_file.is_file():
            raise V3AdapterError(
                f"adapter_config.json missing in '{adapter_path}'. "
                "This does not look like a trained QLoRA output."
            )
        try:
            adapter_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise V3AdapterError(f"could not read adapter_config.json: {exc}") from exc

        declared_base = adapter_cfg.get("base_model_name_or_path", "")
        expected = self.config.base_model
        if declared_base and declared_base != expected:
            raise V3AdapterError(
                f"adapter was trained on '{declared_base}' but config expects "
                f"'{expected}'. Refusing to load a mismatched adapter."
            )
        self.verified = True
        return {
            "adapter_path": str(adapter_path),
            "base_model": declared_base or expected,
            "r": adapter_cfg.get("r"),
            "lora_alpha": adapter_cfg.get("lora_alpha"),
            "target_modules": adapter_cfg.get("target_modules"),
        }

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as exc:
            raise LLMUnavailableError(
                f"torch/transformers/peft required for local V3 inference: {exc}"
            ) from exc

        base = self.config.base_model
        adapter = self.config.adapter_path
        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
        model.eval()
        self._model = model
        self._tokenizer = tokenizer

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        self._ensure_loaded()
        import torch

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        prompt_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        # A real inference request has now succeeded with the trained
        # adapter — only now may /health claim V3 is connected.
        self.inference_succeeded = True
        return text


class KAIMockInferenceLLM(LLMProvider):
    """Deterministic mock for integration tests: answers simple arithmetic
    and echoes a marker so tests can assert the full pipeline ran."""

    _ARITH = __import__("re").compile(r"(\d+)\s*([-+*/])\s*(\d+)(?:\s*([-+*/])\s*(\d+))?")

    def __init__(self):
        self.calls = []

    @staticmethod
    def _full_expression(prompt: str) -> Optional[str]:
        """Extract a 2- or 4-operand arithmetic snippet like '12 * 8 + 4'."""
        import re

        m = re.search(r"\d+\s*[-+*/]\s*\d+(?:\s*[-+*/]\s*\d+)*", prompt)
        return m.group(0) if m else None

    def complete(self, prompt, system=None, max_tokens=512, temperature=0.2) -> str:
        import json as _json
        import re

        self.calls.append({"prompt": prompt, "system": system})
        lower = (prompt or "").lower()
        # Tool-selection stage: pick calculator for arithmetic requests.
        if "json decision" in lower or '"tool"' in lower[:200] and "user request" in lower:
            expr = self._full_expression(prompt)
            if expr:
                return _json.dumps({"tool": "calculator", "args": {"expression": expr}})
            if "search" in lower and "document" in lower:
                return _json.dumps({"tool": "document_search", "args": {"query": prompt}})
            if "memor" in lower:
                return _json.dumps({"tool": "memory_search", "args": {"query": prompt}})
            return _json.dumps({"tool": None, "args": {}})
        # Generation stage: if the calculator result is in the prompt,
        # echo it; else canned reply.
        expr = self._full_expression(prompt)
        if expr:
            try:
                from agents import safe_eval_arithmetic

                val = safe_eval_arithmetic(expr)
                if isinstance(val, float) and not float(val).is_integer():
                    return f"The answer is {val}."
                return f"The answer is {int(val)}."
            except Exception:
                pass
        return "KAI mock response: request understood (mock inference)."


def build_llm(config: APIConfig) -> LLMProvider:
    """Build the configured LLM provider. Never silently substitutes the
    base Qwen model for the V3 adapter."""
    mode = config.llm_mode
    if mode == "mock":
        return KAIMockInferenceLLM()
    if mode == "openai_endpoint":
        if not config.endpoint_base_url or not config.endpoint_model:
            raise V3AdapterError(
                "KAI_API_MODE=openai_endpoint requires KAI_ENDPOINT_BASE_URL "
                "and KAI_ENDPOINT_MODEL"
            )
        return OpenAICompatibleLLM(
            base_url=config.endpoint_base_url,
            model=config.endpoint_model,
            api_key=config.endpoint_api_key or None,
            timeout=config.endpoint_timeout,
        )
    if mode == "local_adapter":
        return LocalV3LLM(config)
    raise V3AdapterError(f"unknown KAI_API_MODE: {mode!r}")


__all__ = [
    "V3AdapterError",
    "LocalV3LLM",
    "KAIMockInferenceLLM",
    "build_llm",
    "MockLLM",
    "OpenAICompatibleLLM",
]
