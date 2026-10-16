import warnings
from pathlib import Path

from .config import PROJECT_ROOT, RAGConfig

SYSTEM_PROMPT = (
    "You are KAI-LLM, a domain-specialized AI assistant for AI/ML engineering, "
    "software architecture, and high-quality code generation."
)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER_PATH = str(PROJECT_ROOT / "qlora_output_kai")

_model_cache = {"model": None, "tokenizer": None}


def build_context(chunks: list, max_chars: int = 3000) -> str:
    if not chunks:
        return ""
    parts = []
    used = 0
    for i, chunk in enumerate(chunks, 1):
        block = f"[{i}] (source: {chunk['source']})\n{chunk['text']}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


def build_kai_prompt(query: str, context: str) -> list:
    if context:
        user_content = (
            f"Use the following retrieved context to answer the question. "
            f"If the context does not contain the answer, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
    else:
        user_content = query
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _load_kai_model(base_model: str = DEFAULT_BASE_MODEL, adapter_path: str = None):
    if _model_cache["model"] is not None:
        return _model_cache["model"], _model_cache["tokenizer"]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = adapter_path or DEFAULT_ADAPTER_PATH
    has_adapter = Path(adapter_path).is_dir() and (
        Path(adapter_path) / "adapter_config.json"
    ).is_file()

    if not has_adapter:
        warnings.warn(
            f"LoRA adapter not found at '{adapter_path}'. "
            f"Falling back to base model '{base_model}'. "
            f"Copy the Colab-trained qlora_output_kai folder to the project root "
            f"to enable the fine-tuned model.",
            stacklevel=2,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if has_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    _model_cache["model"] = model
    _model_cache["tokenizer"] = tokenizer
    return model, tokenizer


def generate_with_rag(
    query: str,
    retriever,
    base_model: str = DEFAULT_BASE_MODEL,
    adapter_path: str = None,
    top_k: int = None,
    max_new_tokens: int = 512,
) -> dict:
    chunks = retriever.retrieve(query, top_k=top_k)
    config: RAGConfig = retriever.config
    context = build_context(chunks, max_chars=config.max_chars_prompt)
    messages = build_kai_prompt(query, context)

    model, tokenizer = _load_kai_model(base_model, adapter_path)

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with __import__("torch").no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)

    return {
        "answer": answer,
        "context": context,
        "sources": [c["source"] for c in chunks],
        "chunks": chunks,
    }
