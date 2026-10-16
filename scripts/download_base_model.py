"""Phase 5B: download Qwen/Qwen2.5-1.5B-Instruct base model to a dedicated
cache on D: without touching the verified V3 adapter at qlora_output_kai."""
import os
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEST = Path(r"D:\KAI-LLM\model_cache\Qwen2.5-1.5B-Instruct")
ADAPTER_DIR = Path(r"D:\KAI-LLM\qlora_output_kai")

# Keep the adapter untouched: download only base-model files, never adapter files.
ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
]
IGNORE_PATTERNS = ["*.pt", "*.bin", "*.msgpack", "*.h5"]

if ADAPTER_DIR.exists():
    adapter_before = sorted(p.name for p in ADAPTER_DIR.iterdir())
    print("Adapter files before download:", adapter_before)
else:
    raise SystemExit("Adapter dir missing — aborting.")

DEST.mkdir(parents=True, exist_ok=True)
print(f"Downloading {REPO_ID} -> {DEST}")
path = snapshot_download(
    repo_id=REPO_ID,
    local_dir=str(DEST),
    allow_patterns=ALLOW_PATTERNS,
    ignore_patterns=IGNORE_PATTERNS,
)
print("Downloaded to:", path)

print("\n--- downloaded files ---")
total = 0
for f in sorted(DEST.rglob("*")):
    if f.is_file():
        size = f.stat().st_size
        total += size
        print(f"  {f.relative_to(DEST)}  {size:,} bytes")
print(f"Total: {total:,} bytes ({total/1024**3:.2f} GB)")

adapter_after = sorted(p.name for p in ADAPTER_DIR.iterdir()) if ADAPTER_DIR.exists() else []
print("\nAdapter files after download:", adapter_after)
assert adapter_before == adapter_after, "Adapter directory was modified!"
print("Adapter directory unchanged: OK")
