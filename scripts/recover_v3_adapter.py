"""Phase 5A: download ONLY the required KAI-LLM V3 adapter files.

Downloads adapter_config.json + adapter_model.safetensors from
kiran2008/KAI-LLM-v3 into D:\\KAI-LLM\\qlora_output_kai.
Does NOT download checkpoints, the base model, or tokenizer extras.
Does NOT overwrite existing files without verification.
"""
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "kiran2008/KAI-LLM-v3"
DEST = Path(r"D:\KAI-LLM\qlora_output_kai")
FILES = ["adapter_config.json", "adapter_model.safetensors"]
# sha256 from the HF tree API (LFS oid) for integrity verification.
EXPECTED_SHA256 = {
    "adapter_model.safetensors": "9598c5856566cfa2b8ce807bf3465d04013341e811bf90946fa691c1b7258d14",
}

DEST.mkdir(parents=True, exist_ok=True)

for fname in FILES:
    target = DEST / fname
    if target.exists():
        print(f"SKIP (exists): {target}")
    else:
        print(f"Downloading {fname} ...")
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=fname,
            local_dir=str(DEST),
            repo_type="model",
        )
        print(f"  -> {path}")

print("\n--- integrity check ---")
cfg = json.loads((DEST / "adapter_config.json").read_text(encoding="utf-8"))
print("base_model_name_or_path:", cfg.get("base_model_name_or_path"))
print("r:", cfg.get("r"), "lora_alpha:", cfg.get("lora_alpha"))
print("target_modules:", cfg.get("target_modules"))
print("peft_type:", cfg.get("peft_type"), "task_type:", cfg.get("task_type"))

weight_file = DEST / "adapter_model.safetensors"
h = hashlib.sha256()
with weight_file.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
digest = h.hexdigest()
print(f"adapter_model.safetensors size: {weight_file.stat().st_size} bytes")
print(f"sha256: {digest}")
expected = EXPECTED_SHA256.get("adapter_model.safetensors")
if expected:
    print("sha256 matches HF LFS oid:", digest == expected)
    if digest != expected:
        raise SystemExit("INTEGRITY FAILURE: sha256 mismatch")
print("OK")
