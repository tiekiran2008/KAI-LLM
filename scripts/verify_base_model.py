"""Phase 5B step 3: verify downloaded base model files and compatibility
with the V3 LoRA adapter. Uses safetensors header inspection only — does
NOT load model weights into memory."""
import json
import struct
from pathlib import Path

MODEL_DIR = Path(r"D:\KAI-LLM\model_cache\Qwen2.5-1.5B-Instruct")
ADAPTER_DIR = Path(r"D:\KAI-LLM\qlora_output_kai")

print("=== base model config ===")
cfg = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
print("architectures:", cfg.get("architectures"))
print("model_type:", cfg.get("model_type"))
print("torch_dtype:", cfg.get("torch_dtype"))
print("num_hidden_layers:", cfg.get("num_hidden_layers"))
print("hidden_size:", cfg.get("hidden_size"))
print("vocab_size:", cfg.get("vocab_size"))
print("num_attention_heads:", cfg.get("num_attention_heads"))

print("\n=== adapter config ===")
adapter_cfg = json.loads((ADAPTER_DIR / "adapter_config.json").read_text(encoding="utf-8"))
base_from_adapter = adapter_cfg.get("base_model_name_or_path")
print("base_model_name_or_path:", base_from_adapter)
assert base_from_adapter == "Qwen/Qwen2.5-1.5B-Instruct", "Base model name mismatch!"
print("r:", adapter_cfg.get("r"), "lora_alpha:", adapter_cfg.get("lora_alpha"))
targets = adapter_cfg.get("target_modules", [])
print("target_modules:", targets)

print("\n=== safetensors header (no weight load) ===")
st_path = MODEL_DIR / "model.safetensors"
with st_path.open("rb") as f:
    header_len = struct.unpack("<Q", f.read(8))[0]
    print(f"header bytes: {header_len}")
    header = json.loads(f.read(header_len))
# remove __metadata__ key if present
tensors = {k: v for k, v in header.items() if k != "__metadata__"}
print(f"tensor count: {len(tensors)}")
dtypes = sorted({v.get("dtype") for v in tensors.values()})
print("dtypes present:", dtypes)
total_params = sum(
    1
    for _ in ()
)  # placeholder
# compute param count from shape
param_count = 0
for v in tensors.values():
    shape = v.get("shape", [])
    n = 1
    for s in shape:
        n *= s
    param_count += n
print(f"approx param count from shapes: {param_count/1e9:.3f} B")

# Check the LoRA target module families exist among tensor names
layer0 = "model.layers.0."
expected_suffixes = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]
print("\n=== adapter target modules present in base model tensors ===")
all_ok = True
for suffix in expected_suffixes:
    key = layer0 + suffix
    present = key in tensors
    print(f"  {key}: {'OK' if present else 'MISSING'}")
    if not present:
        all_ok = False
assert all_ok, "One or more LoRA target modules missing from base model!"
print("All LoRA target modules found in base model: OK")

# Check a couple of adapter tensor names in adapter_model.safetensors header
print("\n=== adapter safetensors header ===")
ast_path = ADAPTER_DIR / "adapter_model.safetensors"
with ast_path.open("rb") as f:
    aheader_len = struct.unpack("<Q", f.read(8))[0]
    aheader = json.loads(f.read(aheader_len))
atensors = {k: v for k, v in aheader.items() if k != "__metadata__"}
print(f"adapter tensor count: {len(atensors)}")
# sample names
sample = sorted(atensors.keys())[:6]
for s in sample:
    print("  ", s)
adapter_targets_in_weights = set()
for name in atensors:
    # e.g. base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    for t in targets:
        if f".{t}." in name:
            adapter_targets_in_weights.add(t)
print("LoRA target modules present in adapter weights:", sorted(adapter_targets_in_weights))
assert adapter_targets_in_weights == set(targets), "Adapter weight modules != target_modules!"
print("Adapter weights cover all target modules: OK")

print("\nCOMPATIBILITY VERIFIED")
