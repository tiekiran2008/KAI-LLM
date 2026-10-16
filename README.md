# KAI-LLM

**KAI-LLM V3** — a fine-tuned 1.5B language model with Retrieval-Augmented
Generation (RAG), long-term memory, LangGraph agents, and a FastAPI backend.
Built to run on a consumer laptop (8 GB RAM) and designed JARVIS-style:
understand the request, pick a tool, verify the result, answer.

- **Base model:** [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **Trained LoRA adapter (V3):** [kiran2008/KAI-LLM-v3](https://huggingface.co/kiran2008/KAI-LLM-v3)
- **Training dataset:** [`dataset/kai_llm_v6.json`](dataset/kai_llm_v6.json) — 1,200 audited instruction/code pairs (see [`dataset/v6_audit_report.txt`](dataset/v6_audit_report.txt))

> **Status:** the full stack (RAG + memory + agents + API) is covered by
> **128 passing tests**. *Actual end-to-end inference with the trained V3
> adapter and full online deployment remain unverified* — see
> [Current limitations](#current-limitations).

---

## Table of contents

- [Architecture](#architecture)
- [Technology stack](#technology-stack)
- [Repository layout](#repository-layout)
- [Fine-tuning approach](#fine-tuning-approach)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [API reference](#api-reference)
- [Testing](#testing)
- [Model & dataset on Hugging Face](#model--dataset-on-hugging-face)
- [Current limitations](#current-limitations)
- [Roadmap](#roadmap)
- [License](#license)

---

## Architecture

```
                        ┌──────────────────────────────────┐
                        │        FastAPI  (api/)           │
                        │  /health  /chat  /documents/ingest│
                        │  /memories  (JWT required)       │
                        └───────────────┬──────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          │                             │                             │
          ▼                             ▼                             ▼
 ┌────────────────────┐       ┌────────────────────┐       ┌────────────────────┐
 │      agents/       │       │        rag/        │       │      memory/       │
 │  LangGraph state   │──────▶│  ChromaDB + MiniLM │       │  Supabase +        │
 │  machine: understand│      │  persistent vector │       │  pgvector semantic │
 │  select tool → run │       │  store, chunking,  │       │  search, RLS +     │
 │  inspect → answer  │       │  PDF/TXT loaders   │       │  JWT identity      │
 └─────────┬──────────┘       └────────────────────┘       └────────────────────┘
           │
           ▼
 ┌────────────────────────────────────────────────────────────┐
 │            LLM provider  (api/llm_providers.py)            │
 │   mock          → deterministic fake LLM (tests / dev)     │
 │   openai_endpoint → any OpenAI-compatible server           │
 │   local_adapter → Qwen2.5-1.5B + LoRA V3 adapter           │
 │                     (strict verification — never silently  │
 │                      falls back to the base model)         │
 └────────────────────────────────────────────────────────────┘
```

Every stage is independently testable and independently usable; the API
layer wires them together without rewriting them.

---

## Technology stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Base LLM | Qwen2.5-1.5B-Instruct (1.54 B params, BF16) |
| Fine-tuning | QLoRA — PEFT + TRL `SFTTrainer` + bitsandbytes 4-bit |
| Inference (local) | torch + transformers + peft, or llama.cpp (Q4_K_M GGUF + LoRA GGUF) |
| Agents | LangGraph state graph, provider-independent tool protocol |
| RAG | ChromaDB (persistent) + `all-MiniLM-L6-v2` embeddings |
| Memory | Supabase (PostgreSQL) + pgvector, row-level security |
| API | FastAPI + uvicorn, JWT-verified Supabase auth |
| Testing | pytest (integration) + plain-Python suites |

---

## Repository layout

```
KAI-LLM/
├── api/            FastAPI backend: /chat, /health, /documents, /memories
│   └── tests/      25 integration tests (mock LLM, fake auth)
├── agents/         LangGraph workflow + safe read-only tools (41 tests)
├── rag/            Retriever, loaders, splitter, embedder (29 tests)
├── memory/         MemoryService, Supabase store, auth bridge (33 tests)
│   └── sql/        Supabase schema + match_memories RPC
├── training/       QLoRA training script (Google Colab, Tesla T4)
├── dataset/        kai_llm_v6.json (1,200 examples) + audit report + parts
├── scripts/        Phase-5 recovery/verification tools + dataset generator
├── requirements.txt Consolidated runtime + test dependencies
├── .env.example    Environment template (placeholders only)
└── README.md       This file
```

---

## Fine-tuning approach

KAI-LLM V3 is **Qwen2.5-1.5B-Instruct** fine-tuned with **QLoRA** on a
custom 1,200-example dataset of AI/ML engineering, coding, RAG, and agent
tasks (categories audited — 0 duplicates, 0 malformed entries, 0 banned
patterns; see [`dataset/v6_audit_report.txt`](dataset/v6_audit_report.txt)).

| Setting | Value |
|---|---|
| Method | QLoRA (4-bit NF5/BF16 base + low-rank adapters) |
| LoRA rank `r` | 64 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Target modules | `q_proj` `k_proj` `v_proj` `o_proj` `gate_proj` `up_proj` `down_proj` |
| Optimizer | `paged_adamw_8bit` |
| Learning rate | 2e-4, cosine schedule, 3 % warmup |
| Batch | 1 × gradient-accumulation 16 |
| Steps | 200 |
| Hardware | Google Colab Tesla T4 (16 GB VRAM) |
| Output | `qlora_output_kai/` — `adapter_config.json` + `adapter_model.safetensors` (~282 MB) |

The adapter is hosted on Hugging Face:
**[kiran2008/KAI-LLM-v3](https://huggingface.co/kiran2008/KAI-LLM-v3)**
(base-model link and metadata included in the model card). It is **not**
bundled in this repository; download it separately (see
[Setup](#setup)).

The training script is [`training/train_qlora.py`](training/train_qlora.py);
pinned Colab dependencies are in [`training/requirements.txt`](training/requirements.txt).

---

## Features

### RAG — `rag/`
- Ingest `.txt` / `.pdf` documents into a persistent ChromaDB collection.
- Deterministic chunking (800 chars, 150 overlap) and stable chunk IDs.
- `all-MiniLM-L6-v2` embeddings; top-k retrieval with distances.
- Prompt builders that cap context size for the 1.5B model.

### Long-term memory — `memory/`
- `MemoryService`: save / retrieve / list / delete user facts.
- Supabase backend with pgvector semantic search (`match_memories` RPC)
  and a client-side cosine fallback.
- **Identity is always verified** from the Supabase JWT — never trusted
  from the request body.
- Saving a memory requires **explicit user approval** (`approved=true`).
- SQL schema + RLS policies: [`memory/sql/001_create_memories.sql`](memory/sql/001_create_memories.sql).

### Agents — `agents/`
- LangGraph state graph: *understand → select tool → validate args →
  execute → inspect result → generate answer*, with a hard tool-call
  budget and timeouts.
- Works with any chat model through a small `LLMProvider` interface —
  native tool-calling is **not** required.
- Safe, read-only tools:
  - `calculator` — whitelisted AST evaluator (never `eval`/`exec`).
  - `document_search` — RAG retrieval over ingested documents.
  - `memory_search` — authenticated semantic memory lookup.
  - `file_inspect` — read-only, confined to an approved workspace dir.
- All tool/document outputs are fenced as untrusted data before they
  reach the model (prompt-injection hardening).

### API — `api/`
- FastAPI factory with dependency injection for tests.
- `/health` never claims V3 is connected without real adapter
  verification *and* a successful inference.
- Mock, OpenAI-compatible-endpoint, and local-adapter LLM modes.

---

## Setup

Requires **Python 3.11+**.

```bash
git clone https://github.com/<your-username>/KAI-LLM.git
cd KAI-LLM
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### Configure the environment

```bash
copy .env.example .env          # Windows
# cp .env.example .env          # macOS/Linux
```

Edit `.env` and fill in what you need:

| Variable | Purpose |
|---|---|
| `KAI_API_MODE` | `mock` (default), `openai_endpoint`, or `local_adapter` |
| `KAI_ENDPOINT_BASE_URL` / `KAI_ENDPOINT_MODEL` | OpenAI-compatible server (Ollama, vLLM, LM Studio…) |
| `SUPABASE_URL` / `SUPABASE_ANON_KEY` | Enables `/memories` + memory search tool |
| `KAI_WORKSPACE_DIR` | Enables the read-only `file_inspect` tool |

### (Optional) Enable long-term memory

1. Create a project at [supabase.com](https://supabase.com).
2. Open the SQL editor and run
   [`memory/sql/001_create_memories.sql`](memory/sql/001_create_memories.sql)
   (creates the `memories` table, pgvector index, RLS policies, and the
   `match_memories` RPC).
3. Put the project URL and anon key in `.env`.

### (Optional) Download the trained V3 adapter

The adapter is **not** in this repo. To run genuine V3 inference:

```bash
# Adapter (~282 MB) — from https://huggingface.co/kiran2008/KAI-LLM-v3
huggingface-cli download kiran2008/KAI-LLM-v3 \
  adapter_config.json adapter_model.safetensors \
  --local-dir ./qlora_output_kai

# Base model (~3 GB) — only needed for local_adapter mode
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir ./model_cache/Qwen2.5-1.5B-Instruct
```

Then set `KAI_API_MODE=local_adapter` in `.env`. The API refuses to start
in this mode if the adapter is missing or its `base_model_name_or_path`
does not match `Qwen/Qwen2.5-1.5B-Instruct` — it **never** silently
substitutes the base model.

> **RAM note (8 GB laptops):** loading BF16 base + adapter needs ≈ 4.4 GB
> peak. A safer low-memory route is llama.cpp with the official Q4_K_M
> GGUF (~1.04 GiB) plus a converted LoRA GGUF (~148 MB) — ≈ 1.35 GiB
> working set. See [Roadmap](#roadmap).

---

## Usage

### Start the API

```bash
# Mock mode (default — no model download needed)
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

```bash
# Genuine V3 (requires the adapter + base model, see Setup)
set KAI_API_MODE=local_adapter
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

### Chat (agent with tools)

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"What is 84 * 3 / 2?\"}"
```

```json
{
  "reply": "84 * 3 / 2 = 126.0",
  "success": true,
  "llm_mode": "mock",
  "steps": ["understand", "select_tool", "execute_tool", "generate_response"],
  "tool_calls": [{"tool": "calculator", "ok": true}],
  "sources": [],
  "memories_used": 0
}
```

### Ingest a document (RAG)

```bash
curl -X POST http://127.0.0.1:8000/documents/ingest \
  -F "file=@notes.txt"
```

### Save a memory (requires Supabase + a signed-in user's JWT)

```bash
curl -X POST http://127.0.0.1:8000/memories \
  -H "Authorization: Bearer <supabase-jwt>" \
  -H "Content-Type: application/json" \
  -d "{\"content\":\"User prefers metric units\",\"approved\":true}"
```

### Use the RAG retriever directly

```python
from rag import Retriever

retriever = Retriever()                    # persists to rag/chroma_db/
retriever.ingest("docs/my_notes.txt")
chunks = retriever.retrieve("What is QLoRA?", top_k=4)
for c in chunks:
    print(c["source"], round(c["distance"], 3), c["text"][:100])
```

---

## API reference

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health` | none | Service + model status. Never loads the model; never claims V3 connected without verification **and** a successful inference. |
| `POST` | `/chat` | optional Bearer | Agent chat. Tools: calculator, document search, memory search (JWT only), file inspect. |
| `POST` | `/documents/ingest` | none | Upload `.txt`/`.pdf` (max 5 MB) into ChromaDB. |
| `GET` | `/memories` | Bearer | List the verified user's memories. |
| `POST` | `/memories` | Bearer | Save a memory — **requires `"approved": true`**. |
| `DELETE` | `/memories/{id}` | Bearer | Delete one of the verified user's memories. |

Full details, security properties, and mode table:
[`api/README.md`](api/README.md).

---

## Testing

**128 tests, all passing** — no GPU, no network, no model weights required:

| Suite | Tests | Command |
|---|---|---|
| RAG | 29 | `python rag/test_retrieval.py` |
| Memory | 33 | `python memory/test_memory.py` |
| Agents | 41 | `python agents/test_agents.py` |
| API (integration) | 25 | `python -m pytest api/tests/test_integration.py -v` |
| **Total** | **128** | |

The API suite exercises the full `/chat` → agent → tool → LLM pipeline
with a deterministic mock LLM and a fake auth bridge. Real Supabase and
real V3 inference are intentionally *not* exercised in CI-style tests.

---

## Model & dataset on Hugging Face

| Artifact | Link |
|---|---|
| KAI-LLM V3 LoRA adapter | https://huggingface.co/kiran2008/KAI-LLM-v3 |
| Base model | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct |
| v1 dataset (historical) | https://huggingface.co/datasets/kiran2008/KAI-LLM-v1-dataset |

Model weights are **not** stored in this repository (see `.gitignore`).

---

## Current limitations

- **Actual V3 inference is not yet verified end-to-end.** The trained
  adapter is recovered, integrity-checked (sha256 + GGUF conversion
  succeeded), and the API's strict verification passes — but no real
  forward pass with the adapter has been confirmed on this hardware.
  `/health` therefore reports `v3_claimed_connected: false` until a real
  inference request succeeds. The base model is **never** silently
  substituted.
- **Online deployment is unverified.** The API is developed and tested
  locally (`127.0.0.1`); no production/HTTPS deployment has been set up.
- **Supabase migration not applied.** Run
  `memory/sql/001_create_memories.sql` yourself; memory endpoints return
  `503` until configured.
- **8 GB RAM laptops** cannot load the BF16 base + adapter (≈ 4.4 GB peak
  with only ~1 GB free). Prefer `openai_endpoint` (Ollama/LM Studio) or
  the llama.cpp Q4_K_M + LoRA-GGUF route (≈ 1.35 GB).
- The agent has no persistent multi-turn chat history yet.

---

## Roadmap

1. **Verify genuine V3 inference** — run llama.cpp Q4_K_M + LoRA GGUF
   locally (or on a free remote GPU) and confirm `v3_claimed_connected`
   flips to `true`.
2. **Wire a llama.cpp backend** into `api/llm_providers.py` as a
   low-memory `local_adapter` alternative.
3. **Apply the Supabase migration** and run end-to-end memory tests with
   a real JWT.
4. **Deploy** behind HTTPS (e.g. Fly.io / Railway / a VPS) with rate
   limiting and persistent ChromaDB storage.
5. **Streaming responses** and multi-turn chat history.

---

## License

See the repository's `LICENSE` file (to be added before public release).
The base model
[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
is subject to its own license on Hugging Face.
