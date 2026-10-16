# KAI-LLM V3 API (Phase 4)

FastAPI backend connecting the existing KAI-LLM components (RAG, memory,
agents) without rewriting them.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | none | Service + model status. Never loads the model; never claims V3 is connected without adapter verification. |
| POST | `/chat` | optional Bearer | Agent chat. Tools: calculator, document search (RAG), memory search (only with a valid Supabase JWT), file inspect (if `KAI_WORKSPACE_DIR` set). |
| POST | `/documents/ingest` | none | Upload `.txt`/`.pdf` (max 5 MB), ingest into ChromaDB. |
| GET | `/memories` | Bearer | List the verified user's memories. |
| POST | `/memories` | Bearer | Save a memory. **Requires `"approved": true`** (explicit user approval). |
| DELETE | `/memories/{id}` | Bearer | Delete one of the verified user's memories. |

User identity for memory operations ALWAYS comes from the verified
Supabase JWT (`client.auth.get_user`), never from the request body.

## KAI-LLM V3 model status (verified this phase)

Verified from `training/train_qlora.py`:

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Adapter output dir: `./qlora_output_kai` (LoRA r=64, alpha=16,
  targets q/k/v/o/gate/up/down_proj)
- **The adapter is NOT bundled with this repository.** It was trained on
  Colab and is published at
  [kiran2008/KAI-LLM-v3](https://huggingface.co/kiran2008/KAI-LLM-v3).
  Download it into `./qlora_output_kai` to enable `local_adapter` mode
  (weights are gitignored).

Therefore:

- `KAI_API_MODE=local_adapter` will **refuse to start** unless the
  adapter folder with a matching `adapter_config.json` exists.
  It never silently substitutes the base Qwen model (requirement 6).
- `/health` reports `v3_claimed_connected: false` until a real inference
  request has actually used the trained adapter.
- To enable V3: copy the Colab-trained `qlora_output_kai/` folder to the
  project root, then start with `KAI_API_MODE=local_adapter`.

## LLM modes (`KAI_API_MODE`)

| Mode | Provider | Use |
|---|---|---|
| `mock` (default) | `KAIMockInferenceLLM` | Integration tests, offline dev. Deterministic. |
| `openai_endpoint` | `OpenAICompatibleLLM` | Any OpenAI-compatible server (Ollama/vLLM/LM Studio/remote) serving KAI-LLM V3. Requires `KAI_ENDPOINT_BASE_URL` + `KAI_ENDPOINT_MODEL`. |
| `local_adapter` | `LocalV3LLM` | In-process Qwen2.5-1.5B + LoRA adapter. Strict adapter verification; lazy model load on first `/chat`. Uses ~3-4 GB RAM (float16). |

## Environment variables

Copy `memory/.env.example` to `.env` and extend:

```
KAI_API_MODE=mock                # mock | openai_endpoint | local_adapter
KAI_ENDPOINT_BASE_URL=           # e.g. http://localhost:11434/v1
KAI_ENDPOINT_MODEL=
KAI_ENDPOINT_API_KEY=
KAI_ADAPTER_PATH=D:\KAI-LLM\qlora_output_kai
KAI_RAG_ENABLED=1
KAI_RAG_PERSIST_DIR=D:\KAI-LLM\rag\chroma_db
KAI_WORKSPACE_DIR=               # enables file_inspect (read-only)
SUPABASE_URL=                    # enables /memories
SUPABASE_ANON_KEY=
```

## Local startup

From `D:\KAI-LLM`:

```
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Then:

```
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"message\":\"calculate 12 * 8 + 4\"}"
curl -X POST http://127.0.0.1:8000/documents/ingest -F "file=@notes.txt"
```

Memory (requires Supabase configured + a signed-in user's JWT):

```
curl -X POST http://127.0.0.1:8000/memories ^
  -H "Authorization: Bearer <supabase-jwt>" ^
  -H "Content-Type: application/json" ^
  -d "{\"content\":\"User prefers metric units\",\"approved\":true}"
```

## Security properties

- **Approval gate**: `POST /memories` returns 403 unless `approved=true`.
  No mutating agent tools are registered; the agent approval callback is
  deny-all.
- **Untrusted data**: document and tool outputs are fenced with
  `<<<UNTRUSTED_TOOL_OUTPUT>>>` markers before reaching the LLM
  (verified by integration test with a malicious document).
- **No silent model substitution**: `LocalV3LLM` raises `V3AdapterError`
  if the adapter is missing or mismatches; it never falls back to base
  Qwen.
- **Identity**: memory user id comes only from JWT verification.
- **Resource caps**: 5 MB upload limit, tool timeouts/retries from
  `AgentConfig`, lazy model/embedding loads.

## Tests

```
python -m pytest api/tests/test_integration.py -v
```

25 tests. See the Phase 4 report for which components are real vs mocked
and what remains blocked (real V3 inference, real Supabase).
