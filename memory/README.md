# KAI-LLM Memory (Phase 2)

Long-term memory for KAI: store approved user facts and retrieve them by
semantic similarity. Separate from the RAG pipeline — memory retrieval is
not document retrieval, and this package never imports from `rag/`.

## What this is

- `MemoryService` — high-level API: `save_memory`, `retrieve_memories`,
  `list_memories`, `delete_memory`, `build_memory_context`.
- `SupabaseMemoryStore` — PostgreSQL backend (Supabase) with pgvector
  semantic search via the `match_memories` RPC; falls back to client-side
  cosine if the RPC is unavailable.
- `InMemoryStore` — local non-persistent store for tests and offline use.
- RLS + app-level identity checks enforce user ownership. The user ID is
  always verified against an authenticated Supabase session — never
  trusted from the caller alone.

## Requirements

Install (project root):

```
pip install -r memory/requirements.txt
```

Environment (see `memory/.env.example`; copy to `D:\KAI-LLM\.env`):

```
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_ANON_KEY=<anon key>
```

The service-role key is **server only** and must never appear in frontend
code or be committed to git (`.env` is gitignored).

## Supabase setup (requires your approval before any cloud work)

1. Create a Supabase project (or use an existing one).
2. In the SQL editor, run `memory/sql/001_create_memories.sql`:
   enables pgvector, creates the `memories` table, HNSW index,
   RLS policies (select/insert/delete limited to `auth.uid() = user_id`),
   and the `match_memories` RPC.
3. Put the URL and anon key in `.env`.

No cloud resources are created by this code — the SQL is applied by you.

## Usage

```python
from memory import MemoryService

# Production: identity comes from an authenticated Supabase session.
service = MemoryService.from_supabase()          # raises if not signed in

# Save only with explicit user approval (approved=True).
service.save_memory(user_id, "User prefers metric units", approved=True)

# Retrieve top-k memories for a query (k capped at max_top_k = 10).
memories = service.retrieve_memories(user_id, "what units does the user prefer?")

# Compact LLM-ready context block (char-capped; never full histories).
context = service.build_memory_context(user_id, "what units does the user prefer?")
```

Local/test mode (no Supabase, nothing persisted):

```python
service = MemoryService.local(authenticated_user_id="local-user")
```

## Guarantees

- `save_memory` without `approved=True` raises `ApprovalRequiredError`.
- Any `user_id` that does not match the verified authenticated identity
  raises `IdentityVerificationError` (plus DB RLS as a second layer).
- `top_k` capped at `max_top_k` (10); context capped at
  `context_max_chars` (1200 chars); per-memory content truncated at
  `per_memory_max_chars` (300).
- Timestamps and unique IDs on every memory.
- No KAI-LLM V3 model is loaded by this package (embeddings only).

## Testing

```
python memory/test_memory.py
```

Tests use `InMemoryStore` / `FakeEmbedder` / mocked Supabase clients — no
credentials, no network, no LLM load. The only real model loaded is
`all-MiniLM-L6-v2` (small embedding model, cached after first run).

## What's next (not implemented)

Phase 3 (AI agents) will build on top of this service. Not started yet.
