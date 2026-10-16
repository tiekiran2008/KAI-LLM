# KAI-LLM RAG — Phase 1

Retrieval-Augmented Generation for KAI-LLM V3 (Qwen2.5-1.5B-Instruct + LoRA adapter).
Lightweight: designed for an 8 GB RAM laptop. Does not modify any trained model files.

## Setup

```bash
pip install -r rag/requirements.txt
```

Already-installed packages (verified): `chromadb`, `sentence-transformers`, `pypdf`, `PyMuPDF`.

## Quick start

### 1. Ingest documents (PDF or TXT)

```python
from rag import Retriever

retriever = Retriever()                 # persists to rag/chroma_db/
retriever.ingest("docs/my_notes.txt")   # single file
retriever.ingest(["a.pdf", "b.txt"])    # multiple files
retriever.ingest("docs/")               # whole directory (recurses .pdf/.txt)
```

### 2. Retrieve context

```python
chunks = retriever.retrieve("How does QLoRA reduce memory usage?", top_k=4)
for c in chunks:
    print(c["source"], c["distance"], c["text"][:120])
```

### 3. Provide retrieved context to KAI-LLM

```python
from rag import build_context, build_kai_prompt

context = build_context(chunks, max_chars=3000)
messages = build_kai_prompt("How does QLoRA reduce memory usage?", context)
# messages -> tokenizer.apply_chat_template(...) -> model.generate(...)
```

### 4. One-call RAG generation (loads the LLM only when called)

```python
from rag import Retriever, generate_with_rag

retriever = Retriever()
result = generate_with_rag(
    "What is the difference between RAG and ReAct?",
    retriever,
    adapter_path="./qlora_output_kai",   # Colab-trained LoRA adapter
)
print(result["answer"])
```

- Base model: `Qwen/Qwen2.5-1.5B-Instruct` (fp16, ~3 GB RAM)
- Adapter: if `./qlora_output_kai/adapter_config.json` is absent, a warning is issued and the base model is used instead.
- The model is loaded lazily (only inside `generate_with_rag`) and cached.

## Standalone retrieval test (no LLM required)

```bash
python rag/test_retrieval.py
```

Covers: PDF/TXT loading, chunking, embedding with `all-MiniLM-L6-v2`, ChromaDB
storage, semantic queries, context assembly, prompt building, idempotent re-ingest.
Uses a temporary directory — no persistent test data is left behind.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Public API exports |
| `config.py` | `RAGConfig` dataclass (model name, chunk size, overlap, top_k, paths) |
| `loaders.py` | `load_pdf`, `load_txt`, `load_document` (.pdf/.txt dispatch) |
| `splitter.py` | `split_text` — word-boundary chunking with overlap |
| `embedder.py` | `Embedder` — all-MiniLM-L6-v2 embeddings (normalized, batched) |
| `vector_store.py` | `VectorStore` — ChromaDB persistent client, upsert/query/count |
| `retriever.py` | `Retriever` — ingest + retrieve orchestration |
| `llm_bridge.py` | `build_context`, `build_kai_prompt`, `generate_with_rag` (lazy LLM) |
| `test_retrieval.py` | Standalone test suite — runs without loading the LLM |
| `requirements.txt` | Dependency list |
| `README.md` | This file |

## Memory notes (8 GB RAM)

- Embedding model: ~90 MB (`all-MiniLM-L6-v2`)
- ChromaDB: on-disk persistence, minimal RAM
- LLM generation (optional): fp16 1.5B base ≈ 3 GB RAM; only loaded on demand
- Ingestion embeds in batches of 32 — no full-corpus embeddings held in memory

## Not included (later phases)

- Long-term memory
- AI agents / ReAct tool loops
