"""Unit tests for KAI-LLM memory (Phase 2).

No credentials, no network, no LLM load. The only real model loaded is the
small all-MiniLM-L6-v2 embedding model (cached from Phase 1 after first run).
Run: python memory/test_memory.py
"""
from __future__ import annotations

import hashlib
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, ".")

from memory import (
    ApprovalRequiredError,
    IdentityVerificationError,
    MemoryConfig,
    MemoryConfigError,
    MemoryEmbedder,
    MemoryNotFoundError,
    MemoryService,
    InMemoryStore,
    SupabaseMemoryStore,
    cosine_similarity,
)


DIM = 384


class FakeEmbedder:
    """Deterministic bag-of-words embedder (384-dim, L2-normalized)."""

    def __init__(self, dim: int = DIM):
        self.dim = dim

    def embed(self, text: str):
        import numpy as np

        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm == 0:
            return [0.0] * self.dim
        return (vec / norm).tolist()


class TestNoHeavyImports(unittest.TestCase):
    def test_memory_does_not_import_rag_or_llm(self):
        self.assertNotIn("rag", sys.modules)
        self.assertNotIn("peft", sys.modules)
        self.assertNotIn("trl", sys.modules)


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        c = MemoryConfig()
        self.assertEqual(c.top_k, 5)
        self.assertEqual(c.max_top_k, 10)
        self.assertTrue(c.has_credentials is False)

    def test_require_credentials_raises(self):
        with self.assertRaises(MemoryConfigError):
            MemoryConfig().require_credentials()

    def test_repr_redacts_keys(self):
        c = MemoryConfig(supabase_url="https://x.supabase.co", supabase_anon_key="secret-anon")
        r = repr(c)
        self.assertNotIn("secret-anon", r)


class TestEmbeddings(unittest.TestCase):
    def test_cosine_identical(self):
        v = FakeEmbedder().embed("hello world test")
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=5)

    def test_cosine_orthogonal(self):
        a = [1.0] + [0.0] * (DIM - 1)
        b = [0.0, 1.0] + [0.0] * (DIM - 2)
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0, places=6)

    def test_real_minilm_embed_and_rank(self):
        emb = MemoryEmbedder()
        dep = emb.embed("Python decorators wrap functions to extend behavior.")
        cat = emb.embed("Cats enjoy sleeping in warm sunny spots for hours.")
        q = emb.embed("How do Python decorators work?")
        self.assertEqual(len(dep), DIM)
        self.assertGreater(cosine_similarity(q, dep), cosine_similarity(q, cat))


class TestInMemoryStore(unittest.TestCase):
    def test_create_and_list(self):
        store = InMemoryStore()
        r = store.create("u1", "hello", [1.0, 0.0])
        self.assertIn("id", r)
        self.assertIn("created_at", r)
        self.assertEqual(r["user_id"], "u1")
        rows = store.list_user("u1")
        self.assertEqual(len(rows), 1)
        self.assertNotIn("embedding", rows[0])

    def test_search_ownership(self):
        store = InMemoryStore()
        store.create("u1", "python tip", [1.0, 0.0])
        store.create("u2", "cooking tip", [0.0, 1.0])
        hits = store.search("u1", [1.0, 0.0], top_k=5)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["user_id"], "u1")

    def test_delete_ownership(self):
        store = InMemoryStore()
        r = store.create("u1", "secret", [1.0])
        self.assertFalse(store.delete("u2", r["id"]))
        self.assertTrue(store.delete("u1", r["id"]))
        self.assertFalse(store.delete("u1", r["id"]))


class TestMemoryServiceApprovalAndValidation(unittest.TestCase):
    def setUp(self):
        self.svc = MemoryService.local(
            authenticated_user_id="user-A", embedder=FakeEmbedder()
        )

    def test_save_requires_approval(self):
        with self.assertRaises(ApprovalRequiredError):
            self.svc.save_memory("user-A", "I like tea", approved=False)

    def test_save_with_approval_succeeds(self):
        rec = self.svc.save_memory("user-A", "I like tea", approved=True)
        self.assertEqual(rec["user_id"], "user-A")
        self.assertIn("id", rec)
        self.assertIn("created_at", rec)

    def test_save_empty_content_raises(self):
        with self.assertRaises(ValueError):
            self.svc.save_memory("user-A", "   ", approved=True)

    def test_save_too_long_content_raises(self):
        cfg = MemoryConfig()
        with self.assertRaises(ValueError):
            self.svc.save_memory("user-A", "x" * (cfg.max_content_chars + 1), approved=True)

    def test_identity_mismatch_raises(self):
        with self.assertRaises(IdentityVerificationError):
            self.svc.save_memory("user-B", "not mine", approved=True)
        with self.assertRaises(IdentityVerificationError):
            self.svc.retrieve_memories("user-B", "query")
        with self.assertRaises(IdentityVerificationError):
            self.svc.list_memories("user-B")
        with self.assertRaises(IdentityVerificationError):
            self.svc.delete_memory("user-B", "some-id")
        with self.assertRaises(IdentityVerificationError):
            self.svc.build_memory_context("user-B", "query")

    def test_service_requires_verified_identity(self):
        with self.assertRaises(IdentityVerificationError):
            MemoryService(store=InMemoryStore(), embedder=FakeEmbedder())

    def test_identity_verifier_callable(self):
        calls = []

        def verifier():
            calls.append(1)
            return "user-A"

        svc = MemoryService(
            store=InMemoryStore(),
            embedder=FakeEmbedder(),
            identity_verifier=verifier,
        )
        svc.save_memory("user-A", "verified", approved=True)
        self.assertGreaterEqual(len(calls), 1)
        with self.assertRaises(IdentityVerificationError):
            svc.save_memory("user-X", "nope", approved=True)


class TestMemoryServiceRetrieval(unittest.TestCase):
    def setUp(self):
        self.svc = MemoryService.local(
            authenticated_user_id="user-A", embedder=FakeEmbedder()
        )
        self.svc.save_memory(
            "user-A",
            "Python decorators wrap functions to extend behavior without changing them.",
            approved=True,
        )
        self.svc.save_memory(
            "user-A",
            "Cats enjoy sleeping in warm sunny spots for many hours each day.",
            approved=True,
        )

    def test_semantic_ranking(self):
        hits = self.svc.retrieve_memories("user-A", "python decorator wrap function")
        self.assertEqual(len(hits), 2)
        self.assertIn("Python decorators", hits[0]["content"])
        self.assertIn("similarity", hits[0])

    def test_top_k_limit(self):
        cfg = MemoryConfig()
        hits = self.svc.retrieve_memories("user-A", "python", top_k=cfg.max_top_k + 50)
        self.assertLessEqual(len(hits), cfg.max_top_k)

    def test_list_newest_first(self):
        rows = self.svc.list_memories("user-A")
        self.assertEqual(len(rows), 2)
        self.assertGreaterEqual(rows[0]["created_at"], rows[1]["created_at"])

    def test_list_limit(self):
        rows = self.svc.list_memories("user-A", limit=1)
        self.assertEqual(len(rows), 1)

    def test_delete(self):
        rows = self.svc.list_memories("user-A")
        mid = rows[0]["id"]
        self.svc.delete_memory("user-A", mid)
        with self.assertRaises(MemoryNotFoundError):
            self.svc.delete_memory("user-A", mid)
        with self.assertRaises(MemoryNotFoundError):
            self.svc.delete_memory("user-A", "does-not-exist")

    def test_ownership_isolation(self):
        other = MemoryService.local(
            authenticated_user_id="user-B", embedder=FakeEmbedder()
        )
        other.save_memory("user-B", "Unrelated fact about bicycles.", approved=True)
        hits = self.svc.retrieve_memories("user-A", "bicycles")
        self.assertEqual(len(hits), 2)
        for h in hits:
            self.assertNotIn("bicycles", h["content"])


class TestBuildContext(unittest.TestCase):
    def setUp(self):
        self.svc = MemoryService.local(
            authenticated_user_id="user-A", embedder=FakeEmbedder()
        )

    def test_empty_when_no_memories(self):
        self.assertEqual(self.svc.build_memory_context("user-A", "anything"), "")

    def test_contains_content_and_is_capped(self):
        long_text = ("The user lives in Helsinki and prefers metric units. " * 30).strip()
        self.svc.save_memory("user-A", long_text, approved=True)
        ctx = self.svc.build_memory_context("user-A", "units", max_chars=200)
        self.assertLessEqual(len(ctx), 200)
        self.assertIn("Helsinki", ctx)
        self.assertIn("[memory", ctx)

    def test_per_memory_truncation(self):
        cfg = MemoryConfig()
        long_text = ("A" * 500).strip()
        self.svc.save_memory("user-A", long_text, approved=True)
        ctx = self.svc.build_memory_context("user-A", "AAA")
        self.assertLessEqual(len(ctx), cfg.context_max_chars)
        self.assertIn("...", ctx)


class FakeResponse(SimpleNamespace):
    pass


class FakeChain:
    def __init__(self, results=None):
        self.ops = []
        self._results = results or {}

    def select(self, *a, **k):
        self.ops.append(("select", a, k))
        return self

    def insert(self, row):
        self.ops.append(("insert", row))
        return self

    def delete(self):
        self.ops.append(("delete",))
        return self

    def eq(self, col, val):
        self.ops.append(("eq", col, val))
        return self

    def order(self, col, desc=False):
        self.ops.append(("order", col, desc))
        return self

    def limit(self, n):
        self.ops.append(("limit", n))
        return self

    def execute(self):
        kinds = [o[0] for o in self.ops]
        if "insert" in kinds:
            row = next(o[1] for o in self.ops if o[0] == "insert")
            return FakeResponse(
                data=[{**row, "created_at": "2026-10-16T00:00:00+00:00"}]
            )
        if "delete" in kinds:
            return FakeResponse(data=self._results.get("delete", []))
        if "select" in kinds:
            return FakeResponse(data=self._results.get("select", []))
        return FakeResponse(data=[])


class FakeTable:
    def __init__(self, results=None):
        self.results = results or {}
        self.last_chain = None

    def _new(self):
        self.last_chain = FakeChain(self.results)
        return self.last_chain

    def select(self, *a, **k):
        return self._new().select(*a, **k)

    def insert(self, row):
        return self._new().insert(row)

    def delete(self):
        return self._new().delete()


class FakeSupabaseClient:
    def __init__(self, rpc_data=None, rpc_error=False, select_data=None, user_id="user-A"):
        self.rpc_data = rpc_data or []
        self.rpc_error = rpc_error
        self.rpc_calls = []
        self.tables = []
        self.select_data = select_data or []
        self.auth = SimpleNamespace(
            get_user=lambda: FakeResponse(
                user=SimpleNamespace(id=user_id) if user_id else None
            )
        )

    def table(self, name):
        t = FakeTable({"select": self.select_data})
        self.tables.append(t)
        return t

    def rpc(self, name, params):
        self.rpc_calls.append((name, params))

        client = self

        class _Rpc:
            def execute(self):
                if client.rpc_error:
                    raise RuntimeError("rpc unavailable")
                return FakeResponse(data=client.rpc_data)

        return _Rpc()


class TestSupabaseMemoryStore(unittest.TestCase):
    def test_create_calls_insert_with_user_owned_row(self):
        client = FakeSupabaseClient()
        store = SupabaseMemoryStore(config=MemoryConfig(), client=client)
        rec = store.create("user-A", "hello", [0.1, 0.2])
        self.assertEqual(rec["user_id"], "user-A")
        t = client.tables[-1]
        kind, row = t.last_chain.ops[0]
        self.assertEqual(kind, "insert")
        self.assertEqual(row["user_id"], "user-A")
        self.assertEqual(row["content"], "hello")
        self.assertEqual(row["embedding"], [0.1, 0.2])

    def test_search_uses_rpc_when_available(self):
        client = FakeSupabaseClient(
            rpc_data=[
                {
                    "id": "m1",
                    "content": "python tip",
                    "created_at": "2026-10-16T00:00:00+00:00",
                    "similarity": 0.91,
                }
            ]
        )
        store = SupabaseMemoryStore(config=MemoryConfig(), client=client)
        hits = store.search("user-A", [1.0, 0.0], top_k=3)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["id"], "m1")
        self.assertEqual(client.rpc_calls[0][0], "match_memories")
        params = client.rpc_calls[0][1]
        self.assertEqual(params["query_user_id"], "user-A")
        self.assertEqual(params["match_count"], 3)

    def test_search_falls_back_to_client_side_cosine(self):
        client = FakeSupabaseClient(
            rpc_error=True,
            select_data=[
                {
                    "id": "m1",
                    "user_id": "user-A",
                    "content": "near",
                    "created_at": "2026-10-16T00:00:00+00:00",
                    "embedding": [1.0, 0.0],
                },
                {
                    "id": "m2",
                    "user_id": "user-A",
                    "content": "far",
                    "created_at": "2026-10-15T00:00:00+00:00",
                    "embedding": [0.0, 1.0],
                },
            ],
        )
        store = SupabaseMemoryStore(config=MemoryConfig(), client=client)
        hits = store.search("user-A", [1.0, 0.0], top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["id"], "m1")
        self.assertIn("similarity", hits[0])

    def test_delete_filters_by_user(self):
        client = FakeSupabaseClient(select_data=[{"id": "m1"}])
        store = SupabaseMemoryStore(config=MemoryConfig(), client=client)
        self.assertTrue(store.delete("user-A", "m1"))

    def test_delete_not_found(self):
        client = FakeSupabaseClient(select_data=[])
        store = SupabaseMemoryStore(config=MemoryConfig(), client=client)
        self.assertFalse(store.delete("user-A", "missing"))


class TestFromSupabaseIdentity(unittest.TestCase):
    def test_requires_verified_session(self):
        client = FakeSupabaseClient(user_id=None)
        with self.assertRaises(IdentityVerificationError):
            MemoryService.from_supabase(config=MemoryConfig(), client=client)

    def test_builds_service_with_verified_id(self):
        client = FakeSupabaseClient(user_id="user-A")
        svc = MemoryService.from_supabase(config=MemoryConfig(), client=client)
        self.assertEqual(svc._authenticated_user_id, "user-A")


if __name__ == "__main__":
    unittest.main(verbosity=2)
