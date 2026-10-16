import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import RAGConfig
from rag.loaders import load_document, load_pdf, load_txt
from rag.llm_bridge import build_context, build_kai_prompt
from rag.retriever import Retriever
from rag.splitter import split_text

PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {label}")
    else:
        FAIL += 1
        print(f"  FAIL: {label} {detail}")


def make_sample_docs(tmp: Path) -> list:
    txt1 = tmp / "rag_overview.txt"
    txt1.write_text(
        "Retrieval-Augmented Generation (RAG) combines information retrieval with "
        "LLM generation. A RAG pipeline works as follows: chunk documents into "
        "smaller pieces, embed each chunk into a vector, store vectors in a vector "
        "database such as ChromaDB, embed the user query, retrieve the top-k most "
        "similar chunks by cosine similarity, and inject the retrieved context "
        "into the LLM prompt before generation. RAG reduces hallucination because "
        "answers are grounded in retrieved evidence. Do not confuse RAG with the "
        "ReAct pattern, which is an agent reasoning loop of thought, action, and "
        "observation.",
        encoding="utf-8",
    )
    txt2 = tmp / "kai_llm.txt"
    txt2.write_text(
        "KAI-LLM is a domain-specialized assistant for AI/ML engineering and "
        "software architecture. It is based on Qwen2.5-1.5B-Instruct and is "
        "fine-tuned with QLoRA. QLoRA trains a low-rank adaptation adapter on top "
        "of a 4-bit quantized base model, which keeps memory usage low enough for "
        "small GPUs. The LoRA adapter target modules include q_proj, k_proj, "
        "v_proj, and o_proj projections of the attention blocks.",
        encoding="utf-8",
    )
    pdf_path = tmp / "lora_notes.pdf"
    try:
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            "LoRA (Low-Rank Adaptation) freezes the pretrained weights and injects "
            "trainable rank decomposition matrices into each transformer layer. "
            "The update to weights is expressed as W + BA where B and A are small "
            "matrices of rank r. This reduces trainable parameters dramatically "
            "compared to full fine-tuning while preserving model quality.",
            fontsize=11,
        )
        doc.save(str(pdf_path))
        doc.close()
    except ImportError:
        pdf_path.write_text("LoRA freezes pretrained weights and trains small matrices.", encoding="utf-8")
    return [str(txt1), str(txt2), str(pdf_path)]


def test_loaders(tmp: Path, paths: list):
    print("\n[1] Document loaders (.txt and .pdf)")
    txt_content = load_txt(paths[0])
    check("load_txt returns RAG text", "Retrieval-Augmented Generation" in txt_content)

    pdf_content = load_pdf(paths[2])
    check("load_pdf extracts LoRA text", "Low-Rank Adaptation" in pdf_content)

    try:
        load_document(str(tmp / "unknown.xyz"))
        check("unsupported extension raises ValueError", False)
    except ValueError:
        check("unsupported extension raises ValueError", True)

    try:
        empty = tmp / "empty.txt"
        empty.write_text("   \n  ", encoding="utf-8")
        load_document(str(empty))
        check("empty document raises ValueError", False)
    except ValueError:
        check("empty document raises ValueError", True)


def test_splitter():
    print("\n[2] Text splitter")
    long_text = " ".join(f"word{i}" for i in range(600))
    chunks = split_text(long_text, chunk_size=200, chunk_overlap=50)
    check("multiple chunks produced", len(chunks) > 1, f"got {len(chunks)}")
    check("all chunks within chunk_size", all(len(c) <= 200 for c in chunks))
    joined = " ".join(chunks)
    expected_words = {f"word{i}" for i in range(600)}
    found_words = set(joined.split())
    check("no words lost", expected_words <= found_words,
          f"missing={len(expected_words - found_words)}")

    short = split_text("hello world", chunk_size=800, chunk_overlap=100)
    check("short text returns single chunk", short == ["hello world"])

    check("empty text returns empty list", split_text("   ") == [])


def test_retrieval(tmp: Path, paths: list):
    print("\n[3] Ingestion into ChromaDB")
    config = RAGConfig(
        persist_dir=str(tmp / "chroma_test"),
        collection_name="test_collection",
        chunk_size=400,
        chunk_overlap=80,
        top_k=3,
    )
    retriever = Retriever(config)
    chunk_count = retriever.ingest(paths)
    check("chunks ingested > 0", chunk_count > 0, f"got {chunk_count}")
    check("store count matches", retriever.count() == chunk_count,
          f"{retriever.count()} vs {chunk_count}")

    print("\n[4] Semantic retrieval queries (no LLM)")
    r1 = retriever.retrieve("What is Retrieval-Augmented Generation and how does a RAG pipeline work?")
    check("RAG query returns results", len(r1) > 0)
    top1_text = r1[0]["text"] if r1 else ""
    check("RAG query top hit mentions RAG", "RAG" in top1_text or "Retrieval-Augmented" in top1_text,
          f"top hit: {top1_text[:80]}")
    check("RAG query returns sources", all("source" in c for c in r1))
    check("distances are ascending", all(
        r1[i]["distance"] <= r1[i + 1]["distance"] + 1e-6 for i in range(len(r1) - 1)
    ) if len(r1) > 1 else True)

    r2 = retriever.retrieve("How does LoRA reduce trainable parameters?")
    check("LoRA query returns results", len(r2) > 0)
    top2_text = r2[0]["text"] if r2 else ""
    check("LoRA query top hit mentions LoRA", "LoRA" in top2_text or "Low-Rank" in top2_text,
          f"top hit: {top2_text[:80]}")

    r3 = retriever.retrieve("What base model is KAI-LLM fine-tuned from?")
    check("KAI-LLM query returns results", len(r3) > 0)
    top3_text = r3[0]["text"] if r3 else ""
    check("KAI-LLM query top hit mentions Qwen",
          "Qwen" in top3_text or "QLoRA" in top3_text,
          f"top hit: {top3_text[:80]}")

    print("\n[5] Context assembly for KAI-LLM prompt (no LLM)")
    context = build_context(r1, max_chars=3000)
    check("context non-empty", len(context) > 0)
    check("context contains numbered source tags", "[1]" in context)
    check("context respects max_chars", len(context) <= 3000)

    long_context = build_context(r1 + r1 + r1, max_chars=500)
    check("context truncated at max_chars", len(long_context) <= 500)

    messages = build_kai_prompt("What is RAG?", context)
    check("prompt has system role", messages[0]["role"] == "system")
    check("prompt system matches KAI-LLM identity", "KAI-LLM" in messages[0]["content"])
    check("prompt has user role with context", messages[1]["role"] == "user"
          and "Context:" in messages[1]["content"])
    check("prompt question included", "What is RAG?" in messages[1]["content"])

    no_ctx = build_kai_prompt("Hello", "")
    check("empty context omits Context section", "Context:" not in no_ctx[1]["content"])

    print("\n[6] Idempotent re-ingestion")
    count_before = retriever.count()
    retriever.ingest([paths[0]])
    check("re-ingesting same file does not duplicate chunks",
          retriever.count() == count_before,
          f"{count_before} -> {retriever.count()}")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="kai_rag_test_"))
    print(f"Test workspace: {tmp}")
    try:
        paths = make_sample_docs(tmp)
        test_loaders(tmp, paths)
        test_splitter()
        test_retrieval(tmp, paths)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 50)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    print("=" * 50)
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
