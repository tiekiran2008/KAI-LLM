from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RAGConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    persist_dir: str = field(
        default_factory=lambda: str(Path(__file__).resolve().parent / "chroma_db")
    )
    collection_name: str = "kai_llm_documents"
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 4
    max_chars_prompt: int = 3000
    embed_batch_size: int = 32
    supported_extensions: tuple = (".txt", ".pdf")
