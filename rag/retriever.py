from pathlib import Path

from .config import RAGConfig
from .embedder import Embedder
from .loaders import load_document
from .splitter import split_text
from .vector_store import VectorStore


class Retriever:
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.embedder = Embedder(self.config)
        self.store = VectorStore(self.config)

    def ingest(self, paths) -> int:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        paths = list(paths)
        if len(paths) == 1 and Path(paths[0]).is_dir():
            directory = Path(paths[0])
            paths = sorted(
                str(p)
                for ext in self.config.supported_extensions
                for p in directory.rglob(f"*{ext}")
            )
        if not paths:
            raise ValueError("No supported documents (.txt, .pdf) found to ingest.")

        total_chunks = 0
        for path in paths:
            path = str(path)
            source = Path(path).name
            text = load_document(path)
            chunks = split_text(
                text,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            if not chunks:
                continue

            ids = [f"{source}::{i}" for i in range(len(chunks))]
            embeddings = self.embedder.embed_documents(chunks)
            metadatas = [
                {"source": source, "path": path, "chunk": i, "total_chunks": len(chunks)}
                for i in range(len(chunks))
            ]
            self.store.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
            total_chunks += len(chunks)
            print(f"  Ingested '{source}': {len(chunks)} chunks")

        return total_chunks

    def retrieve(self, query: str, top_k: int = None) -> list:
        top_k = top_k or self.config.top_k
        query_embedding = self.embedder.embed_query(query)
        results = self.store.query(query_embedding, top_k=top_k)
        return [
            {
                "text": item["text"],
                "source": item["metadata"].get("source", "unknown"),
                "chunk": item["metadata"].get("chunk", -1),
                "distance": item["distance"],
            }
            for item in results
        ]

    def count(self) -> int:
        return self.store.count()

    def clear(self):
        self.store.client.delete_collection(self.store.collection.name)
        self.store.collection = self.store.client.get_or_create_collection(
            name=self.store.config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
