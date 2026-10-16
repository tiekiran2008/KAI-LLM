from .config import RAGConfig


class VectorStore:
    def __init__(self, config: RAGConfig = None):
        import chromadb

        config = config or RAGConfig()
        self.config = config
        self.client = chromadb.PersistentClient(path=config.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, ids: list, documents: list, embeddings: list, metadatas: list = None):
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, query_embedding: list, top_k: int = None) -> list:
        top_k = top_k or self.config.top_k
        count = self.collection.count()
        if count == 0:
            return []
        n = min(top_k, count)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        items = []
        for i in range(len(results["ids"][0])):
            items.append(
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] or {},
                    "distance": results["distances"][0][i],
                }
            )
        return items

    def count(self) -> int:
        return self.collection.count()

    def delete_source(self, source: str):
        self.collection.delete(where={"source": source})
