from .config import RAGConfig


class Embedder:
    def __init__(self, config: RAGConfig = None):
        from sentence_transformers import SentenceTransformer

        config = config or RAGConfig()
        self.model_name = config.embedding_model
        self.batch_size = config.embed_batch_size
        self.model = SentenceTransformer(self.model_name)

    def embed_documents(self, texts: list) -> list:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list:
        embedding = self.model.encode(
            [text],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding[0].tolist()
