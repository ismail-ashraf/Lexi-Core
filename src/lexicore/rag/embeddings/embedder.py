from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self,model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.tolist()
