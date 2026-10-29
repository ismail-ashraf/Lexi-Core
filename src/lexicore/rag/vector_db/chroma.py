
from pathlib import Path

import chromadb


class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "lexicore_chunks",
    ):
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        if not ids:
            return

        if not (
            len(ids) == len(texts) == len(embeddings) == len(metadatas)
        ):
            raise ValueError(
                "IDs, texts, embeddings, and metadatas must have equal lengths"
            )

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
            self,
            query_embedding: list[float],
            n_results: int = 5,
            document_id: int | None = None,
    ):
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")

        if n_results <= 0:
            raise ValueError("n_results must be positive")

        total_chunks = self.collection.count()

        if total_chunks == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, total_chunks),
        }

        if document_id is not None:
            query_kwargs["where"] = {"document_id": document_id}

        return self.collection.query(**query_kwargs)

    def count(self) -> int:
        return self.collection.count()