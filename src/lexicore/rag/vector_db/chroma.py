
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

    def delete_document(self, document_id: int) -> None:
        if document_id <= 0:
            raise ValueError("document_id must be positive")

        self.collection.delete(
            where={"document_id": document_id}
        )

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        document_id: int | None = None,
        allowed_document_ids: list[int] | None = None,
    ) -> dict:
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")

        if n_results <= 0:
            raise ValueError("n_results must be positive")

        empty_result = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        # No ready documents means no searchable documents.
        if not allowed_document_ids:
            return empty_result

        # Restrict search to ready documents only.
        filters = [
            {
                "document_id": {
                    "$in": allowed_document_ids
                }
            }
        ]

        # Optional document-specific search.
        if document_id is not None:
            if document_id not in allowed_document_ids:
                return empty_result

            filters.append(
                {"document_id": document_id}
            )

        where = (
            filters[0]
            if len(filters) == 1
            else {"$and": filters}
        )

        # ChromaDB count() does not support where filters.
        total_chunks = self.collection.count()

        if total_chunks == 0:
            return empty_result

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, total_chunks),
            where=where,
        )

    def count(self) -> int:
        return self.collection.count()
