import logging
from sqlalchemy import select
from lexicore.rag.database.connection import session_maker
from lexicore.rag.database.models import Chunk
from lexicore.rag.chunking.processor import DocumentProcessor
from lexicore.rag.embeddings.embedder import Embedder
from lexicore.rag.vector_db.chroma import ChromaVectorStore
from lexicore.rag.llm.llm_service import LLMService

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder | None = None,
        vector_store: ChromaVectorStore | None = None,
        llm: LLMService | None = None,
    ):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or ChromaVectorStore()
        self.llm = llm or LLMService()

    def index_chunks(self, batch_size: int = 32) -> dict[str, int]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        report = {"chunks_indexed": 0}

        with session_maker() as db:
            last_id = 0

            while True:
                chunks = db.scalars(
                    select(Chunk)
                    .where(Chunk.id > last_id)
                    .order_by(Chunk.id)
                    .limit(batch_size)
                ).all()

                if not chunks:
                    break

                texts = [chunk.content for chunk in chunks]
                embeddings = self.embedder.embed_texts(texts)

                ids = [str(chunk.id) for chunk in chunks]
                metadatas = [
                    {
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                    }
                    for chunk in chunks
                ]

                self.vector_store.add_chunks(
                    ids=ids,
                    texts=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )

                report["chunks_indexed"] += len(chunks)
                last_id = chunks[-1].id

                logger.info("Indexed %s chunks", len(chunks))

        return report

    def search(
        self,
        query: str,
        n_results: int = 5,
        candidate_k: int = 30,
        document_id: int | None = None,

    ) -> dict:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if n_results <= 0:
            raise ValueError("n_results must be positive")

        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")

        query_embedding = self.embedder.embed_query(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=max(n_results, candidate_k),
            document_id=document_id,
        )

        # Chroma returns cosine-distance results from closest to farthest.
        # Keep the top n_results while preserving the result structure.
        for key in (
            "ids",
            "documents",
            "metadatas",
            "distances",
            "embeddings",
            "uris",
            "data",
        ):
            values = results.get(key)

            if values and values[0] is not None:
                results[key] = [values[0][:n_results]]

        return results

    def ask(
        self,
        query: str,
        n_results: int = 5,
        candidate_k: int = 30,
        document_id: int | None = None,
    ) -> dict:
        results = self.search(
            query=query,
            n_results=n_results,
            candidate_k=candidate_k,
            document_id=document_id,
        )

        documents = results.get("documents", [[]])[0] or []
        ids = results.get("ids", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []

        if not documents:
            return {
                "query": query,
                "answer": (
                    "I could not find relevant contract information "
                    "to answer this question."
                ),
                "sources": [],
            }

        context_parts = []
        sources = []

        for index, document in enumerate(documents):
            chunk_id = ids[index] if index < len(ids) else None
            metadata = (
                metadatas[index]
                if index < len(metadatas) and metadatas[index]
                else {}
            )
            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            context_parts.append(document)

            sources.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": metadata.get("document_id"),
                    "chunk_index": metadata.get("chunk_index"),
                    "distance": distance,
                    "content": document,
                }
            )

        context = "\n\n".join(context_parts)

        answer = self.llm.generate_answer(
            query=query,
            context=context,
        )

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
        }

    def index_document(self, document_id: int, batch_size: int = 32) -> dict[str, int]:
        if document_id <= 0:
            raise ValueError("document_id must be positive")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        report = {"chunks_indexed": 0}

        with session_maker() as db:
            last_id = 0

            while True:
                chunks = db.scalars(
                    select(Chunk)
                    .where(
                        Chunk.document_id == document_id,
                        Chunk.id > last_id,
                    )
                    .order_by(Chunk.id)
                    .limit(batch_size)
                ).all()

                if not chunks:
                    break

                texts = [chunk.content for chunk in chunks]
                embeddings = self.embedder.embed_texts(texts)

                ids = [str(chunk.id) for chunk in chunks]
                metadatas = [
                    {
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                    }
                    for chunk in chunks
                ]

                self.vector_store.add_chunks(
                    ids=ids,
                    texts=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )

                report["chunks_indexed"] += len(chunks)
                last_id = chunks[-1].id

        if report["chunks_indexed"] == 0:
            raise ValueError(
                f"No chunks found for document ID {document_id}"
            )

        logger.info(
            "Indexed document ID %s: %s chunks",
            document_id,
            report["chunks_indexed"],
        )

        return report


def main():
    logging.basicConfig(level=logging.INFO)

    processor = DocumentProcessor()
    chunking_report = processor.process_pending_documents()
    print(f"Chunking complete: {chunking_report}")

    pipeline = RAGPipeline()
    indexing_report = pipeline.index_chunks()
    print(f"Indexing complete: {indexing_report}")


if __name__ == "__main__":
    main()


