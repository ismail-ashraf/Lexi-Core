
import logging

from sqlalchemy import select, update

from lexicore.rag.database.connection import session_maker
from lexicore.rag.database.models import Chunk, Document
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
        """
        Index documents waiting for the indexing stage.
        Only documents with processing status are eligible.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        report = {"chunks_indexed": 0}

        with session_maker() as db:
            document_ids = db.scalars(
                select(Document.id)
                .where(Document.status == "processing")
                .order_by(Document.id)
            ).all()

        for document_id in document_ids:
            try:
                result = self.index_document(
                    document_id=document_id,
                    batch_size=batch_size,
                )
                report["chunks_indexed"] += result["chunks_indexed"]

            except Exception:
                logger.exception(
                    "Failed to index document %s during bulk indexing",
                    document_id,
                )

        logger.info(
            "Indexed %s chunks",
            report["chunks_indexed"],
        )

        return report

    def index_document(
        self,
        document_id: int,
        batch_size: int = 32,
    ) -> dict[str, int]:
        if document_id <= 0:
            raise ValueError("document_id must be positive")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # Atomically claim the indexing stage.
        with session_maker() as db:
            result = db.execute(
                update(Document)
                .where(
                    Document.id == document_id,
                    Document.status == "processing",
                )
                .values(
                    status="indexing",
                    last_error=None,
                )
            )
            db.commit()

            if result.rowcount != 1:
                document = db.get(Document, document_id)

                if document is None:
                    raise ValueError(
                        f"Document {document_id} does not exist"
                    )

                raise ValueError(
                    f"Document {document_id} cannot be indexed "
                    f"from status '{document.status}'"
                )

        try:
            # Load all chunk IDs belonging to this document.
            with session_maker() as db:
                chunk_ids = db.scalars(
                    select(Chunk.id)
                    .where(Chunk.document_id == document_id)
                    .order_by(Chunk.id)
                ).all()

            if not chunk_ids:
                raise ValueError(
                    f"No chunks found for document ID {document_id}"
                )

            # Remove vectors from a previous attempt.
            self.vector_store.delete_document(document_id)

            report = {"chunks_indexed": 0}

            # Embed and index chunks in batches.
            for offset in range(0, len(chunk_ids), batch_size):
                batch_ids = chunk_ids[offset:offset + batch_size]

                with session_maker() as db:
                    chunks = db.scalars(
                        select(Chunk)
                        .where(
                            Chunk.document_id == document_id,
                            Chunk.id.in_(batch_ids),
                        )
                        .order_by(Chunk.id)
                    ).all()

                if len(chunks) != len(batch_ids):
                    raise RuntimeError(
                        f"Expected {len(batch_ids)} chunks, "
                        f"but loaded {len(chunks)}"
                    )

                texts = [chunk.content for chunk in chunks]
                embeddings = self.embedder.embed_texts(texts)

                if len(embeddings) != len(chunks):
                    raise RuntimeError(
                        "Embedding count does not match chunk count"
                    )

                self.vector_store.add_chunks(
                    ids=[str(chunk.id) for chunk in chunks],
                    texts=texts,
                    embeddings=embeddings,
                    metadatas=[
                        {
                            "document_id": chunk.document_id,
                            "chunk_index": chunk.chunk_index,
                        }
                        for chunk in chunks
                    ],
                )

                report["chunks_indexed"] += len(chunks)

            if report["chunks_indexed"] != len(chunk_ids):
                raise RuntimeError(
                    f"Expected to index {len(chunk_ids)} chunks, "
                    f"but indexed {report['chunks_indexed']}"
                )

            # Mark ready only after every chunk was indexed.
            with session_maker() as db:
                result = db.execute(
                    update(Document)
                    .where(
                        Document.id == document_id,
                        Document.status == "indexing",
                    )
                    .values(
                        status="ready",
                        last_error=None,
                    )
                )
                db.commit()

                if result.rowcount != 1:
                    raise RuntimeError(
                        f"Could not mark document {document_id} as ready"
                    )

            logger.info(
                "Indexed document %s: %s chunks",
                document_id,
                report["chunks_indexed"],
            )

            return report

        except Exception as exc:
            logger.exception(
                "Failed to index document %s",
                document_id,
            )

            # Remove partially indexed vectors.
            cleanup_succeeded = False

            try:
                self.vector_store.delete_document(document_id)
                cleanup_succeeded = True
            except Exception:
                logger.exception(
                    "Failed to clean up vectors for document %s",
                    document_id,
                )

            error_message = str(exc)

            if not cleanup_succeeded:
                error_message += (
                    " | WARNING: Vector cleanup failed; "
                    "stale vectors may remain in ChromaDB."
                )

            with session_maker() as db:
                result = db.execute(
                    update(Document)
                    .where(
                        Document.id == document_id,
                        Document.status == "indexing",
                    )
                    .values(
                        status="failed",
                        attempt_count=Document.attempt_count + 1,
                        last_error=error_message,
                    )
                )
                db.commit()

                if result.rowcount != 1:
                    logger.error(
                        "Could not update failure status for document %s",
                        document_id,
                    )

            raise

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

        if document_id is not None and document_id <= 0:
            raise ValueError("document_id must be positive")

        # Get searchable document IDs from SQL.
        with session_maker() as db:
            statement = select(Document.id).where(
                Document.status == "ready"
            )

            if document_id is not None:
                statement = statement.where(
                    Document.id == document_id
                )

            ready_document_ids = db.scalars(statement).all()

        if not ready_document_ids:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        query_embedding = self.embedder.embed_query(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=max(n_results, candidate_k),
            document_id=document_id,
            allowed_document_ids=ready_document_ids,
        )

        # Return only the requested number of results.
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

        sources = []
        context_parts = []

        for index, content in enumerate(documents):
            chunk_id = ids[index] if index < len(ids) else None

            metadata = (
                metadatas[index]
                if index < len(metadatas) and metadatas[index]
                else {}
            )

            source_document_id = metadata.get("document_id")
            chunk_index = metadata.get("chunk_index")

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            context_parts.append(
                f"Document ID: {source_document_id}\n"
                f"Chunk ID: {chunk_id}\n"
                f"Content:\n{content}"
            )

            sources.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": source_document_id,
                    "chunk_index": chunk_index,
                    "distance": distance,
                    "content": content,
                }
            )

        answer = self.llm.generate_answer(
            query=query,
            context="\n\n".join(context_parts),
        )

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
        }

    def index_all_documents(
        self,
        batch_size: int = 32,
    ) -> dict[str, int]:
        return self.index_chunks(batch_size=batch_size)


def main():
    logging.basicConfig(level=logging.INFO)

    pipeline = RAGPipeline()
    report = pipeline.index_chunks()
    print(f"Indexing complete: {report}")


if __name__ == "__main__":
    main()
