import logging
from pathlib import Path
from sqlalchemy import select
from lexicore.rag.database.connection import session_maker
from lexicore.rag.database.models import Document, Chunk
from lexicore.rag.chunking.chunker import DocumentChunker

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, chunker: DocumentChunker | None = None):
        self.chunker = chunker or DocumentChunker()

    @staticmethod
    def read_document(storage_path: str) -> str:
        path = Path(storage_path)

        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="cp1252")

    def process_pending_documents(self) -> dict[str, int]:
        report = {
            "processed": 0,
            "failed": 0,
            "chunks_created": 0,
        }

        with session_maker() as db:
            documents = db.scalars(
                select(Document).where(Document.status == "pending")
            ).all()

            for document in documents:
                document_id = document.id

                try:
                    text = self.read_document(document.storage_path)
                    chunks = self.chunker.split_document(text)

                    if not chunks:
                        raise ValueError("Document produced no chunks")

                    for chunk_data in chunks:
                        db.add(
                            Chunk(
                                document_id=document_id,
                                content=chunk_data["content"],
                                chunk_index=chunk_data["chunk_index"],
                            )
                        )

                    document.status = "processed"
                    document.last_error = None
                    db.commit()

                    report["processed"] += 1
                    report["chunks_created"] += len(chunks)

                    logger.info(
                        "Processed document ID %s: %s chunks",
                        document_id,
                        len(chunks),
                    )

                except Exception as exc:
                    db.rollback()

                    document = db.get(Document, document_id)

                    if document is not None:
                        document.status = "failed"
                        document.attempt_count += 1
                        document.last_error = str(exc)
                        db.commit()

                    report["failed"] += 1

                    logger.exception(
                        "Failed to process document ID %s",
                        document_id,
                    )

        return report
