import logging
from pathlib import Path
from sqlalchemy import select, update
from lexicore.rag.database.connection import session_maker
from lexicore.rag.database.models import Document, Chunk
from lexicore.rag.chunking.chunker import DocumentChunker
from pypdf import PdfReader

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunker: DocumentChunker | None = None):
        self.chunker = chunker or DocumentChunker()

    @staticmethod
    def read_document(storage_path: str):
        path = Path(storage_path)
        suffix = path.suffix.lower()

        if suffix == ".txt":
            try:
                return path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return path.read_text(encoding="cp1252")

        if suffix == ".pdf":
            reader = PdfReader(str(path))

            if reader.is_encrypted:
                raise ValueError("Encrypted PDF files are not supported")

            text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

            if not text.strip():
                raise ValueError(
                    "No extractable text found in PDF. OCR may be required."
                )

            return text

        raise ValueError(f"Unsupported document format: {suffix}")

    def process_document(self, document_id: int) -> dict[str, int]:
        report = {
            "processed": 0,
            "failed": 0,
            "chunks_created": 0,
            "skipped": 0,
        }

        # Claim the document so another worker cannot process
        # the same pending document at the same time.
        with session_maker() as db:
            result = db.execute(
                update(Document)
                .where(
                    Document.id == document_id,
                    Document.status == "pending",
                )
                .values(status="processing")
            )
            db.commit()

            if result.rowcount != 1:
                report["skipped"] = 1
                return report

            document = db.get(Document, document_id)

            if document is None:
                report["failed"] = 1
                return report

            storage_path = document.storage_path

        try:
            text = self.read_document(storage_path)
            chunks = self.chunker.split_document(text)

            if not chunks:
                raise ValueError("Document produced no chunks")

            # Save chunks and mark the document processed
            # in the same database transaction.
            with session_maker() as db:
                document = db.get(Document, document_id)

                if document is None:
                    raise ValueError(
                        f"Document {document_id} no longer exists"
                    )

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

            report["processed"] = 1
            report["chunks_created"] = len(chunks)

            logger.info(
                "Processed document ID %s: %s chunks",
                document_id,
                len(chunks),
            )

        except Exception as exc:
            with session_maker() as db:
                document = db.get(Document, document_id)

                if document is not None:
                    document.status = "failed"
                    document.attempt_count += 1
                    document.last_error = str(exc)
                    db.commit()

            report["failed"] = 1

            logger.exception(
                "Failed to process document ID %s",
                document_id,
            )

        return report

    def process_pending_documents(self) -> dict[str, int]:
        report = {
            "processed": 0,
            "failed": 0,
            "chunks_created": 0,
        }

        with session_maker() as db:
            document_ids = db.scalars(
                select(Document.id)
                .where(Document.status == "pending")
                .order_by(Document.id)
            ).all()

        for document_id in document_ids:
            result = self.process_document(document_id)

            for key in report:
                report[key] += result[key]

        return report

