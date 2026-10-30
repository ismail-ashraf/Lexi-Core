
import logging

from sqlalchemy import delete, select, update

from lexicore.parsing.document_parser import DocumentParser
from lexicore.rag.database.connection import session_maker
from lexicore.rag.database.models import (
    Document,
    Chunk,
    ClassificationResult,
    DocumentEntity,
)
from lexicore.rag.chunking.chunker import DocumentChunker
from lexicore.services.document_classifier import DocumentClassifier
from lexicore.services.entity_extractor import NERService

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(
        self,
        chunker: DocumentChunker | None = None,
        classifier: DocumentClassifier | None = None,
        ner_service: NERService | None = None,
        parser: DocumentParser | None = None,
    ):
        self.chunker = chunker or DocumentChunker()
        self.classifier = classifier or DocumentClassifier()
        self.ner_service = ner_service or NERService()
        self.parser = parser or DocumentParser()

    def process_document(self, document_id: int) -> dict[str, int]:
        report = {
            "processed": 0,
            "failed": 0,
            "chunks_created": 0,
            "skipped": 0,
        }

        # 1. Atomically claim the pending document.
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
            # 2. Parse the document into text.
            text = self.parser.parse(storage_path)

            if not text.strip():
                raise ValueError("Document contains no text")

            # 3. Run document classification and NER.
            classification = self.classifier.predict(text)
            entities = self.ner_service.predict(text)

            # 4. Split the extracted text into chunks.
            chunks = self.chunker.split_document(text)

            if not chunks:
                raise ValueError("Document produced no chunks")

            # 5. Persist classification, entities, and chunks.
            with session_maker() as db:
                document = db.get(Document, document_id)

                if document is None:
                    raise ValueError(
                        f"Document {document_id} no longer exists"
                    )

                existing_classification = db.scalar(
                    select(ClassificationResult).where(
                        ClassificationResult.document_id == document_id
                    )
                )

                if existing_classification is None:
                    db.add(
                        ClassificationResult(
                            document_id=document_id,
                            label=classification["label"],
                            confidence=classification["confidence"],
                        )
                    )
                else:
                    existing_classification.label = classification["label"]
                    existing_classification.confidence = (
                        classification["confidence"]
                    )

                # Replace previous analysis results.
                db.execute(
                    delete(DocumentEntity).where(
                        DocumentEntity.document_id == document_id
                    )
                )
                db.execute(
                    delete(Chunk).where(
                        Chunk.document_id == document_id
                    )
                )

                for entity in entities:
                    db.add(
                        DocumentEntity(
                            document_id=document_id,
                            text=entity["text"],
                            label=entity["label"],
                            score=entity["score"],
                            start=entity.get("start"),
                            end=entity.get("end"),
                        )
                    )

                for chunk_data in chunks:
                    db.add(
                        Chunk(
                            document_id=document_id,
                            content=chunk_data["content"],
                            chunk_index=chunk_data["chunk_index"],
                        )
                    )

                # Celery's next task handles Chroma indexing.
                document.status = "processing"
                document.last_error = None

                db.commit()

            report["processed"] = 1
            report["chunks_created"] = len(chunks)

            logger.info(
                "Processed document ID %s: %s chunks, %s entities, "
                "classification=%s",
                document_id,
                len(chunks),
                len(entities),
                classification["label"],
            )

        except Exception as exc:
            logger.exception(
                "Failed to process document ID %s",
                document_id,
            )

            with session_maker() as db:
                document = db.get(Document, document_id)

                if document is not None:
                    document.status = "failed"
                    document.attempt_count += 1
                    document.last_error = str(exc)
                    db.commit()

            report["failed"] = 1

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
