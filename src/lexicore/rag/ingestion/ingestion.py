import hashlib
import logging
import re
from pathlib import Path
from sqlalchemy import select
from lexicore.rag.database.connection import session_maker
from lexicore.rag.database.models import Document
from lexicore.configrations.config import settings

logger = logging.getLogger(__name__)


class DocumentIngestion:
    def __init__(self, source_dir: str | Path | None = None):
        # Default location of the original CUAD text files
        self.source_dir = settings.CUAD_DATASET_DIR

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize text before calculating its content hash."""
        text = text.replace("\x00", "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @staticmethod
    def calculate_hash(text: str) -> str:
        """Return a SHA-256 hash of the cleaned document text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def ingest_documents(self) -> dict[str, int]:
        """
        Read text files and store document metadata in PostgreSQL.

        Original files remain on disk. The full document text is not
        stored in the documents table.
        """
        if not self.source_dir.exists():
            raise FileNotFoundError(
                f"Source directory does not exist: {self.source_dir}"
            )

        files = sorted(self.source_dir.glob("*.txt"))

        if not files:
            logger.warning("No .txt files found in %s", self.source_dir)

        report = {
            "total_files": len(files),
            "inserted": 0,
            "duplicates": 0,
            "empty_files": 0,
        }

        # Prevent duplicate content within the same ingestion run.
        seen_hashes: set[str] = set()

        try:
            with session_maker() as db:
                for file_path in files:
                    try:
                        try:
                            raw_text = file_path.read_text(encoding="utf-8")
                        except UnicodeDecodeError:
                            raw_text = file_path.read_text(encoding="cp1252")
                    except FileNotFoundError:
                        logger.warning("File disappeared before reading: %s", file_path)
                        continue

                    cleaned_text = self.clean_text(raw_text)

                    if not cleaned_text:
                        report["empty_files"] += 1
                        logger.warning(
                            "Skipping empty document: %s",
                            file_path.name,
                        )
                        continue

                    content_hash = self.calculate_hash(cleaned_text)

                    if content_hash in seen_hashes:
                        report["duplicates"] += 1
                        continue

                    seen_hashes.add(content_hash)

                    # Check whether this content already exists in PostgreSQL.
                    existing_id = db.scalar(
                        select(Document.id).where(
                            Document.content_hash == content_hash
                        )
                    )

                    if existing_id is not None:
                        report["duplicates"] += 1
                        logger.debug(
                            "Document already exists: %s",
                            file_path.name,
                        )
                        continue

                    document = Document(
                        filename=file_path.name,
                        storage_path=str(file_path.resolve()),
                        content_hash=content_hash,
                        status="pending",
                        attempt_count=0,
                        last_error=None,
                    )

                    db.add(document)
                    report["inserted"] += 1

                # Persist the new document records.
                db.commit()

        except Exception:
            logger.exception("Document ingestion failed.")
            raise

        self.print_report(report)
        return report

    @staticmethod
    def print_report(report: dict[str, int]) -> None:
        print("\n========== INGESTION REPORT ==========")
        print(f"Total files: {report['total_files']}")
        print(f"Inserted into PostgreSQL: {report['inserted']}")
        print(f"Duplicates skipped: {report['duplicates']}")
        print(f"Empty files skipped: {report['empty_files']}")
        print("======================================\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ingestion = DocumentIngestion()
    ingestion.ingest_documents()