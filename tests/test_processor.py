
from sqlalchemy import select

from lexicore.rag.database.connection import session_maker
from lexicore.rag.database.models import Document
from lexicore.rag.chunking.processor import DocumentProcessor


with session_maker() as db:
    document = db.scalars(
        select(Document)
        .where(Document.status == "pending")
        .limit(1)
    ).first()

    if document is None:
        print("No pending documents found.")
    else:
        document_id = document.id
        print(f"Testing document ID: {document_id}")
        processor = DocumentProcessor()
        print("Processor currently processes all pending documents.")
