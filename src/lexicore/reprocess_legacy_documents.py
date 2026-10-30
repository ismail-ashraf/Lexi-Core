
from sqlalchemy import select, update

from lexicore.rag.database.connection import session_maker
from lexicore.rag.database.models import Document
from lexicore.background.tasks import process_and_index_document


def main():
    # Find legacy processed documents.
    with session_maker() as db:
        document_ids = db.scalars(
            select(Document.id)
            .where(Document.status == "processed")
            .order_by(Document.id)
        ).all()

    if not document_ids:
        print("No legacy processed documents found.")
        return

    print(f"Found {len(document_ids)} legacy documents.")

    # Move only legacy documents to pending.
    with session_maker() as db:
        result = db.execute(
            update(Document)
            .where(
                Document.id.in_(document_ids),
                Document.status == "processed",
            )
            .values(status="pending")
        )
        db.commit()

        queued_ids = document_ids[:result.rowcount]

    print(f"Reset {len(queued_ids)} documents to pending.")

    # Submit each document through the existing Celery workflow.
    for document_id in queued_ids:
        try:
            task_result = process_and_index_document(document_id)

            print(
                f"Queued document {document_id} "
                f"| Task ID: {task_result.id}"
            )

        except Exception as exc:
            print(
                f"Failed to queue document {document_id}: {exc}"
            )

    print("Reprocessing submission completed.")


if __name__ == "__main__":
    main()
