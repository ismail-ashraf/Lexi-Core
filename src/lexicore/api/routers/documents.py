
import hashlib
from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from lexicore.background.tasks import process_and_index_document
from lexicore.rag.database.connection import get_db
from lexicore.rag.database.models import Document


UPLOAD_DIR = Path("data/uploads")
ALLOWED_EXTENSIONS = {".txt", ".pdf"}

ACTIVE_STATUSES = {"pending", "processing", "indexing"}

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.get("/")
def get_documents(
    db: Session = Depends(get_db),
) -> list[dict]:
    documents = (
        db.query(Document)
        .order_by(Document.id.desc())
        .all()
    )

    return [
        {
            "document_id": document.id,
            "filename": document.filename,
            "status": document.status,
            "attempt_count": document.attempt_count,
            "last_error": document.last_error,
        }
        for document in documents
    ]


@router.get("/{document_id}")
def get_document_status(
    document_id: int,
    db: Session = Depends(get_db),
) -> dict:
    document = db.get(Document, document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return {
        "document_id": document.id,
        "filename": document.filename,
        "status": document.status,
        "attempt_count": document.attempt_count,
        "last_error": document.last_error,
    }


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict:
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .txt and .pdf files are supported currently",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid4().hex}{suffix}"
    file_path = UPLOAD_DIR / filename
    committed = False

    try:
        file_path.write_bytes(content)

        document = Document(
            filename=file.filename or filename,
            storage_path=str(file_path.resolve()),
            content_hash=hashlib.sha256(content).hexdigest(),
            status="pending",
            attempt_count=0,
            last_error=None,
        )

        db.add(document)
        db.commit()
        db.refresh(document)
        committed = True

    except Exception as exc:
        db.rollback()

        if not committed:
            file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded document",
        ) from exc

    try:
        task = process_and_index_document(document.id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Document {document.id} was uploaded successfully, "
                "but processing could not be queued. "
                "Retry using the document processing endpoint."
            ),
        ) from exc

    return {
        "message": "Document uploaded; processing and indexing queued",
        "document_id": document.id,
        "task_id": task.id,
        "status": "queued",
    }


@router.delete("/{document_id}")
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
) -> dict:
    document = db.get(Document, document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    if document.status in ACTIVE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot delete a document while its status is "
                f"'{document.status}'. Wait until processing finishes."
            ),
        )

    storage_path = Path(document.storage_path)

    # Import lazily to avoid router import cycles.
    from lexicore.api.routers.rag import get_pipeline

    try:
        pipeline = get_pipeline()
        pipeline.vector_store.delete_document(document_id)
    except Exception:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to remove document vectors from ChromaDB",
        ) from None

    try:
        db.delete(document)
        db.commit()
    except Exception:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Vectors were removed, but the database deletion failed. "
                "The document record may still exist."
            ),
        ) from None

    file_deleted = True

    try:
        storage_path.unlink(missing_ok=True)
    except OSError:
        file_deleted = False

    return {
        "message": "Document deleted from the database and vector store",
        "document_id": document_id,
        "file_deleted": file_deleted,
    }
