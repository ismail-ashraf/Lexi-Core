from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import or_
from sqlalchemy.orm import Session

from lexicore.rag.database.connection import get_db
from lexicore.rag.database.models import Conversation, Message


router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"],
)


class ConversationCreate(BaseModel):
    title: str = Field(default="New chat", min_length=1, max_length=255)
    document_id: int | None = Field(default=None, gt=0)


class ConversationUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    document_id: int | None = Field(default=None, gt=0)


def conversation_summary(conversation: Conversation, message_count: int) -> dict:
    return {
        "conversation_id": conversation.id,
        "title": conversation.title,
        "document_id": conversation.document_id,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
        "message_count": message_count,
    }


@router.get("/")
def list_conversations(
    q: str | None = Query(default=None, max_length=255),
    db: Session = Depends(get_db),
) -> list[dict]:
    query = db.query(Conversation)

    if q and q.strip():
        search_term = f"%{q.strip()}%"

        matching_message = (
            db.query(Message.id)
            .filter(
                Message.conversation_id == Conversation.id,
                Message.content.ilike(search_term),
            )
            .exists()
        )

        query = query.filter(
            or_(
                Conversation.title.ilike(search_term),
                matching_message,
            )
        )

    conversations = (
        query.order_by(Conversation.updated_at.desc())
        .all()
    )

    results = []

    for conversation in conversations:
        message_count = (
            db.query(Message)
            .filter(Message.conversation_id == conversation.id)
            .count()
        )

        results.append(
            conversation_summary(conversation, message_count)
        )

    return results


@router.post("/", status_code=201)
def create_conversation(
    request: ConversationCreate,
    db: Session = Depends(get_db),
) -> dict:
    if request.document_id is not None:
        from lexicore.rag.database.models import Document

        document = db.get(Document, request.document_id)

        if document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found",
            )

    conversation = Conversation(
        title=request.title.strip() or "New chat",
        document_id=request.document_id,
    )

    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return conversation_summary(conversation, 0)


@router.get("/{conversation_id}")
def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
) -> dict:
    conversation = db.get(Conversation, conversation_id)

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .all()
    )

    return {
        "conversation_id": conversation.id,
        "title": conversation.title,
        "document_id": conversation.document_id,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
        "messages": [
            {
                "message_id": message.id,
                "role": message.role,
                "content": message.content,
                "sources": message.sources or [],
                "created_at": message.created_at,
            }
            for message in messages
        ],
    }


@router.patch("/{conversation_id}")
def update_conversation(
    conversation_id: int,
    request: ConversationUpdate,
    db: Session = Depends(get_db),
) -> dict:
    conversation = db.get(Conversation, conversation_id)

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    changes = request.model_dump(exclude_unset=True)

    if "title" in changes:
        title = changes["title"]

        if title is None or not title.strip():
            raise HTTPException(
                status_code=422,
                detail="Conversation title cannot be empty",
            )

        conversation.title = title.strip()

    if "document_id" in changes:
        document_id = changes["document_id"]

        if document_id is not None:
            from lexicore.rag.database.models import Document

            document = db.get(Document, document_id)

            if document is None:
                raise HTTPException(
                    status_code=404,
                    detail="Document not found",
                )

        conversation.document_id = document_id

    conversation.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(conversation)

    message_count = (
        db.query(Message)
        .filter(Message.conversation_id == conversation.id)
        .count()
    )

    return conversation_summary(conversation, message_count)


@router.delete("/{conversation_id}")
def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
) -> dict:
    conversation = db.get(Conversation, conversation_id)

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    db.delete(conversation)
    db.commit()

    return {
        "message": "Conversation deleted successfully",
        "conversation_id": conversation_id,
    }
