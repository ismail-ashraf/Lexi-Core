
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from lexicore.rag.database.connection import get_db
from lexicore.rag.database.models import Conversation, Message
from lexicore.rag.pipeline import RAGPipeline


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])

_pipeline = None


def get_pipeline() -> RAGPipeline:
    global _pipeline

    if _pipeline is None:
        _pipeline = RAGPipeline()

    return _pipeline


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    document_id: int | None = Field(default=None, gt=0)
    conversation_id: int | None = Field(default=None, gt=0)
    n_results: int = Field(default=5, ge=1, le=20)


class SourceResponse(BaseModel):
    chunk_id: str
    document_id: int
    chunk_index: int
    distance: float
    content: str


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceResponse]


@router.post("/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    db: Session = Depends(get_db),
):
    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty",
        )

    conversation = None

    if request.conversation_id is not None:
        conversation = db.get(Conversation, request.conversation_id)

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found",
            )

        conversation.document_id = request.document_id
        conversation.updated_at = datetime.now(timezone.utc)

        if conversation.title == "New chat":
            conversation.title = query[:60].strip() or "New chat"

        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=query,
        )

        db.add(user_message)

        try:
            db.commit()
        except Exception:
            db.rollback()
            logger.exception("Failed to save user message")
            raise HTTPException(
                status_code=500,
                detail="Failed to save conversation message",
            ) from None

    try:
        pipeline = get_pipeline()

        result = pipeline.ask(
            query=query,
            n_results=request.n_results,
            document_id=request.document_id,
        )

        sources = result.get("sources", [])
        answer = result.get("answer", "")

        if not isinstance(answer, str):
            answer = str(answer)

        if conversation is not None:
            assistant_message = Message(
                conversation_id=conversation.id,
                role="assistant",
                content=answer,
                sources=sources,
            )

            conversation.updated_at = datetime.now(timezone.utc)

            db.add(assistant_message)

            try:
                db.commit()
            except Exception:
                db.rollback()
                logger.exception("Failed to save assistant message")
                raise HTTPException(
                    status_code=500,
                    detail="Answer generated but failed to save it",
                ) from None

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Failed to answer RAG query")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer",
        ) from None
