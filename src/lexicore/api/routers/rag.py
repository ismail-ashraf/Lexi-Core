import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
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
def ask_question(request: AskRequest):
    if not request.query.strip():
        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty",
        )

    try:
        pipeline = get_pipeline()

        result = pipeline.ask(
            query=request.query,
            n_results=request.n_results,
            document_id=request.document_id,
        )

        # The API owns source references.
        sources = result.get("sources", [])

        # Remove model-generated chunk references from the answer.
        # The frontend should display citations using `sources`.
        answer = result.get("answer", "")

        return {
            "query": request.query,
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

