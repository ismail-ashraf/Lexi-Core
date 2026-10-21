from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from lexicore.services.entity_extractor import NERService
import logging

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/ner",
    tags=["Named Entity Recognition"],
)


class NERRequest(BaseModel):
    text: str = Field(min_length=1)


class EntityResponse(BaseModel):
    text: str
    label: str
    score: float
    start: int | None = None
    end: int | None = None


class NERResponse(BaseModel):
    entities: list[EntityResponse]


ner_service: NERService | None = None


def set_ner_service(service: NERService) -> None:
    global ner_service
    ner_service = service


@router.post("", response_model=NERResponse)
def recognize_entities(request: NERRequest):
    if ner_service is None:
        raise HTTPException(
            status_code=503,
            detail="NER model is not initialized.",
        )

    try:
        entities = ner_service.predict(text=request.text)
        return {"entities": entities}


    except Exception:
        logger.exception("NER inference failed")
        raise HTTPException(
            status_code=500,
            detail="NER inference failed. Check server logs.",
        )


