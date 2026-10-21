from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from lexicore.services.document_classifier import DocumentClassifier
import logging

router = APIRouter(
    prefix="/classify-document",
    tags=["Document Classification"],
)


class ClassificationRequest(BaseModel):
    text: str = Field(min_length=1)


class ClassificationResponse(BaseModel):
    label: str
    confidence: float


classifier: DocumentClassifier | None = None


def set_classifier(service: DocumentClassifier) -> None:
    global classifier
    classifier = service


@router.post("",response_model=ClassificationResponse,)
def classify_document(request: ClassificationRequest):
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classification model is not initialized.",
        )

    try:
        return classifier.predict(request.text)
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500,
            detail="Document classification failed.",
        )

