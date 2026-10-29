from fastapi import FastAPI
from lexicore.api.routers.health import router as health_router
from lexicore.api.routers.ner import router as ner_router
from lexicore.api.routers import classification
from lexicore.api.routers import ner
from lexicore.api.routers.classification import router as classification_router
from lexicore.services.document_classifier import DocumentClassifier
from lexicore.services.entity_extractor import NERService
from contextlib import asynccontextmanager
from lexicore.api.routers.documents import router as documents_router
from lexicore.api.routers.rag import router as rag_router

@asynccontextmanager
async def lifespan(_:FastAPI):
    classification_service = DocumentClassifier()
    ner_service = NERService()

    classification.set_classifier(classification_service)
    ner.set_ner_service(ner_service)
    yield

app = FastAPI(title="LexiCore API",lifespan=lifespan)
app.include_router(health_router)
app.include_router(classification_router)
app.include_router(ner_router)
app.include_router(documents_router)
app.include_router(rag_router)