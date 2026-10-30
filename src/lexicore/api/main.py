
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from lexicore.api.routers.health import router as health_router
from lexicore.api.routers.ner import router as ner_router
from lexicore.api.routers import classification, ner
from lexicore.api.routers.classification import router as classification_router
from lexicore.api.routers.documents import router as documents_router
from lexicore.api.routers.rag import router as rag_router
from lexicore.api.routers.conversations import router as conversations_router

from lexicore.services.document_classifier import DocumentClassifier
from lexicore.services.entity_extractor import NERService


@asynccontextmanager
async def lifespan(_: FastAPI):
    classification_service = DocumentClassifier()
    ner_service = NERService()

    classification.set_classifier(classification_service)
    ner.set_ner_service(ner_service)
    yield


app = FastAPI(
    title="LexiCore API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(classification_router)
app.include_router(ner_router)
app.include_router(documents_router)
app.include_router(rag_router)
app.include_router(conversations_router)
