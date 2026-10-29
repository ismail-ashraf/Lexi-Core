from celery import chain
from celery.signals import worker_ready
from lexicore.background.celery_app import celery_app
from lexicore.rag.chunking.processor import DocumentProcessor
from lexicore.rag.pipeline import RAGPipeline

_pipeline = None

def get_pipeline() -> RAGPipeline:
    global _pipeline

    if _pipeline is None:
        _pipeline = RAGPipeline()

    return _pipeline


@worker_ready.connect
def load_pipeline_on_worker_ready(sender=None, **kwargs):
    get_pipeline()


@celery_app.task(name="lexicore.process_document")
def process_document_task(document_id: int) -> dict:
    processor = DocumentProcessor()
    result = processor.process_document(document_id)

    if result["processed"] != 1:
        raise RuntimeError(
            f"Document processing failed or was skipped: {result}"
        )

    return result


@celery_app.task(name="lexicore.index_document")
def index_document_task(
    document_id: int,
) -> dict:
    pipeline = get_pipeline()
    return pipeline.index_document(document_id)


def process_and_index_document(document_id: int):
    workflow = chain(
        process_document_task.s(document_id),
        index_document_task.s(document_id),
    )
    return workflow.apply_async()
