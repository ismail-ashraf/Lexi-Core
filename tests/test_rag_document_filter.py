
from unittest.mock import MagicMock

import pytest

from lexicore.rag.pipeline import RAGPipeline


@pytest.fixture
def pipeline():
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2, 0.3]

    vector_store = MagicMock()
    vector_store.search.return_value = {
        "ids": [["10", "20"]],
        "documents": [["Contract A chunk", "Contract B chunk"]],
        "metadatas": [[
            {"document_id": 1, "chunk_index": 0},
            {"document_id": 2, "chunk_index": 0},
        ]],
        "distances": [[0.1, 0.2]],
    }

    llm = MagicMock()
    llm.generate_answer.return_value = "Test answer"

    return RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
    )


def test_search_passes_document_id_to_vector_store(pipeline):
    pipeline.search(
        query="What is the termination notice period?",
        document_id=15,
    )

    pipeline.vector_store.search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        n_results=30,
        document_id=15,
    )


def test_search_passes_none_when_document_id_is_not_provided(pipeline):
    pipeline.search(query="What is the termination notice period?")

    pipeline.vector_store.search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        n_results=30,
        document_id=None,
    )


def test_ask_passes_document_id_to_search(pipeline):
    pipeline.search = MagicMock(return_value={
        "ids": [["10"]],
        "documents": [["Contract A chunk"]],
        "metadatas": [[{"document_id": 15, "chunk_index": 0}]],
        "distances": [[0.1]],
    })

    pipeline.ask(
        query="What is the termination notice period?",
        document_id=15,
    )

    pipeline.search.assert_called_once_with(
        query="What is the termination notice period?",
        n_results=5,
        candidate_k=30,
        document_id=15,
    )


def test_ask_uses_only_returned_context(pipeline):
    pipeline.vector_store.search.return_value = {
        "ids": [["10"]],
        "documents": [["Contract A chunk"]],
        "metadatas": [[{"document_id": 15, "chunk_index": 0}]],
        "distances": [[0.1]],
    }

    pipeline.ask(
        query="What is the termination notice period?",
        document_id=15,
    )

    context = pipeline.llm.generate_answer.call_args.kwargs["context"]

    assert "Contract A chunk" in context
    assert "Document ID: 15" in context
    assert "Contract B chunk" not in context
