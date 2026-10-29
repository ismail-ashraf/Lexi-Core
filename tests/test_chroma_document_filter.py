
import pytest

from lexicore.rag.vector_db.chroma import ChromaVectorStore


@pytest.fixture
def vector_store(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma_test"),
        collection_name="test_document_filter",
    )

    store.add_chunks(
        ids=["chunk_1", "chunk_2", "chunk_3"],
        texts=[
            "Termination notice is 30 days.",
            "Termination notice is 60 days.",
            "Payment is due within 15 days.",
        ],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        metadatas=[
            {"document_id": 15, "chunk_index": 0},
            {"document_id": 20, "chunk_index": 0},
            {"document_id": 15, "chunk_index": 1},
        ],
    )

    return store


def test_search_returns_only_chunks_from_requested_document(vector_store):
    results = vector_store.search(
        query_embedding=[1.0, 0.0, 0.0],
        n_results=10,
        document_id=15,
    )

    assert set(results["ids"][0]) == {"chunk_1", "chunk_3"}
    assert all(
        metadata["document_id"] == 15
        for metadata in results["metadatas"][0]
    )


def test_search_filters_a_different_document(vector_store):
    results = vector_store.search(
        query_embedding=[1.0, 0.0, 0.0],
        n_results=10,
        document_id=20,
    )

    assert results["ids"][0] == ["chunk_2"]
    assert all(
        metadata["document_id"] == 20
        for metadata in results["metadatas"][0]
    )


def test_search_returns_empty_results_for_unknown_document(vector_store):
    results = vector_store.search(
        query_embedding=[1.0, 0.0, 0.0],
        n_results=10,
        document_id=999,
    )

    assert results["ids"] == [[]]
    assert results["documents"] == [[]]
