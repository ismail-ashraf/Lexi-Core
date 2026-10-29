
from lexicore.rag.pipeline import RAGPipeline


def test_rag_evaluation():
    pipeline = RAGPipeline()

    # Case 1: Existing contract
    result = pipeline.ask(
        query="What is the termination notice period?",
        n_results=5,
        document_id=133,
    )

    assert result["sources"], "Expected relevant sources"
    assert all(
        source["document_id"] == 133
        for source in result["sources"]
    )

    # Case 2: Non-existing contract
    result = pipeline.ask(
        query="What is the termination notice period?",
        document_id=999999,
    )

    assert result["sources"] == []
    assert "could not find relevant contract information" in (
        result["answer"].lower()
    )

    # Case 3: Global search across contracts
    result = pipeline.ask(
        query="What is the termination notice period?",
        n_results=5,
    )

    assert result["sources"], "Expected sources in global search"
    assert len(
        {source["document_id"] for source in result["sources"]}
    ) > 1, "Expected sources from multiple contracts"

    print("PASS: All RAG evaluation cases passed.")


if __name__ == "__main__":
    test_rag_evaluation()
