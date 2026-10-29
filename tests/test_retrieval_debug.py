
from lexicore.rag.pipeline import RAGPipeline


def test_inspect_retrieval():
    pipeline = RAGPipeline()

    query = "What is the termination notice period?"
    results = pipeline.search(query, n_results=5)

    for i, (chunk_id, distance, metadata, content) in enumerate(
        zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
            results["documents"][0],
        ),
        start=1,
    ):
        print(f"\n--- Result {i} ---")
        print("Chunk ID:", chunk_id)
        print("Metadata:", metadata)
        print("Cosine distance:", round(distance, 4))
        print("Content:", content[:400])
