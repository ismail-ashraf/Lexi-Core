
from lexicore.rag.pipeline import RAGPipeline


def main():
    pipeline = RAGPipeline()

    query = "What is the initial term of the agreement?"
    target_document_id = 410
    target_chunk_index = 23

    query_embedding = pipeline.embedder.embed_query(query)

    # Retrieve chunks from the selected document only.
    collection = pipeline.vector_store.collection

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=100,
        where={"document_id": target_document_id},
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0] or []
    metadatas = results["metadatas"][0] or []
    distances = results["distances"][0] or []

    candidates = [
        {
            "text": text,
            "metadata": metadata,
            "distance": distance,
        }
        for text, metadata, distance in zip(
            documents, metadatas, distances
        )
    ]

    print(f"\nQUERY: {query}")
    print(f"Target document: {target_document_id}")
    print(f"Retrieved chunks: {len(candidates)}")

    if not candidates:
        print("No chunks found. Check the document_id metadata type.")
        return

    def print_results(title, items, score_key=None):
        print("\n" + "=" * 90)
        print(title)
        print("=" * 90)

        for rank, item in enumerate(items, start=1):
            metadata = item["metadata"]
            chunk_index = metadata.get("chunk_index")

            marker = (
                "  <-- TARGET"
                if str(chunk_index) == str(target_chunk_index)
                else ""
            )

            score_text = (
                f" | Score: {item[score_key]:.4f}"
                if score_key and item.get(score_key) is not None
                else f" | Distance: {item['distance']:.4f}"
            )

            print(
                f"\nRank: {rank}"
                f" | Document: {metadata.get('document_id')}"
                f" | Chunk: {chunk_index}"
                f"{score_text}{marker}"
            )
            print(item["text"])

    # Baseline: Chroma's similarity order.
    print_results("BASELINE — VECTOR SEARCH", candidates)

    # Rerank the same candidates.
    rerank_results = {
        "documents": [[item["text"] for item in candidates]],
        "metadatas": [[item["metadata"] for item in candidates]],
        "distances": [[item["distance"] for item in candidates]],
        "ids": [[str(i) for i in range(len(candidates))]],
    }

    reranked = pipeline.reranker.rerank(
        query=query,
        results=rerank_results,
        top_k=len(candidates),
    )

    reranked_docs = reranked["documents"][0]
    reranked_metas = reranked["metadatas"][0]
    reranked_scores = reranked["reranker_scores"][0]

    reranked_items = [
        {
            "text": text,
            "metadata": metadata,
            "score": float(score),
            "distance": 0.0,
        }
        for text, metadata, score in zip(
            reranked_docs, reranked_metas, reranked_scores
        )
    ]

    print_results(
        "AFTER RERANKING",
        reranked_items,
        score_key="score",
    )

    # Report the target chunk's position before and after reranking.
    def find_target_rank(items):
        for rank, item in enumerate(items, start=1):
            if (
                str(item["metadata"].get("chunk_index"))
                == str(target_chunk_index)
            ):
                return rank
        return None

    baseline_rank = find_target_rank(candidates)
    reranked_rank = find_target_rank(reranked_items)

    print("\n" + "=" * 90)
    print("TARGET CHUNK RANK COMPARISON")
    print("=" * 90)
    print(f"Target chunk: {target_chunk_index}")
    print(f"Baseline rank: {baseline_rank}")
    print(f"Reranked rank: {reranked_rank}")

    if baseline_rank is None:
        print("Target chunk was not retrieved by Chroma.")
    elif reranked_rank is None:
        print("Target chunk was missing from reranked results.")
    elif reranked_rank < baseline_rank:
        print("Reranker improved the target chunk's rank.")
    elif reranked_rank > baseline_rank:
        print("Reranker lowered the target chunk's rank.")
    else:
        print("Target chunk's rank did not change.")


if __name__ == "__main__":
    main()
