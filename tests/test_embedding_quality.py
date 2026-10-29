
from lexicore.rag.embeddings.embedder import Embedder


def test_quantized_embeddings():
    embedder = Embedder()

    texts = [
        "The agreement may be terminated with 30 days written notice.",
        "Either party may terminate this contract by giving 30 days notice.",
        "The supplier shall pay invoices within 60 days.",
    ]

    embeddings = embedder.embed_texts(texts)

    assert len(embeddings) == 3
    assert len(embeddings[0]) == 384

    # Similarity between two sentences about termination.
    similarity = sum(
        a * b for a, b in zip(embeddings[0], embeddings[1])
    )

    # Normalized embeddings should have cosine similarity in [-1, 1].
    assert -1 <= similarity <= 1

    print("Embedding dimensions:", len(embeddings[0]))
    print("Termination similarity:", round(similarity, 4))
    print("Different-topic similarity:", round(
        sum(a * b for a, b in zip(embeddings[0], embeddings[2])), 4
    ))
