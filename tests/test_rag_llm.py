
from lexicore.rag.pipeline import RAGPipeline


pipeline = RAGPipeline()

result = pipeline.ask(
    query="What is the termination notice period?",
    n_results=5,
    document_id=999999,
)

print("\nAnswer:")
print(result["answer"])

print("\nSources:")
for source in result["sources"]:
    print(
        f"Chunk: {source['chunk_id']} | "
        f"Document: {source['document_id']} | "
        f"Distance: {source['distance']}"
    )

assert result["sources"] == [], (
    "Expected no sources for a document that does not exist!"
)

print("\nPASS: No sources returned for the unknown document.")
