from lexicore.rag.llm.llm_service import LLMService

llm = LLMService()

answer = llm.generate_answer(
    query="What is the termination notice period?",
    context=(
        "[Chunk ID: 123] Either party may terminate this agreement "
        "by providing 30 days' written notice."
    ),
)

print(answer)
