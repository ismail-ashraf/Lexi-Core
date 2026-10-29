
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    def __init__(self,chunk_size: int = 1000,chunk_overlap: int = 150,):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be >= 0 and smaller than chunk_size"
            )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "; ", " ", ""],
        )

    def split_document(self, text: str) -> list[dict]:
        text = text.strip()

        if not text:
            return []

        chunks = self.splitter.split_text(text)

        return [
            {"content": chunk, "chunk_index": index}
            for index, chunk in enumerate(chunks)
            if chunk.strip()
        ]
