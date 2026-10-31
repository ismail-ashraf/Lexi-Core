# LexiCore

### AI-Powered Document Intelligence & Retrieval-Augmented Generation

LexiCore is an AI-powered document intelligence backend designed to process, index, and retrieve information from documents, enabling users to ask questions and receive context-aware answers grounded in their uploaded content.

The system combines document ingestion, vector search, and Large Language Models to provide a retrieval-augmented generation (RAG) pipeline through a RESTful API.

## Overview

LexiCore provides a backend foundation for working with large collections of documents. It handles document processing, text chunking, embedding generation, vector storage, semantic retrieval, and answer generation.

The RAG pipeline retrieves relevant document chunks before generating an answer, helping the model stay grounded in the available context rather than relying solely on its internal knowledge.

## Key Features

- **Document Ingestion:** Process documents and prepare their content for indexing.
- **Semantic Search:** Retrieve relevant document chunks using vector embeddings.
- **Retrieval-Augmented Generation (RAG):** Generate answers using retrieved document context.
- **Document-Level Filtering:** Restrict retrieval to a specific document when requested.
- **Persistent Vector Storage:** Store embeddings and metadata in ChromaDB.
- **Conversation Management:** Manage conversations and messages through API endpoints.
- **Background Processing:** Handle document processing and indexing through background tasks.
- **RESTful API:** Expose document and RAG functionality through FastAPI.
- **Database Migrations:** Manage relational database schema changes with Alembic.

## Architecture

```mermaid
flowchart TD
    A[Client / Frontend] --> B[FastAPI]
    B --> C[Document API]
    B --> D[RAG API]
    B --> E[Conversation API]

    C --> F[Document Processing]
    F --> G[Text Chunking]
    G --> H[Embedding Generation]
    H --> I[ChromaDB]

    D --> J[Query Embedding]
    J --> K[Semantic Retrieval]
    K --> I
    K --> L[Retrieved Context]
    L --> M[Groq LLM]
    M --> N[Generated Answer]

    C --> O[(PostgreSQL)]
    E --> O
    F --> O
```

## RAG Pipeline

The RAG workflow consists of the following stages:

1. **Document Processing**  
   Uploaded documents are processed and their text is extracted.

2. **Chunking**  
   Document content is divided into smaller chunks suitable for embedding and retrieval.

3. **Embedding Generation**  
   Each chunk is converted into a vector representation.

4. **Vector Indexing**  
   Chunk embeddings and metadata are stored in ChromaDB using cosine similarity.

5. **Query Embedding**  
   The user's question is converted into an embedding using the same embedding pipeline.

6. **Semantic Retrieval**  
   Relevant chunks are retrieved from the vector store. Retrieval can be restricted to a specific document.

7. **Answer Generation**  
   The retrieved context is passed to a Groq-hosted LLM to generate a context-aware response.

The generation prompt is designed to keep answers grounded in the retrieved context and reduce unsupported responses.

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| Relational Database | PostgreSQL |
| Vector Database | ChromaDB |
| Database Migrations | Alembic |
| Embeddings | Embedding Model Pipeline |
| LLM Inference | Groq API |
| LLM | `openai/gpt-oss-20b` |
| Package Management | uv |
| API Documentation | OpenAPI / Swagger UI |

## Project Structure

```text
LexiCore/
├── alembic/
│   └── versions/
├── frontend/
├── src/
│   └── lexicore/
│       ├── api/
│       │   ├── main.py
│       │   └── routers/
│       │       ├── documents.py
│       │       ├── rag.py
│       │       └── conversations.py
│       ├── background/
│       │   └── tasks.py
│       ├── configrations/
│       │   └── config.py
│       └── rag/
│           ├── database/
│           ├── ingestion/
│           ├── pipeline.py
│           └── vector_db/
│               └── chroma.py
├── pyproject.toml
├── uv.lock
├── alembic.ini
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.12+
- uv
- PostgreSQL
- Groq API Key

### 1. Clone the Repository

```bash
[https://github.com/ismail-ashraf/LexiCore.git]
cd LexiCore
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Configure Environment Variables

Create a `.env` file in the project root and configure the environment variables required by the application.

Typical configuration includes:

```env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/lexicore
GROQ_API_KEY=your_groq_api_key
```

Refer to the application configuration for the complete list of supported environment variables.

**Important:** Never commit API keys, database credentials, or local database backups to version control.

### 4. Run Database Migrations

```bash
uv run alembic upgrade head
```

### 5. Start the API Server

```bash
uv run uvicorn lexicore.api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### 6. Explore the API

Open the interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## API Capabilities

LexiCore exposes API endpoints for:

| Capability | Description |
|---|---|
| Document Management | Upload, retrieve, and manage documents |
| Document Processing | Process and index uploaded content |
| RAG Question Answering | Ask questions based on document content |
| Conversation Management | Create and manage conversations and messages |

The exact request and response schemas are available through the interactive API documentation.

## Vector Storage

LexiCore uses ChromaDB as its persistent vector store.

Each indexed chunk is stored with:

- A unique chunk identifier
- The chunk's text
- Its embedding vector
- Document-level metadata
- Chunk index metadata

The vector collection uses cosine distance for similarity-based retrieval.

Document-level metadata enables filtering retrieved chunks by document ID.

## LLM Integration

LexiCore integrates with Groq for remote LLM inference.

The default configured model is:

```text
openai/gpt-oss-20b
```

The LLM receives retrieved document context alongside the user's question. This design allows the system to generate answers based on the indexed document collection without requiring the LLM to run locally.

## Development

Install dependencies:

```bash
uv sync
```

Run the API in development mode:

```bash
uv run uvicorn lexicore.api.main:app --reload
```

Apply database migrations:

```bash
uv run alembic upgrade head
```

## Design Goals

LexiCore is built around several backend and AI engineering principles:

- Separation of document processing, retrieval, and generation responsibilities.
- Persistent storage for document metadata and vector embeddings.
- Document-level retrieval control.
- API-driven architecture.
- Background processing for document indexing.
- Configuration-driven application behavior.
- Maintainable database schema evolution.

## Future Improvements

Potential areas for further development include:

- Retrieval evaluation and benchmarking.
- Improved chunking and embedding strategies.
- Retrieval quality optimization.
- Production deployment and observability.
- Improved frontend experience.

## Author

**Ismail Ashraf**

---

*LexiCore — Document Intelligence through Retrieval-Augmented Generation.*
