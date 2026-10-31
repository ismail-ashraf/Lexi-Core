# LexiCore

### Full-Stack AI-Powered Document Intelligence & Retrieval-Augmented Generation

LexiCore is a full-stack AI-powered document intelligence system designed to transform unstructured documents into structured, searchable, and actionable information.

It combines document parsing, Natural Language Processing (NLP), Named Entity Recognition (NER), BERT-based language models, semantic search, and Retrieval-Augmented Generation (RAG) to help users extract information, understand document content, and ask context-aware questions.

LexiCore provides both an interactive frontend and a modular Python backend, bringing document processing, AI-powered search, and conversational question answering together in one application.

## Overview

LexiCore goes beyond traditional document storage and search by providing multiple layers of document understanding.

Users can interact with their documents through the frontend, while the backend handles document processing, AI inference, indexing, retrieval, and answer generation.

The system includes:

- **Document Parsing:** Extract and prepare text from uploaded documents.
- **Named Entity Recognition (NER):** Identify and extract meaningful entities from document content.
- **BERT-Based NLP:** Use transformer-based language models for contextual document understanding and classification tasks.
- **Semantic Search:** Retrieve relevant document content using vector embeddings.
- **Retrieval-Augmented Generation:** Generate answers grounded in retrieved document content.
- **Conversation Management:** Organize and maintain question-answering conversations.
- **Interactive Frontend:** Provide a user-friendly interface for document interaction and AI-powered question answering.

## Application Experience

LexiCore includes a frontend interface that allows users to interact with the document intelligence system without directly using API endpoints.

The frontend communicates with the FastAPI backend through REST API requests. The backend handles the actual document processing and AI operations, while the frontend presents the results in an accessible interface.

### Frontend Workflow

The typical user workflow is:

1. **Upload Documents**  
   Users upload their documents through the frontend interface.

2. **Document Processing**  
   The frontend sends the uploaded files to the backend. The backend handles parsing, text extraction, and document processing.

3. **AI Processing and Indexing**  
   The backend performs the required NLP operations and generates embeddings for semantic indexing.

4. **Ask Questions**  
   Users submit natural-language questions through the chat interface.

5. **Context Retrieval**  
   The backend retrieves relevant document chunks from ChromaDB using semantic similarity.

6. **Answer Generation**  
   The retrieved context is passed to the LLM through the RAG pipeline.

7. **Display Results**  
   The backend returns the generated answer and its associated source information, which the frontend displays to the user.

### Frontend Capabilities

The frontend provides the user-facing layer of LexiCore, including:

- Document upload and interaction.
- Access to document-related information.
- AI-powered question and answer interface.
- Conversation management.
- Display of generated answers and retrieved source references.

## Frontend Preview

<!-- Add actual application screenshots here -->

The following screenshots can be used to showcase the application:

| Dashboard | Document Management |
|---|---|
| ![LexiCore Dashboard](docs/images/dashboard.png) | ![Document Management](docs/images/documents.png) |

| AI Chat Interface | Conversations |
|---|---|
| ![AI Chat](docs/images/chat.png) | ![Conversations](docs/images/conversations.png) |

> Replace these image paths with actual screenshots from the frontend. Screenshots should show the real application interface.

## System Architecture

```mermaid
flowchart TD
    U[User] --> FE[LexiCore Frontend]
    FE <-->|REST API Requests / Responses| API[FastAPI Backend]

    API --> DOC[Document API]
    API --> RAG[RAG API]
    API --> CONV[Conversation API]

    DOC --> ING[Document Ingestion]
    ING --> PARSE[Document Parsing]
    PARSE --> NLP[NLP Processing]

    NLP --> NER[Named Entity Recognition]
    NLP --> BERT[BERT-Based NLP]
    NLP --> CHUNK[Text Chunking]

    CHUNK --> EMB[Embedding Generation]
    EMB --> VDB[(ChromaDB)]

    RAG --> QE[Query Embedding]
    QE --> RET[Semantic Retrieval]
    RET --> VDB
    RET --> CTX[Retrieved Context]
    CTX --> LLM[Groq LLM]
    LLM --> ANS[Generated Answer]

    DOC --> DB[(PostgreSQL)]
    CONV --> DB
    ING --> DB

    ANS --> API
    API --> FE
```

## Key Features

- **Document Ingestion:** Process uploaded documents and prepare them for downstream AI tasks.
- **Document Parsing:** Extract and prepare document content for NLP and retrieval.
- **Named Entity Recognition:** Identify relevant entities within document text.
- **BERT-Based Document Understanding:** Apply transformer-based NLP models to document-related tasks.
- **Semantic Search:** Retrieve relevant document chunks using vector embeddings.
- **Retrieval-Augmented Generation (RAG):** Generate answers using retrieved document context.
- **Document-Level Filtering:** Restrict retrieval to a specific document when requested.
- **Persistent Vector Storage:** Store embeddings and metadata in ChromaDB.
- **Conversation Management:** Manage conversations and messages through API endpoints.
- **Background Processing:** Handle document processing and indexing through background tasks.
- **RESTful API:** Expose document intelligence and RAG functionality through FastAPI.
- **Interactive Frontend:** Allow users to access the system through a web interface.

## Core AI Components

### 1. Document Parsing

The parsing layer converts uploaded documents into usable text content.

Its purpose is to prepare different document inputs for downstream NLP and retrieval operations.

The extracted content serves as input for entity extraction, document understanding, chunking, and indexing.

### 2. Named Entity Recognition (NER)

LexiCore includes an NER component for identifying and extracting meaningful entities from document text.

The system uses GLiNER-based entity extraction, with an ONNX inference path for optimized model execution.

NER transforms unstructured text into structured information that can be used in downstream document intelligence workflows.

### 3. BERT-Based NLP

LexiCore incorporates BERT-based transformer models for contextual language understanding and document-related NLP tasks.

These models provide contextual representations of text that support document classification and other language processing workflows.

### 4. Retrieval-Augmented Generation (RAG)

The RAG component allows users to ask questions about their uploaded documents.

It combines semantic retrieval with LLM-based answer generation to provide responses grounded in relevant document content.

The workflow consists of:

1. **Text Chunking:** Divide extracted document content into smaller chunks.
2. **Embedding Generation:** Convert each chunk into a vector representation.
3. **Vector Indexing:** Store embeddings and metadata in ChromaDB.
4. **Query Embedding:** Convert the user's question into an embedding.
5. **Semantic Retrieval:** Retrieve relevant chunks using vector similarity and document-level filters.
6. **Context Construction:** Prepare retrieved content for the generation stage.
7. **Answer Generation:** Send the question and retrieved context to a Groq-hosted LLM.

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| Frontend | Web Frontend |
| Relational Database | PostgreSQL |
| Vector Database | ChromaDB |
| Database Migrations | Alembic |
| NLP / Transformer Models | BERT |
| Named Entity Recognition | GLiNER |
| Model Inference Optimization | ONNX |
| Embedding Generation | Embedding Model Pipeline |
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
│   ├── package.json
│   └── ...
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
- Node.js and npm
- Groq API Key

### 1. Clone the Repository

```bash
git clone https://github.com/ismail-ashraf/Lexi-Core.git
cd LexiCore
```

### 2. Install Backend Dependencies

```bash
uv sync
```

### 3. Configure Environment Variables

Create a `.env` file in the project root and configure the environment variables required by the application.

Example:

```env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/lexicore
GROQ_API_KEY=your_groq_api_key
```

Refer to the application configuration for the complete list of supported environment variables.

**Important:** Never commit API keys, database credentials, model weights, or local database backups to version control.

### 4. Run Database Migrations

```bash
uv run alembic upgrade head
```

### 5. Start the Backend

```bash
uv run uvicorn lexicore.api.main:app --reload
```

The backend API will be available at:

```text
http://127.0.0.1:8000
```

### 6. Start the Frontend

Navigate to the frontend directory:

```bash
cd frontend
```

Install frontend dependencies:

```bash
npm install
```

Start the development server using the script configured in `frontend/package.json`.

For example, if the project uses a Vite development script:

```bash
npm run dev
```

The frontend development URL will be displayed in the terminal.

### 7. Explore the API

Open the interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## API Capabilities

| Capability | Description |
|---|---|
| Document Management | Upload, retrieve, and manage documents |
| Document Parsing | Extract and prepare document content |
| NLP Processing | Perform document understanding tasks |
| Named Entity Recognition | Extract meaningful entities from text |
| RAG Question Answering | Ask questions based on document content |
| Conversation Management | Create and manage conversations and messages |

The exact request and response schemas are available through the interactive API documentation.

## Vector Storage

LexiCore uses ChromaDB as its persistent vector store.

Each indexed chunk is stored with:

- A unique chunk identifier.
- The chunk's text.
- Its embedding vector.
- Document-level metadata.
- Chunk index metadata.

The vector collection uses cosine distance for similarity-based retrieval.

Document-level metadata enables filtering retrieved chunks by document ID.

## LLM Integration

LexiCore integrates with Groq for remote LLM inference.

The configured model is:

```text
openai/gpt-oss-20b
```

The LLM receives retrieved document context alongside the user's question. This allows the system to generate answers based on indexed document collections without requiring the LLM to run locally.

## Design Goals

LexiCore is built around the following engineering principles:

- Full-stack architecture with a dedicated frontend and backend.
- Modular AI components with clearly separated responsibilities.
- Separation of document parsing, NLP, retrieval, and generation.
- Persistent storage for document metadata and vector embeddings.
- Document-level retrieval control.
- Optimized model inference where applicable.
- API-driven frontend/backend communication.
- Background processing for document indexing.
- Configuration-driven application behavior.
- Maintainable database schema evolution.

## Future Improvements

Potential areas for further development include:

- Retrieval evaluation and benchmarking.
- Improved chunking and embedding strategies.
- NER and document classification evaluation.
- Retrieval quality optimization.
- Production deployment and observability.
- Further frontend enhancements.

## Author

**Ismail Ashraf**

---

*LexiCore — Turning Documents into Structured, Searchable Intelligence.*
