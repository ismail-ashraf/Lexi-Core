# LexiCore

### Full-Stack AI-Powered Document Intelligence & Retrieval-Augmented Generation

LexiCore is a full-stack AI-powered document intelligence system designed to transform unstructured documents into structured, searchable, and actionable information.

It combines document parsing, Natural Language Processing (NLP), Named Entity Recognition (NER), BERT-based language models, semantic search, and Retrieval-Augmented Generation (RAG) to help users extract information, understand document content, and ask context-aware questions.

LexiCore provides an interactive frontend and a modular Python backend, supported by asynchronous background processing using Celery and Redis.

## Overview

LexiCore goes beyond traditional document storage and search by providing multiple layers of document understanding.

Users interact with their documents through the frontend, while the backend handles document processing, AI inference, indexing, retrieval, and answer generation.

The system includes:

- **Document Parsing:** Extract and prepare text from uploaded documents.
- **Named Entity Recognition (NER):** Identify and extract meaningful entities from document content.
- **BERT-Based NLP:** Use transformer-based language models for contextual document understanding and classification tasks.
- **Semantic Search:** Retrieve relevant document content using vector embeddings.
- **Retrieval-Augmented Generation:** Generate answers grounded in retrieved document content.
- **Asynchronous Processing:** Execute background jobs using Celery workers and Redis.
- **Conversation Management:** Organize and maintain question-answering conversations.
- **Interactive Frontend:** Provide a user-friendly interface for document interaction and AI-powered question answering.

## Application Experience

LexiCore includes a frontend interface that allows users to interact with the document intelligence system without directly using API endpoints.

The frontend communicates with the FastAPI backend through REST API requests. The backend handles document processing and AI operations, while the frontend presents the results in an accessible interface.

### Frontend Workflow

The typical user workflow is:

1. **Upload Documents:** Users upload documents through the frontend.
2. **Document Processing:** The frontend sends uploaded files to the FastAPI backend.
3. **Background Execution:** The backend dispatches processing tasks to Celery through Redis.
4. **AI Processing and Indexing:** A separate worker executes the required processing, NLP, and indexing operations.
5. **Ask Questions:** Users submit natural-language questions through the chat interface.
6. **Context Retrieval:** The backend retrieves relevant document chunks from ChromaDB using semantic similarity.
7. **Answer Generation:** Retrieved context is passed to the LLM through the RAG pipeline.
8. **Display Results:** The backend returns generated answers and associated source information, which the frontend displays.

### Frontend Capabilities

- Document upload and interaction.
- Access to document-related information.
- AI-powered question and answer interface.
- Conversation management.
- Display of generated answers and retrieved source references.

## Frontend Preview

LexiCore provides an interactive web interface for managing documents and communicating with the AI-powered document intelligence system.

Explore the application screenshots below:

**Quick Navigation**

- [Chat Interface](#1-chat-interface)
- [Document Management](#2-document-management)
- [AI Conversation](#3-ai-conversation)

### 1. Chat Interface

The chat interface provides a clean starting point for interacting with LexiCore's AI assistant.

[![LexiCore Chat Interface](/docs/images/chat_interface.png)](/docs/images/chat_interface.png)

### 2. Document Management

The document management interface allows users to upload and manage their documents and access document-related information.

[![LexiCore Document Management](/docs/images/document_managment.png)](/docs/images/document_managment.png)

### 3. AI Conversation

Users can ask questions about their uploaded documents and receive context-aware answers generated through the RAG pipeline.

[![LexiCore AI Conversation](/docs/images/ai_conversation.png)](/docs/images/ai_conversation.png)

*Click any screenshot to open the full-size image.*

## System Architecture

LexiCore follows a modular architecture that separates API handling, asynchronous task execution, AI processing, and data persistence.

```mermaid
flowchart TD
    U[User] <--> FE[LexiCore Frontend]
    FE <-->|REST API| API[FastAPI Backend]

    API --> DOC[Document API]
    API --> RAG[RAG API]
    API --> CONV[Conversation API]

    DOC --> ING[Document Ingestion]
    ING --> PARSE[Document Parsing]
    PARSE --> NLP[NLP Processing]

    NLP --> NER[GLiNER NER]
    NLP --> BERT[BERT-Based NLP]
    NLP --> CHUNK[Text Chunking]

    ING --> CELERY[Celery Task Queue]
    CELERY <--> REDIS[(Redis Broker)]
    CELERY --> WORKER[Separate Celery Worker]

    WORKER --> PARSE
    WORKER --> NLP
    WORKER --> EMB[Embedding Generation]
    EMB --> VDB[(ChromaDB)]

    RAG --> QE[Query Embedding]
    QE --> RET[Semantic Retrieval]
    RET --> VDB
    RET --> CTX[Retrieved Context]
    CTX --> LLM[Groq LLM]
    LLM --> ANS[Generated Answer]

    DOC --> DB[(PostgreSQL)]
    CONV --> DB
    WORKER --> DB

    ANS --> API
    API --> FE
```

## Asynchronous Background Processing

LexiCore uses **Celery and Redis** to separate long-running background operations from the FastAPI request-handling process.

Instead of executing document processing and indexing directly inside an API request, the backend can dispatch tasks to a Celery worker.

### How It Works

1. **Task Dispatch:** FastAPI submits a background task to Celery.
2. **Message Brokering:** Redis acts as the message broker and holds queued task messages.
3. **Task Execution:** A separate Celery worker consumes queued tasks and executes them.
4. **Database Updates:** The worker updates document processing status and persists relevant results.
5. **Vector Indexing:** The worker generates embeddings and stores document chunks in ChromaDB.
6. **Completion:** The document processing workflow updates its final status.

### Why Celery and Redis?

- **Non-blocking API:** Long-running document processing does not need to execute inside the API request lifecycle.
- **Process Separation:** API serving and background execution run as separate processes.
- **Task Queuing:** Redis provides a broker for passing tasks between the API and workers.
- **Scalability:** Additional workers can be started to handle more background workloads independently.
- **Separation of Responsibilities:** FastAPI handles HTTP requests, while Celery workers execute background jobs.

### Worker Architecture

```mermaid
flowchart LR
    A[FastAPI] -->|Dispatch Task| B[(Redis)]
    B -->|Consume Task| C[Celery Worker]
    C --> D[Document Processing]
    D --> E[Embedding Generation]
    E --> F[(ChromaDB)]
    C --> G[(PostgreSQL)]
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
- **Asynchronous Task Execution:** Process background jobs using Celery and Redis.
- **Conversation Management:** Manage conversations and messages through API endpoints.
- **RESTful API:** Expose document intelligence and RAG functionality through FastAPI.
- **Interactive Frontend:** Allow users to access the system through a web interface.

## Core AI Components

### 1. Document Parsing

The parsing layer converts uploaded documents into usable text content.

Its purpose is to prepare document inputs for downstream NLP and retrieval operations.

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
| Task Queue | Celery |
| Message Broker | Redis |
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
├── docs/
│   └── images/
│       ├── chat-interface.png
│       ├── documents.png
│       └── ai-conversation.png
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
│       │   ├── celery_app.py
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
- Redis
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

Configure the Redis broker URL using the environment variable expected by your Celery configuration.

**Important:** Never commit API keys, database credentials, model weights, or local database backups to version control.

### 4. Run Database Migrations

```bash
uv run alembic upgrade head
```

### 5. Start the FastAPI Backend

Open a terminal in the project root:

```bash
uv run uvicorn lexicore.api.main:app --reload
```

The backend API will be available at:

```text
http://127.0.0.1:8000
```

### 6. Start the Celery Worker

Open a **separate terminal** in the project root and run:

```bash
uv run celery -A lexicore.background.celery_app:celery_app worker --loglevel=info --pool=solo
```

The worker will connect to Redis and listen for queued tasks.

Keep this terminal running while using background processing features.

### 7. Start the Frontend

Open another terminal:

```bash
cd frontend
```

Install frontend dependencies:

```bash
npm install
```

Start the development server using the script configured in `frontend/package.json`.

For example, if the project uses Vite:

```bash
npm run dev
```

The frontend development URL will be displayed in the terminal.

### 8. Explore the API

Open the interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Running the Complete Application

For local development, the application components run as separate processes:

| Component | Responsibility |
|---|---|
| Redis | Message broker for background tasks |
| FastAPI | Handles HTTP requests and API operations |
| Celery Worker | Executes asynchronous document processing tasks |
| Frontend | Provides the user interface |
| PostgreSQL | Stores relational application data |
| ChromaDB | Stores vector embeddings and document chunks |

Make sure Redis, PostgreSQL, FastAPI, and the Celery worker are running before testing workflows that depend on background processing.

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
- Asynchronous background execution using Celery and Redis.
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
