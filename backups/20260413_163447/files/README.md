# ML/DL Agentic Chatbot with NVIDIA NIM

A production-ready chatbot web app for machine learning and deep learning Q&A with Agentic RAG, low hallucination behavior, and NVIDIA NIM integration.

## Features
- **Modern Chat UI**: Responsive React frontend with dark/light mode.
- **Agentic RAG**: Grounded retrieval from uploaded and preloaded documents.
- **NVIDIA NIM Integration**: Powered by `meta/llama-3.1-8b-instruct` via NVIDIA NIM.
- **Streaming Responses**: Token-by-token streaming for a better user experience.
- **Conversation Management**: Rename and delete chat threads.
- **File Upload**: Ingest PDF, TXT, and MD files into a local ChromaDB vector store.
- **Secure**: Key handling via environment variables.

## Tech Stack
- **Frontend**: React, TypeScript, Tailwind CSS, Zustand, Lucide React.
- **Backend**: Node.js, Express, LangChain.
- **LLM Options**: NVIDIA NIM or OpenRouter.
- **Embedding Options**: OpenAI or Hugging Face (Free).
- **Database**: Supabase (Auth & Metadata).
- **Vector DB**: ChromaDB (Local persistence).
- **LLM API**: NVIDIA NIM (OpenAI-compatible).

## Prerequisites
- Node.js 20+
- Docker & Docker Compose (for ChromaDB)
- NVIDIA API Key
- OpenAI API Key (for embeddings)
- Supabase Project

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ml-chatbot
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

Required keys:
- `NVIDIA_API_KEY`: Get it from NVIDIA NIM.
- `OPEN_ROUTER_API_KEY`: If using OpenRouter for LLM.
- `OPENAI_API_KEY`: For generating document embeddings (optional if using Hugging Face).
- `HUGGINGFACE_API_KEY`: For free embeddings via Hugging Face Inference API.
- `SUPABASE_URL` & `SUPABASE_ANON_KEY`: From your Supabase project settings.

### 3. Start ChromaDB
Run ChromaDB using Docker Compose (if you have Docker):
```bash
docker compose up -d
```

Note: this repo maps Chroma to host port `8001` (because `8000` is commonly in use on some machines). Configure `CHROMA_URL` accordingly.

If you don't have Docker, run Chroma locally via the CLI:
```bash
chroma run --host 127.0.0.1 --port 8000 --path ./data/chromadb
```

If port `8000` is already in use on your machine, pick another port (e.g. `8001`) and set `CHROMA_URL` in `.env`:
```bash
CHROMA_URL=http://localhost:8001
```

### 4. Install Dependencies
```bash
npm install
```

### 5. Ingest Seed Documents
Preload the knowledge base with sample ML/DL concepts:
```bash
npm run seed
```

### 6. Start the Application
Run both frontend and backend in development mode:
```bash
npm run dev
```
The app will be available at `http://localhost:5173`.

## Switching to Qdrant Vector DB
To switch from ChromaDB to Qdrant later:

1.  **Update Dependencies**: Install `@qdrant/js-client-rest`.
2.  **Modify Vector Store Service**: Update `api/services/vectorstore.service.ts` to use `QdrantVectorStore` instead of `Chroma`.
3.  **Update Docker Compose**: Add Qdrant service and remove ChromaDB.
4.  **Re-ingest Data**: Run `npm run seed` to re-index documents in Qdrant.

## Architecture
- `api/`: Express backend with services for RAG, LLM, and VectorStore.
- `src/`: React frontend with components and Zustand store.
- `data/`: Local storage for ChromaDB and uploaded files.
- `scripts/`: Utility scripts for data ingestion.
- `python_backend/`: Python rebuild using FastAPI + LangGraph + LangChain.

## Python Rebuild (LangGraph + LangChain)
If you want the modern Python pipeline, see [python_backend/README.md](python_backend/README.md).

## Hallucination Guardrails
The RAG pipeline uses a specialized prompt that instructs the model to answer based ONLY on the provided context. If the information is missing, it will state that it doesn't have enough grounded information rather than inventing facts.
