# RAG Tool

A Retrieval-Augmented Generation (RAG) service that provides semantic search and question-answering capabilities.

## Overview

FastAPI backend service that leverages LlamaIndex for document processing and retrieval, using LlamaIndex's default in-memory vector store and OpenAI's models for embeddings and LLM generation.

## Key Features

- Collection-based document management with metadata and tags support
- Semantic search with similarity scoring and source attribution
- OpenAI integration (text-embedding-3-small for embeddings, gpt-4o for generation)
- PDF document processing with PyMuPDF
- Simple in-memory vector store (no external dependencies)

### Running the Backend Locally

To run the backend server locally without Docker:

1. **Set up environment variables:**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and set your OpenAI API key
   # OPENAI_API_KEY=your_openai_api_key_here
   ```

2. **Install Python dependencies:**

   ```bash
   cd src/backend
   pip install -r requirements.txt
   ```

> Note: We recommend working in a virtual environment for this

3. **Run the server:**
   ```bash
   python api.py
   ```

The server will start on `http://localhost:8080` by default.

**Alternative: Using uvicorn directly**
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8080 --reload
   ```

The app runs by default on PORT 8080.

You should then be able to navigate to the following page:

    http://localhost:8080/docs

> **Note**: All data (collections, documents, and embeddings) is stored in memory. This means data will be lost when the server is restarted. This is suitable for development and testing, but for production use, consider implementing a persistent storage solution.

### CLI Usage
You can access the API using REST endpoints from the CLI. The `/health` endpoint should return a 200:

    $ curl http://localhost:8080/health
    {"status":"ok"}

You can also ping the collections endpoint:

    $ curl http://localhost:8080/collections
    {"collections":[],"total":0}%

## Technology Stack

- **Backend**: FastAPI
- **Vector Store**: LlamaIndex SimpleVectorStore (in-memory)
- **Document Processing**: LlamaIndex
- **AI Models**: OpenAI
- **Language**: Python 3.x

## API Documentation

Interactive API documentation is available at `http://localhost:8080/docs` when running locally.



### Available Endpoints

- **Health Check**
  - `GET /health` — Public health check endpoint

- **Collections Management**
  - `GET /collections` — List all collections
  - `POST /collections/{collection_name}` — Create a new collection
  - `DELETE /collections/{collection_name}` — Delete a collection

- **Document Management**
  - `POST /collections/{collection_name}/add-pdf` — Add a PDF document to a collection
    - Parameters: `file` (PDF), `source_id` (optional), `tags` (optional), `extras` (optional)
  - `POST /collections/{collection_name}/add-url` — Add content from a URL to a collection
    - Parameters: `url` (string), `source_id` (optional), `tags` (optional), `extras` (optional)

- **Source Management**
  - `GET /collections/{collection_name}/sources` — List all sources in a collection
  - `DELETE /collections/{collection_name}/sources/{source_id}` — Delete source content
  - `GET /collections/{collection_name}/sources/{source_id}/chunks` — Get source chunks

- **Querying**
  - `GET /collections/{collection_name}/query` — Semantic search
    - Parameters: `q`, `top_k`, `tags`, `source_id`, `page_number`
  - `GET /collections/{collection_name}/answer` — Question answering
    - Parameters: `q`, `top_k`, `tags`, `source_id`, `page_number`
    
