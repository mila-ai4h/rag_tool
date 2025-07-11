# RAG Tool

A Retrieval-Augmented Generation (RAG) service that provides semantic search and question-answering capabilities.

## Overview

FastAPI backend service that leverages LlamaIndex for document processing and retrieval, using Qdrant as the vector store and OpenAI's models for embeddings and LLM generation.

## Key Features

- Collection-based document management with metadata and tags support
- Semantic search with similarity scoring and source attribution
- OpenAI integration (text-embedding-3-small for embeddings, gpt-4o for generation)
- PDF document processing with PyMuPDF

## Run Locally

Clone the repository locally, copy the `.env.example` file to `.env` in the root of the project and update the values of the API keys that need to be changed:


```bash
git clone https://github.com/mila-ai4h/rag_tool && cd rag_tool/
cp .env.example .env
```

You will need at minimum to set the `OPENAI_API_KEY` to a valid value. Other keys can stay at their defaults for now.
  OPENAI_API_KEY=<OPENAI_API_KEY>

> Note: You will need to set the `API_KEY` for authentication to the server using the `x-api-key` in the request header. By default, the `API_KEY` is set to "secret-key".


```bash
# Stop and rebuild
docker compose --profile local down
docker compose build
docker compose --profile local up -d
```

The app runs by default on PORT 8080.

You should then be able to navigate to the following page:

    http://localhost:8080/docs

### CLI Usage
You can access the API using REST endpoints from the CLI. The `/health` endpoint does not require an API key and should return a 200:

    $ curl http://localhost:8080/health
    {"status":"ok"}

You can also ping the collections endpoint, but will need to use the API key set in the .env file:

    $ curl http://localhost:8080/collections -H "x-api-key: your-secret-key"
    {"collections":[],"total":0}%

## Technology Stack

- **Backend**: FastAPI
- **Vector Store**: Qdrant
- **Document Processing**: LlamaIndex
- **AI Models**: OpenAI
- **Language**: Python 3.x

## API Documentation

Interactive API documentation is available at `http://localhost:8080/docs` when running locally.

### Authentication

All endpoints except `/health` require an API key to be passed in the `X-API-Key` header.

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

## Debugging

View vector store content:
```bash
curl -X POST http://localhost:6333/collections/test/points/scroll \
  -H "Content-Type: application/json" \
  -d '{"limit": 10, "with_payload": true, "with_vector": false}' | jq .
```

## Releases

- 14 May 2025:
    - FastAPI backend + llamaindex vector store
    - Leverage llamaindex for text parsing, chunking and embedding
    - Use OpenAI embedding models (via llamaindex interface)
    - Use OpenAI LLMs (via llamaindex interface)
    - Use PyMuPDF for extracting text from PDF
    - Define API endpoints to:
        - Create collection
        - Delect collection
        - List collection
        - Add a PDF to a collection (which extracts the text, chunks it, gneerates embeddings and store them in the vector store)
        - Retrieve chunks based a semantic matching and optionally use an LLM to answer a question
    - Langgraph example script

- 23 May 2025:
    - Refactor code base
    - Add health check
    - Integrate qdrant as vector store
    - Use SentenceSplitter text parser
    - Track documents by source_id
    - Support deleting the chunks corresponding to a source_id from vector store
    - Avoid creating duplicate chunks when indexing the same document twice
    - Add optional tags to document. Support filtering results by tags ("AND" combination)
    - Support "extras" metadata (key-value pairs that users can use for their own needs, e.g. which link should I use to download a document)
    - Track chunks by id
    - Require API key to authenticate
    - List source documents whose content has been indexed
    - Return chunks for a given source document
    - Return the following data along with each chunk:
        - Chunk id
        - Chunk text
        - Source id
        - File name
        - Page number
        - Tags
        - Extras
        - Time at which document was uploaded
    - Return similarity scores along with each chunk returned by the /query and /answer endpoints
    - /query and /answer endpoints can filter by tags, source_id, page_number
    - /query and /answer endpoints can modify the top_k chunks returned
    - Remove Langgraph example script

- 26 May 2025:
    - Add support for indexing web content via URLs
    - New endpoint `/collections/{collection_name}/add-url` to process and index content from web pages
    - Add provisioning script and data for CASH project

## Backlog:

- Deploy to cloud
- Add basic integration tests
- Explore better PDF text parsers as PyMuPDF does not fare very well on certain documents
- Requests are currently blocking: create thread pool and use async requests
- Support hybrid search
