# RAG Tool

A modular Retrieval-Augmented Generation (RAG) library built on LlamaIndex, exposed via a FastAPI REST API, and easily integrated into LangGraph agents.

## Features

- **Indexer**: Scan and index PDF documents into named collections.
- **Query Engine**: Perform similarity searches or generate LLM-based answers over indexed collections.
- **REST API**: Endpoints to trigger indexing and query collections.
- **LangGraph Tool**: A wrapper to use the RAG service as a LangGraph "tool."

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running the Server](#running-the-server)
  - [Indexing Documents](#indexing-documents)
  - [Querying Collections](#querying-collections)
  - [Inspecting Collections](#inspecting-collections)
  - [LangGraph Integration](#langgraph-integration)
- [Development](#development)
- [Testing](#testing)
- [Docker](#docker)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- Python 3.10+
- An OpenAI API key (or another LLM provider)
- pip

## Installation

```bash
git clone https://github.com/your-org/rag_tool.git
cd rag_tool
pip install -r requirements.txt
```

## Configuration

Copy the .env template provided in the repo root:

```bash
cp .env.example .env
```

Open .env and fill in your values:

- `OPENAI_API_KEY` – your LLM API key
- (Optional) adjust `EMBEDDING_MODEL`, `LLM_MODEL`, `VECTOR_STORE_PATH`, etc.

Verify config.py reads these values from environment variables.

## Usage

### Running the Server

Run the FastAPI server with Uvicorn:

```bash
uvicorn rag_tool.api:app --reload --host 0.0.0.0 --port 8000
```

Verify by browsing or curling:

```bash
curl http://localhost:8000/docs
```

### Indexing Documents

#### Local Development
When running the server locally, you can index a folder of PDFs into a named collection via CLI or API:

CLI (requires fire):
```bash
cd rag_tool
python -m scripts.index_folder \
    --collection mydocs \
    --folder /path/to/pdfs
```

REST API:
```bash
curl -X POST http://localhost:8000/collections/mydocs/index \
     -H "Content-Type: application/json" \
     -d '{ "folder_path": "/path/to/pdfs" }'
```

#### Cloud Deployment
When running in a Docker container or cloud environment, use the document-by-document indexing approach:

1. Create a new collection:
```bash
curl -X POST http://localhost:8000/collections/mydocs
```

2. Add and index individual documents:
```bash
curl -X POST http://localhost:8000/collections/mydocs/documents \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/document.pdf"
```

The response will confirm the number of documents indexed.

### Querying Collections

Raw similarity search:

```bash
curl "http://localhost:8000/collections/mydocs/query?q=your+keywords"
```

LLM-generated answer:

```bash
curl "http://localhost:8000/collections/mydocs/query?q=What+is+the+main+topic%3F&answer=true"
```

Responses include either text snippets with similarity scores or a generated answer.

### Managing Collections

List all collections with their statistics:

```bash
curl http://localhost:8000/collections
```

Response example:
```json
[
  {
    "name": "test",
    "total_documents": 84,
    "source_count": 3,
    "average_chunk_size": 1442.33
  }
]
```

Delete a collection:

```bash
curl -X DELETE http://localhost:8000/collections/test
```

Response example:
```json
{
  "status": "success",
  "message": "Collection 'test' deleted successfully"
}
```

### Inspecting Collections

You can inspect your collections using the provided CLI tool:

```bash
# List all available collections
python -m scripts.inspect_collection

# Get detailed statistics for a specific collection
python -m scripts.inspect_collection --collection=mydocs
```

The inspection tool provides information about documents, nodes, sources, and chunk sizes.

### LangGraph Integration

Create and register the RAGTool in your agent code:

```python
from rag_tool.langgraph_tool import RAGTool
from langgraph import Agent

agent = Agent(
    tools=[
        RAGTool(),  # uses RAG_API_URL from .env
        # ... other tools
    ],
)
```

Your agent can now call `rag_query(collection="mydocs", query="...")` as a tool.

For a complete example, see `scripts/test_langgraph.py`.

## Development

- Edit code under `rag_tool/`.
- Add new endpoints in `api.py` or new features in `indexer.py` and `query_engine.py`.
- Update `config.py` for additional settings.

## Testing

```bash
pytest tests/
```

## Docker

### Prerequisites
- Docker Desktop installed and running
- Git repository cloned locally

### Building and Running
```bash
# Build the Docker image
docker build -t rag_tool:latest .

# Run with environment variables from .env file
docker run -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data/vectorstores:/app/vectorstores \
  -e VECTOR_STORE_PATH=/app/vectorstores \
  rag_tool:latest

# Or run with specific environment variables
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key_here \
  -e EMBEDDING_MODEL=text-embedding-3-small \
  -e LLM_MODEL=gpt-4o \
  -v $(pwd)/data/vectorstores:/app/vectorstores \
  -e VECTOR_STORE_PATH=/app/vectorstores \
  rag_tool:latest
```

### Development with Docker
For development with hot-reload:
```bash
# Mount the source code directory for live updates and vectorstores
docker run -p 8000:8000 \
  --env-file .env \
  -v $(pwd):/app \
  -e VECTOR_STORE_PATH=/app/data/vectorstores \
  rag_tool:latest \
  uvicorn rag_tool.api:app --host 0.0.0.0 --port 8000 --reload
```

## Contributing

1. Fork this repo
2. Create a feature branch (`git checkout -b feature/...`)
3. Commit your changes
4. Open a Pull Request

Please follow PEP8 and include tests for new functionality.

## License

MIT