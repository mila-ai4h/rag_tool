from fastapi import FastAPI, Security, HTTPException, Depends, UploadFile, File, Form, Query
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
import tempfile
import os
import logging
import json
from typing import Optional, List, Dict, Any

from .config import (
    API_KEY,
    API_KEY_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    EMBED_MODEL,
    EMBED_DIMENSIONS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEFAULT_TOP_K,
)
from .indexer import (
    Indexer,
    CollectionCreated,
    CollectionExists,
    CollectionError,
    CollectionList,
    CollectionDeleted,
    DocumentIndexed,
    DocumentError,
    SourceDeleted,
    SourceError,
    SourceList,
    SourceListError,
)
from .query_engine import QueryEngine, QueryResponse, SourceChunksResponse, AnswerResponse
from qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Security dependency
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Unauthorized")

# Initialize Qdrant client, Indexer and QueryEngine
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
indexer = Indexer(
    client,
    embed_model=EMBED_MODEL,
    embed_dimensions=EMBED_DIMENSIONS,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
query_engine = QueryEngine(client)

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/collections", response_model=CollectionList, dependencies=[Depends(verify_api_key)])
def list_collections():
    return indexer.list_collections()


@app.post(
    "/collections/{name}",
    response_model=CollectionCreated,
    dependencies=[Depends(verify_api_key)],
)
def create_collection(name: str):
    result = indexer.create_collection(name)
    if isinstance(result, CollectionExists):
        raise HTTPException(status_code=400, detail=f"Collection '{name}' already exists")
    if isinstance(result, CollectionError):
        raise HTTPException(status_code=500, detail=result.error)
    return result

@app.delete(
    "/collections/{name}",
    response_model=CollectionDeleted,
    dependencies=[Depends(verify_api_key)],
)
def delete_collection(name: str):
    result = indexer.delete_collection(name)
    if isinstance(result, CollectionError):
        raise HTTPException(status_code=500, detail=result.error)
    return result

def validate_extras(extras_dict: Dict[str, Any]) -> None:
    """Validate that extras only contains simple key-value pairs (strings, numbers, booleans)."""
    if not isinstance(extras_dict, dict):
        raise ValueError("extras must be a JSON object")
    
    for key, value in extras_dict.items():
        # Check key is a string
        if not isinstance(key, str):
            raise ValueError(f"extras keys must be strings, got {type(key).__name__}")
        
        # Check value is a simple type
        if not isinstance(value, (str, int, float, bool)) or value is None:
            raise ValueError(
                f"extras values must be strings, numbers, booleans, or null, "
                f"got {type(value).__name__} for key '{key}'"
            )

@app.post(
    "/collections/{name}/add-pdf",
    response_model=DocumentIndexed,
    dependencies=[Depends(verify_api_key)],
)
def add_pdf(
    name: str,
    file: UploadFile = File(..., description="The PDF file to upload"),
    source_id: Optional[str] = Form(None, description="Optional custom source ID. If not provided, a UUID will be generated"),
    tags: Optional[str] = Form(
        None, 
        description="""Comma-separated list of tags to associate with the document.
        Example: "engineering,2024,project-x"
        Spaces around commas are automatically trimmed."""
    ),
    extras: Optional[str] = Form(
        None,
        description="""Optional JSON string containing additional metadata as key-value pairs.
        Must be a valid JSON object with simple key-value pairs only.
        Keys must be strings, values must be strings, numbers, booleans, or null.
        Examples:
        - Simple: {"pdf_link": "https://site.com/file.pdf"}
        - Multiple fields: {"author": "John Doe", "version": 1.0, "is_draft": false}
        
        Note: 
        - Use double quotes for strings, not single quotes
        - No nested objects or arrays allowed
        - No complex types allowed"""
    ),
):
    """
    Upload and index a PDF document into the specified collection.
    
    The document will be split into chunks, embedded, and stored in the vector database.
    Each chunk will inherit the document's metadata (tags, extras, etc.).
    
    Args:
        name: Name of the collection to add the document to
        file: The PDF file to upload
        source_id: Optional custom source ID. If not provided, a UUID will be generated
        tags: Optional comma-separated list of tags. Example: "engineering,2024,project-x"
        extras: Optional JSON string with additional metadata. Must be a valid JSON object
               with simple key-value pairs only (strings, numbers, booleans, or null).
               Example: {"pdf_link": "https://site.com/file.pdf", "version": 1.0}
    
    Returns:
        DocumentIndexed: Information about the indexed document including:
            - collection_name: Name of the collection
            - source_id: ID of the document
            - pages_indexed: Number of pages processed
            - chunks_created: Number of chunks created
            - tags: List of tags applied
            - uploaded_at: Timestamp of upload
            - extras: Additional metadata provided
            - message: Success message
    
    Raises:
        HTTPException: 
            - 400: If the file is not a PDF
            - 400: If extras is not a valid JSON string
            - 400: If extras contains complex types (nested objects, arrays)
            - 500: If there are any processing errors
    """
    # Verify file is PDF
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Parse tags
    tag_list = [tag.strip() for tag in tags.split(',')] if tags else []

    # Parse extras if provided
    extras_dict = None
    if extras:
        try:
            extras_dict = json.loads(extras)
            validate_extras(extras_dict)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400, 
                detail="extras must be a valid JSON string"
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400, 
                detail=str(e)
            )

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        content = file.file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        # Index the PDF
        result = indexer.index_pdf(name, temp_path, file.filename, source_id, tag_list, extras_dict)
        
        if isinstance(result, DocumentError):
            raise HTTPException(status_code=500, detail=result.error)
        
        return result
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass  # Ignore cleanup errors

@app.get(
    "/collections/{collection_name}/sources",
    response_model=SourceList,
    dependencies=[Depends(verify_api_key)],
)
def list_sources(collection_name: str):
    """List all sources in a collection with their details."""
    result = indexer.list_sources(collection_name)
    if isinstance(result, SourceListError):
        raise HTTPException(status_code=500, detail=result.error)
    return result

@app.delete(
    "/collections/{collection_name}/sources/{source_id}",
    response_model=SourceDeleted,
    dependencies=[Depends(verify_api_key)],
)
def delete_source(collection_name: str, source_id: str):
    """Delete all content associated with a given source_id from the collection."""
    result = indexer.delete_by_source_id(collection_name, source_id)
    if isinstance(result, SourceError):
        raise HTTPException(status_code=500, detail=result.error)
    return result

@app.get(
    "/collections/{collection_name}/sources/{source_id}/chunks",
    response_model=SourceChunksResponse,
    dependencies=[Depends(verify_api_key)],
)
def get_source_chunks(
    collection_name: str,
    source_id: str,
    page_number: Optional[int] = Query(None, description="Optional page number to filter chunks by"),
):
    """
    Retrieve all chunks for a specific source_id in a collection.
    
    Args:
        collection_name: Name of the collection to search in
        source_id: The source_id to retrieve chunks for
        page_number: Optional page number to filter chunks by
        
    Returns:
        SourceChunksResponse containing all chunks for the source
    """
    try:
        # Verify collection exists
        existing = {c.name for c in client.get_collections().collections}
        if collection_name not in existing:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' does not exist"
            )
        
        return query_engine.get_source_chunks(
            collection_name=collection_name,
            source_id=source_id,
            page_number=page_number
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/collections/{collection_name}/query",
    response_model=QueryResponse,
    dependencies=[Depends(verify_api_key)],
)
def query_collection(
    collection_name: str,
    q: str = Query(..., description="The search query text"),
    top_k: int = Query(DEFAULT_TOP_K, description="Number of results to return"),
    tags: Optional[str] = Query(None, description="Comma-separated list of tags to filter by (AND operation)"),
    source_id: Optional[str] = Query(None, description="Filter results by source_id"),
    page_number: Optional[int] = Query(None, description="Filter results by page number"),
):
    """
    Query a collection for similar chunks.
    
    Args:
        collection_name: Name of the collection to search in
        q: The search query text
        top_k: Number of results to return (default from config)
        tags: Optional comma-separated list of tags to filter by (AND operation)
        source_id: Optional source_id to filter by
        page_number: Optional page number to filter by
        
    Returns:
        QueryResponse containing the matching results
    """
    try:
        # Verify collection exists
        existing = {c.name for c in client.get_collections().collections}
        if collection_name not in existing:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' does not exist"
            )
        
        # Parse tags from comma-separated string
        tag_list = [tag.strip() for tag in tags.split(',')] if tags else None
        
        return query_engine.query(
            collection_name=collection_name,
            query_text=q,
            top_k=top_k,
            tags=tag_list,
            source_id=source_id,
            page_number=page_number
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/collections/{collection_name}/answer",
    response_model=AnswerResponse,
    dependencies=[Depends(verify_api_key)],
)
def answer_question(
    collection_name: str,
    q: str = Query(..., description="The question to answer"),
    top_k: int = Query(DEFAULT_TOP_K, description="Number of chunks to use for answering"),
    tags: Optional[str] = Query(None, description="Comma-separated list of tags to filter by (AND operation)"),
    source_id: Optional[str] = Query(None, description="Filter chunks by source_id"),
    page_number: Optional[int] = Query(None, description="Filter chunks by page number"),
):
    """
    Generate an answer to a question using relevant chunks from the collection.
    
    Args:
        collection_name: Name of the collection to search in
        q: The question to answer
        top_k: Number of chunks to use (default from config)
        tags: Optional comma-separated list of tags to filter by (AND operation)
        source_id: Optional source_id to filter by
        page_number: Optional page number to filter by
        
    Returns:
        AnswerResponse containing the generated answer and chunks used
    """
    try:
        # Verify collection exists
        existing = {c.name for c in client.get_collections().collections}
        if collection_name not in existing:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' does not exist"
            )
        
        # Parse tags from comma-separated string
        tag_list = [tag.strip() for tag in tags.split(',')] if tags else None
        
        return query_engine.answer(
            collection_name=collection_name,
            query_text=q,
            top_k=top_k,
            tags=tag_list,
            source_id=source_id,
            page_number=page_number
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

