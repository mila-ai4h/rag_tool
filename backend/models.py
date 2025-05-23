from typing import Optional, List, Dict, Any
from pydantic import BaseModel

# Collection Management Models


class CollectionInfo(BaseModel):
    name: str
    vector_size: int
    distance: str
    points_count: int


class CollectionList(BaseModel):
    collections: List[CollectionInfo]
    total: int


class CollectionCreated(BaseModel):
    collection_name: str
    message: str = "Collection created successfully"


class CollectionDeleted(BaseModel):
    collection_name: str
    message: str = "Collection deleted successfully"


class CollectionExists(BaseModel):
    collection_name: str
    message: str = "Collection already exists"


class CollectionNotFound(BaseModel):
    """Model for when a collection does not exist."""
    collection_name: str
    message: str = "Collection not found"

# Document and Source Management Models


class SourceInfo(BaseModel):
    source_id: str
    filename: str
    type: str = "pdf"
    first_page: int
    last_page: int
    chunks_count: int
    tags: List[str]
    extras: Optional[Dict[str, Any]] = None
    uploaded_at: str  # ISO format timestamp of when the document was uploaded
    # Optional key-value pairs for additional metadata


class SourceList(BaseModel):
    collection_name: str
    sources: List[SourceInfo]
    total: int
    message: str = "Sources listed successfully"


class DocumentIndexed(BaseModel):
    collection_name: str
    source_id: str
    filename: str  # Original filename of the uploaded document
    type: str = "pdf"
    pages_indexed: int
    chunks_created: int
    tags: List[str]
    # Optional key-value pairs for additional metadata
    extras: Optional[Dict[str, Any]] = None
    uploaded_at: str  # ISO format timestamp of when the document was uploaded
    message: str = "Document indexed successfully"


class SourceDeleted(BaseModel):
    collection_name: str
    source_id: str
    points_deleted: int
    message: str = "Source content deleted successfully"


class SourceChunk(BaseModel):
    """Model for a single source chunk, without similarity score."""
    chunk_id: str
    text: str
    page_number: int


class SourceChunksResponse(BaseModel):
    """Model for retrieving all chunks from a source."""
    chunks: List[SourceChunk]
    total: int
    source_id: str
    filename: str
    total_pages: int
    type: str
    tags: List[str]
    # Optional key-value pairs for additional metadata
    extras: Optional[Dict[str, Any]] = None
    uploaded_at: str  # ISO format timestamp of when the document was uploaded


# Query and Response Models


class QueryResult(BaseModel):
    """Model for a single query result."""
    chunk_id: str
    text: str
    source_id: str
    filename: str
    type: str
    page_number: int
    tags: List[str]
    # Optional key-value pairs for additional metadata
    extras: Optional[Dict[str, Any]] = None
    uploaded_at: str  # ISO format timestamp of when the document was uploaded
    similarity_score: float  # Score between 0 and 1, where 1 is most similar


class QueryResponse(BaseModel):
    """Model for the complete query response."""
    results: List[QueryResult]
    total: int


class AnswerResponse(BaseModel):
    """Model for the answer response."""
    answer: str
    chunks: List[QueryResult]
    total_chunks: int

# Error Models


class CollectionError(BaseModel):
    collection_name: str
    error: str


class DocumentError(BaseModel):
    collection_name: str
    error: str


class SourceError(BaseModel):
    collection_name: str
    source_id: str
    error: str


class SourceListError(BaseModel):
    collection_name: str
    error: str


class DocumentEmptyError(BaseModel):
    """Model for when a PDF document has no content."""
    collection_name: str
    filename: str
    message: str = "No text content found in PDF file"
