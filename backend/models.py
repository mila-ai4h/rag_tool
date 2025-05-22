from typing import Optional, List
from pydantic import BaseModel

# Collection models
class CollectionCreated(BaseModel):
    collection_name: str
    message: str = "Collection created successfully"

class CollectionDeleted(BaseModel):
    collection_name: str
    message: str = "Collection deleted successfully"

class CollectionExists(BaseModel):
    collection_name: str
    message: str = "Collection already exists"

class CollectionError(BaseModel):
    collection_name: str
    error: str

class CollectionInfo(BaseModel):
    name: str
    vector_size: int
    distance: str
    points_count: int

class CollectionList(BaseModel):
    collections: List[CollectionInfo]
    total: int

class DocumentIndexed(BaseModel):
    collection_name: str
    source_id: str
    pages_indexed: int
    chunks_created: int
    tags: List[str]
    type: str = "pdf"
    message: str = "Document indexed successfully"

class DocumentError(BaseModel):
    collection_name: str
    error: str

class SourceDeleted(BaseModel):
    collection_name: str
    source_id: str
    points_deleted: int
    message: str = "Source content deleted successfully"

class SourceError(BaseModel):
    collection_name: str
    source_id: str
    error: str

class SourceInfo(BaseModel):
    source_id: str
    filename: str
    chunks_count: int
    first_page: int
    last_page: int
    tags: List[str]
    type: str = "pdf"

class SourceList(BaseModel):
    collection_name: str
    sources: List[SourceInfo]
    total: int
    message: str = "Sources listed successfully"

class SourceListError(BaseModel):
    collection_name: str
    error: str
