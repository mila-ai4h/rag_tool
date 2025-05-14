import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from .indexer import Indexer
from .query_engine import QueryEngine
from .config import TOP_K

app = FastAPI()
indexer = Indexer()

class IndexRequest(BaseModel):
    folder_path: str

class CollectionInfo(BaseModel):
    name: str
    total_documents: int
    source_count: int
    average_chunk_size_chars: float
    average_chunk_size_tokens: float
    total_tokens: int
    total_chars: int

class CollectionListItem(BaseModel):
    name: str

class DeleteResponse(BaseModel):
    status: str
    message: str

class CreateResponse(BaseModel):
    collection: str
    count: int

class AddDocumentResponse(BaseModel):
    status: str
    message: str
    document: str

@app.get("/collections", response_model=List[str])
def list_collections():
    """
    List all available collections.
    Returns an array of collection names.
    """
    return indexer.list_collections()

@app.get("/collections/{name}", response_model=CollectionInfo)
def get_collection_info(name: str):
    """
    Get detailed statistics for a specific collection.
    """
    try:
        stats = indexer.inspect_collection(name)
        return CollectionInfo(
            name=stats["collection_name"],
            total_documents=stats["total_documents"],
            source_count=stats["source_count"],
            average_chunk_size_chars=stats["average_chunk_size_chars"],
            average_chunk_size_tokens=stats["average_chunk_size_tokens"],
            total_tokens=stats["total_tokens"],
            total_chars=stats["total_chars"]
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/collections/{name}", response_model=DeleteResponse)
def delete_collection(name: str):
    """
    Delete a collection and all its associated files.
    """
    try:
        result = indexer.delete_collection(name)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections/{name}", response_model=CreateResponse)
def create_collection(name: str):
    """
    Create a new empty collection.
    """
    # Check if collection already exists
    if name in indexer.list_collections():
        raise HTTPException(status_code=400, detail=f"Collection '{name}' already exists")
    
    try:
        result = indexer.create_collection(name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections/{name}/index", response_model=CreateResponse)
def index_collection(name: str, req: IndexRequest):
    """
    Create a collection and index all PDFs from a folder.
    """
    # Check if collection already exists
    if name in indexer.list_collections():
        raise HTTPException(status_code=400, detail=f"Collection '{name}' already exists")
        
    if not os.path.isdir(req.folder_path):
        raise HTTPException(400, detail="folder_path not found")
    try:
        res = indexer.create_collection(name, req.folder_path)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections/{name}/documents", response_model=AddDocumentResponse)
async def add_document(name: str, file: UploadFile = File(...)):
    """
    Add a single document to an existing collection.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, detail="Only PDF files are supported")
    
    try:
        # Save the uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Add the document to the collection
        result = indexer.add_document(name, temp_path)
        
        # Clean up the temporary file
        os.remove(temp_path)
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections/{name}/query")
def query_collection(
    name: str, 
    q: str, 
    answer: bool = False, 
    top_k: Optional[int] = Query(
        default=None,
        description=f"Number of top documents to retrieve (default: {TOP_K} from config)"
    )
):
    """
    Query a collection with optional LLM answer generation.
    
    Args:
        name: Name of the collection to query
        q: The search query
        answer: Whether to generate an LLM answer (default: False)
        top_k: Number of top documents to retrieve (default: TOP_K from config)
    """
    try:
        qe = QueryEngine(name)
    except Exception:
        raise HTTPException(404, detail="collection not found")
    
    if answer:
        # Get LLM answer which includes the relevant snippets
        answer_result = qe.answer(q, top_k=top_k) if top_k is not None else qe.answer(q)
        # Extract snippets from the answer result
        snippets = [{"text": node.node.get_text(), "score": node.score} 
                   for node in answer_result.source_nodes]
        return {
            "answer": answer_result.response,
            "snippets": snippets
        }
    else:
        # Get raw snippets only
        docs = qe.raw_search(q, top_k=top_k) if top_k is not None else qe.raw_search(q)
        snippets = [{"text": d.node.get_text(), "score": d.score} for d in docs]
        return {"snippets": snippets}
