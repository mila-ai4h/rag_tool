from typing import List, Optional
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, FilterSelector
from pydantic import BaseModel
import logging
import json
from openai import OpenAI

from .config import DEFAULT_TOP_K, EMBED_MODEL, EMBED_DIMENSIONS, LLM_MODEL

# Configure logging
logger = logging.getLogger(__name__)

class QueryResult(BaseModel):
    """Model for a single query result."""
    node_id: str
    text: str
    source_id: str
    filename: str
    page_number: int
    tags: List[str]
    similarity_score: float  # Score between 0 and 1, where 1 is most similar

class SourceChunk(BaseModel):
    """Model for a single source chunk, without similarity score."""
    node_id: str
    text: str
    source_id: str
    filename: str
    page_number: int
    tags: List[str]

class QueryResponse(BaseModel):
    """Model for the complete query response."""
    results: List[QueryResult]
    total: int

class SourceChunksResponse(BaseModel):
    """Model for retrieving all chunks from a source."""
    chunks: List[SourceChunk]
    total: int
    source_id: str
    filename: str
    total_pages: int

class AnswerResponse(BaseModel):
    """Model for the answer response."""
    answer: str
    chunks: List[QueryResult]
    total_chunks: int

class QueryEngine:
    def __init__(self, client):
        """Initialize the query engine with a Qdrant client."""
        self.client = client
        self.embed_model = OpenAIEmbedding(model=EMBED_MODEL)
        self.llm_client = OpenAI()
        logger.info("Initialized QueryEngine with model=%s", EMBED_MODEL)

    def _extract_text_from_node(self, node_json: str) -> tuple[str, str]:
        """Extract the actual text content and node ID from a LlamaIndex TextNode JSON string."""
        try:
            node_data = json.loads(node_json)
            # Extract both the text content and node ID
            text = node_data.get("text", "")
            node_id = node_data.get("id_", "")  # LlamaIndex uses "id_" for the node ID
            # Clean up any escaped newlines and other escape sequences
            text = text.encode().decode('unicode_escape')
            # Remove any leading/trailing whitespace and normalize newlines
            text = text.strip().replace('\r\n', '\n')
            return text, node_id
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Error parsing node JSON: %s", str(e))
            return "", ""

    def _build_filter(
        self,
        tags: Optional[List[str]] = None,
        source_id: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> Optional[Filter]:
        """Build a Qdrant filter for tags, source_id, and page_number."""
        conditions = []
        
        if tags:
            # For each tag, we want to ensure it exists in the tags array
            for tag in tags:
                conditions.append(
                    FieldCondition(
                        key="tags",
                        match=MatchValue(value=tag)
                    )
                )
        
        if source_id:
            conditions.append(
                FieldCondition(
                    key="source_id",
                    match=MatchValue(value=source_id)
                )
            )
            
        if page_number is not None:
            conditions.append(
                FieldCondition(
                    key="page_number",
                    match=MatchValue(value=page_number)
                )
            )
        
        return Filter(must=conditions) if conditions else None

    def query(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = DEFAULT_TOP_K,
        tags: Optional[List[str]] = None,
        source_id: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> QueryResponse:
        """
        Query the collection for similar chunks.
        
        Args:
            collection_name: Name of the collection to search in
            query_text: The text to search for
            top_k: Number of results to return (default from config)
            tags: Optional list of tags to filter by (AND operation)
            source_id: Optional source_id to filter by
            page_number: Optional page number to filter by
            
        Returns:
            QueryResponse containing the matching results
        """
        logger.info(
            "Querying collection=%s with top_k=%d, tags=%s, source_id=%s, page_number=%s",
            collection_name, top_k, tags, source_id, page_number
        )

        try:
            # Generate query embedding
            query_embedding = self.embed_model.get_text_embedding(query_text)
            
            # Build filter if needed
            search_filter = self._build_filter(tags, source_id, page_number)
            
            # Perform vector search
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False
            )
            
            # Convert results to QueryResult objects
            results = []
            for scored_point in search_result:
                payload = scored_point.payload
                node_content = payload.get("_node_content", "")
                text, node_id = self._extract_text_from_node(node_content)
                
                results.append(QueryResult(
                    node_id=node_id,
                    text=text,
                    source_id=payload.get("source_id", ""),
                    filename=payload.get("filename", ""),
                    page_number=payload.get("page_number", 0),
                    tags=payload.get("tags", []),
                    similarity_score=scored_point.score
                ))
            
            logger.info("Found %d results for query", len(results))
            return QueryResponse(results=results, total=len(results))
            
        except Exception as e:
            logger.exception("Error during query: %s", str(e))
            raise 

    def get_source_chunks(
        self,
        collection_name: str,
        source_id: str,
        page_number: Optional[int] = None
    ) -> SourceChunksResponse:
        """
        Retrieve all chunks for a specific source_id.
        
        Args:
            collection_name: Name of the collection to search in
            source_id: The source_id to retrieve chunks for
            page_number: Optional page number to filter by
            
        Returns:
            SourceChunksResponse containing all chunks for the source
        """
        logger.info(
            "Retrieving chunks for collection=%s, source_id=%s, page_number=%s",
            collection_name, source_id, page_number
        )

        try:
            # Build filter for source_id and optional page_number
            search_filter = self._build_filter(source_id=source_id, page_number=page_number)
            
            # Use search with a zero vector to get all points matching the filter
            # We use a large limit to get all chunks
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=[0.0] * EMBED_DIMENSIONS,  # Zero vector
                limit=10000,  # Adjust if needed for larger collections
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False
            )
            
            # Convert points to SourceChunk objects
            chunks = []
            filename = ""
            pages = set()
            
            for scored_point in search_result:
                payload = scored_point.payload
                node_content = payload.get("_node_content", "")
                text, node_id = self._extract_text_from_node(node_content)
                
                # Store filename from first chunk (should be same for all chunks)
                if not filename:
                    filename = payload.get("filename", "")
                
                # Track page numbers
                if "page_number" in payload:
                    pages.add(payload["page_number"])
                
                chunks.append(SourceChunk(
                    node_id=node_id,
                    text=text,
                    source_id=source_id,
                    filename=filename,
                    page_number=payload.get("page_number", 0),
                    tags=payload.get("tags", [])
                ))
            
            # Sort chunks by page number and then by node_id for stable ordering
            chunks.sort(key=lambda x: (x.page_number, x.node_id))
            
            logger.info(
                "Found %d chunks for source_id=%s across %d pages",
                len(chunks), source_id, len(pages)
            )
            
            return SourceChunksResponse(
                chunks=chunks,
                total=len(chunks),
                source_id=source_id,
                filename=filename,
                total_pages=len(pages)
            )
            
        except Exception as e:
            logger.exception("Error retrieving source chunks: %s", str(e))
            raise 

    def _generate_answer(self, query: str, chunks: List[QueryResult]) -> str:
        """
        Generate an answer using LLM based on the query and retrieved chunks.
        
        Args:
            query: The user's question
            chunks: List of relevant chunks to use for answering
            
        Returns:
            Generated answer from LLM
        """
        # Prepare the context from chunks
        context = "\n\n".join([
            f"Chunk from page {chunk.page_number} of {chunk.filename}:\n{chunk.text}"
            for chunk in chunks
        ])
        
        # Create the prompt
        prompt = f"""We have provided context information below.
---------------------
{context}
---------------------
Given this information, please answer the question: {query}
"""
        
        try:
            # Call LLM
            response = self.llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1  # Low temperature for more focused answers
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.exception("Error generating answer: %s", str(e))
            raise

    def answer(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = DEFAULT_TOP_K,
        tags: Optional[List[str]] = None,
        source_id: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> AnswerResponse:
        """
        Generate an answer based on relevant chunks from the collection.
        
        Args:
            collection_name: Name of the collection to search in
            query_text: The question to answer
            top_k: Number of chunks to retrieve (default from config)
            tags: Optional list of tags to filter by (AND operation)
            source_id: Optional source_id to filter by
            page_number: Optional page number to filter by
            
        Returns:
            AnswerResponse containing the generated answer and used chunks
        """
        logger.info(
            "Generating answer for collection=%s with query=%s, top_k=%d, tags=%s, source_id=%s, page_number=%s",
            collection_name, query_text, top_k, tags, source_id, page_number
        )

        try:
            # First, get relevant chunks using the existing query method
            query_response = self.query(
                collection_name=collection_name,
                query_text=query_text,
                top_k=top_k,
                tags=tags,
                source_id=source_id,
                page_number=page_number
            )
            
            # Generate answer using the retrieved chunks
            answer = self._generate_answer(query_text, query_response.results)
            
            return AnswerResponse(
                answer=answer,
                chunks=query_response.results,
                total_chunks=len(query_response.results)
            )
            
        except Exception as e:
            logger.exception("Error generating answer: %s", str(e))
            raise 