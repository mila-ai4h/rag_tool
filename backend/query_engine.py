from typing import List, Optional, Union
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, FilterSelector
import logging
import json
from openai import OpenAI

from .models import (
    QueryResult,
    SourceChunk,
    QueryResponse,
    SourceChunksResponse,
    AnswerResponse,
    CollectionNotFound
)

# Configure logging
logger = logging.getLogger(__name__)


class QueryEngine:
    def __init__(
        self,
        client,
        embed_model: str = "text-embedding-3-small",
        embed_dimensions: int = 1536,
        default_top_k: int = 5,
        llm_model: str = "gpt-4"
    ):
        """Initialize the query engine with a Qdrant client and configuration parameters.

        Args:
            client: Qdrant client instance
            embed_model: Name of the OpenAI embedding model to use
            embed_dimensions: Dimension of the embedding vectors
            default_top_k: Default number of results to return
            llm_model: Name of the OpenAI LLM model to use for answers
        """
        self.client = client
        self.embed_model = OpenAIEmbedding(model=embed_model)
        self.llm_client = OpenAI()
        self.embed_dimensions = embed_dimensions
        self.default_top_k = default_top_k
        self.llm_model = llm_model
        logger.info(
            "Initialized QueryEngine with embed_model=%s, embed_dimensions=%d, default_top_k=%d, llm_model=%s",
            embed_model,
            embed_dimensions,
            default_top_k,
            llm_model)

    def _extract_text_from_node(self, node_json: str) -> tuple[str, str]:
        """Extract the actual text content and node ID from a LlamaIndex TextNode JSON string."""
        try:
            node_data = json.loads(node_json)
            # Extract both the text content and node ID
            text = node_data.get("text", "")
            # LlamaIndex uses "id_" for the node ID
            chunk_id = node_data.get("id_", "")
            # Clean up any escaped newlines and other escape sequences
            text = text.encode().decode('unicode_escape')
            # Remove any leading/trailing whitespace and normalize newlines
            text = text.strip().replace('\r\n', '\n')
            return text, chunk_id
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
        top_k: int = None,
        tags: Optional[List[str]] = None,
        source_id: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> Union[QueryResponse, CollectionNotFound]:
        """
        Query the collection for similar chunks.

        Args:
            collection_name: Name of the collection to search in
            query_text: The text to search for
            top_k: Number of results to return (defaults to instance default_top_k)
            tags: Optional list of tags to filter by (AND operation)
            source_id: Optional source_id to filter by
            page_number: Optional page number to filter by

        Returns:
            QueryResponse containing the matching results
            CollectionNotFound if the collection does not exist
        """
        # Use instance default_top_k if not specified
        top_k = top_k or self.default_top_k

        logger.info(
            "Querying collection=%s with top_k=%d, tags=%s, source_id=%s, page_number=%s",
            collection_name,
            top_k,
            tags,
            source_id,
            page_number)

        try:
            # Verify collection exists
            existing = {
                c.name for c in self.client.get_collections().collections}
            if collection_name not in existing:
                logger.error("Collection '%s' does not exist", collection_name)
                return CollectionNotFound(collection_name=collection_name)

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
                text, chunk_id = self._extract_text_from_node(node_content)

                results.append(QueryResult(
                    chunk_id=chunk_id,
                    text=text,
                    source_id=payload.get("source_id", ""),
                    filename=payload.get("filename"),
                    url=payload.get("url"),
                    type=payload.get("type", ""),
                    page_number=payload.get("page_number", 0),
                    tags=payload.get("tags", []),
                    extras=payload.get("extras", None),
                    uploaded_at=payload.get("uploaded_at", ""),
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
    ) -> Union[SourceChunksResponse, CollectionNotFound]:
        """
        Retrieve all chunks for a specific source_id.

        Args:
            collection_name: Name of the collection to search in
            source_id: The source_id to retrieve chunks for
            page_number: Optional page number to filter by

        Returns:
            SourceChunksResponse containing all chunks for the source
            CollectionNotFound if the collection does not exist
        """
        logger.info(
            "Retrieving chunks for collection=%s, source_id=%s, page_number=%s",
            collection_name,
            source_id,
            page_number)

        try:
            # Verify collection exists
            existing = {
                c.name for c in self.client.get_collections().collections}
            if collection_name not in existing:
                logger.error("Collection '%s' does not exist", collection_name)
                return CollectionNotFound(collection_name=collection_name)

            # Build filter for source_id and optional page_number
            search_filter = self._build_filter(
                source_id=source_id, page_number=page_number)

            # Use search with a zero vector to get all points matching the filter
            # We use a large limit to get all chunks
            search_result = self.client.search(
                collection_name=collection_name,
                # Use instance embed_dimensions
                query_vector=[0.0] * self.embed_dimensions,
                limit=10000,  # Adjust if needed for larger collections
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False
            )

            # Initialize metadata variables
            chunks = []
            filename = None
            url = None
            type = "unknown"  # Default type
            tags = []
            extras = None
            uploaded_at = ""
            pages = set()

            # If we have results, get metadata from first chunk
            if search_result:
                first_point = search_result[0]
                if first_point.payload:
                    filename = first_point.payload.get("filename")
                    url = first_point.payload.get("url")
                    type = first_point.payload.get("type", "unknown")
                    tags = first_point.payload.get("tags", [])
                    extras = first_point.payload.get("extras")
                    uploaded_at = first_point.payload.get("uploaded_at", "")

            # Process all chunks
            for scored_point in search_result:
                payload = scored_point.payload
                if not payload:
                    continue

                node_content = payload.get("_node_content", "")
                text, chunk_id = self._extract_text_from_node(node_content)

                # Get page number, defaulting to 1 for URLs
                page_num = payload.get("page_number")
                if type == "url" and page_num is None:
                    page_num = 1
                elif page_num is None:
                    page_num = 0
                pages.add(page_num)

                chunks.append(SourceChunk(
                    chunk_id=chunk_id,
                    text=text,
                    page_number=page_num
                ))

            # Sort chunks by page number and then by chunk_id for stable
            # ordering
            chunks.sort(key=lambda x: (x.page_number, x.chunk_id))

            # For URLs, ensure we have at least one page
            if type == "url" and not pages:
                pages = {1}

            logger.info(
                "Found %d chunks for source_id=%s across %d pages",
                len(chunks), source_id, len(pages)
            )

            return SourceChunksResponse(
                chunks=chunks,
                total=len(chunks),
                source_id=source_id,
                filename=filename,
                url=url,
                total_pages=len(pages),
                type=type,
                tags=tags,
                extras=extras,
                uploaded_at=uploaded_at
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
        context_parts = []
        for chunk in chunks:
            source_info = f"Chunk from "
            if chunk.type == "pdf":
                source_info += f"page {chunk.page_number} of {chunk.filename}"
            else:  # url
                source_info += f"{chunk.url}"
            context_parts.append(f"{source_info}:\n{chunk.text}")

        context = "\n\n".join(context_parts)

        # Create the prompt
        prompt = f"""We have provided context information below.
---------------------
{context}
---------------------
Given this information, please answer the question: {query}
"""

        try:
            # Call LLM using instance llm_model
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
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
        top_k: int = None,
        tags: Optional[List[str]] = None,
        source_id: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> Union[AnswerResponse, CollectionNotFound]:
        """
        Generate an answer based on relevant chunks from the collection.

        Args:
            collection_name: Name of the collection to search in
            query_text: The question to answer
            top_k: Number of chunks to retrieve (defaults to instance default_top_k)
            tags: Optional list of tags to filter by (AND operation)
            source_id: Optional source_id to filter by
            page_number: Optional page number to filter by

        Returns:
            AnswerResponse containing the generated answer and used chunks
            CollectionNotFound if the collection does not exist
        """
        # Use instance default_top_k if not specified
        top_k = top_k or self.default_top_k

        logger.info(
            "Generating answer for collection=%s with query=%s, top_k=%d, tags=%s, source_id=%s, page_number=%s",
            collection_name,
            query_text,
            top_k,
            tags,
            source_id,
            page_number)

        try:
            # Verify collection exists
            existing = {
                c.name for c in self.client.get_collections().collections}
            if collection_name not in existing:
                logger.error("Collection '%s' does not exist", collection_name)
                return CollectionNotFound(collection_name=collection_name)

            # First, get relevant chunks using the existing query method
            query_response = self.query(
                collection_name=collection_name,
                query_text=query_text,
                top_k=top_k,
                tags=tags,
                source_id=source_id,
                page_number=page_number
            )

            # If query returned CollectionNotFound, propagate it
            if isinstance(query_response, CollectionNotFound):
                return query_response

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
