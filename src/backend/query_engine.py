import json
import logging
from typing import List, Optional, Union

from config import OPENAI_API_KEY
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from models import (
    AnswerResponse,
    CollectionNotFound,
    QueryResponse,
    QueryResult,
    SourceChunk,
    SourceChunksResponse,
)
from openai import OpenAI

# Configure logging
logger = logging.getLogger(__name__)


class QueryEngine:
    def __init__(
        self,
        indexer,
        embed_model: str = "text-embedding-3-small",
        embed_dimensions: int = 1536,
        default_top_k: int = 5,
        llm_model: str = "gpt-4",
    ):
        """Initialize the query engine with an indexer and configuration parameters.

        Args:
            indexer: Indexer instance that manages memory stores
            embed_model: Name of the OpenAI embedding model to use
            embed_dimensions: Dimension of the embedding vectors
            default_top_k: Default number of results to return
            llm_model: Name of the OpenAI LLM model to use for answers
        """
        self.indexer = indexer
        self.embed_model = OpenAIEmbedding(model=embed_model)
        self.llm_client = OpenAI(api_key=OPENAI_API_KEY)
        self.embed_dimensions = embed_dimensions
        self.default_top_k = default_top_k
        self.llm_model = llm_model
        logger.info(
            "Initialized QueryEngine with embed_model=%s, embed_dimensions=%d, default_top_k=%d, llm_model=%s",
            embed_model,
            embed_dimensions,
            default_top_k,
            llm_model,
        )

    def _extract_text_from_node(self, text: str, node_id: str) -> tuple[str, str]:
        """Extract the actual text content and node ID."""
        try:
            # Clean up any escaped newlines and other escape sequences
            text = text.encode().decode("unicode_escape")
            # Remove any leading/trailing whitespace and normalize newlines
            text = text.strip().replace("\r\n", "\n")
            return text, node_id
        except Exception as e:
            logger.error("Error extracting text from node: %s", str(e))
            return "", ""

    def _build_metadata_filter(
        self,
        tags: Optional[List[str]] = None,
        source_id: Optional[str] = None,
        page_number: Optional[int] = None,
    ) -> callable:
        """Build a metadata filter function for tags, source_id, and page_number."""

        def filter_func(node_id: str, text: str, metadata: dict):
            # Check source_id
            if source_id and metadata.get("source_id") != source_id:
                return False

            # Check page_number
            if page_number is not None and metadata.get("page_number") != page_number:
                return False

            # Check tags (all tags must be present)
            if tags:
                node_tags = metadata.get("tags", [])
                if not all(tag in node_tags for tag in tags):
                    return False

            return True

        return filter_func

    def query(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = None,
        tags: Optional[List[str]] = None,
        source_id: Optional[str] = None,
        page_number: Optional[int] = None,
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
            page_number,
        )

        try:
            # Get memory store for this collection
            memory_store = self.indexer.get_memory_store(collection_name)

            if not memory_store:
                logger.error("Collection '%s' does not exist", collection_name)
                return CollectionNotFound(collection_name=collection_name)

            # Build metadata filter if needed
            metadata_filter = None
            if tags or source_id is not None or page_number is not None:
                metadata_filter = self._build_metadata_filter(
                    tags, source_id, page_number
                )

            # Get all documents and filter them
            all_documents = memory_store.get_all_documents()
            filtered_documents = []

            for node_id, text, metadata in all_documents:
                if metadata_filter is None or metadata_filter(node_id, text, metadata):
                    filtered_documents.append((node_id, text, metadata))

            # For now, we'll return the first top_k documents
            # In a real implementation, you would use embeddings for similarity search
            results = []
            for i, (node_id, text, metadata) in enumerate(filtered_documents[:top_k]):
                text, chunk_id = self._extract_text_from_node(text, node_id)

                results.append(
                    QueryResult(
                        chunk_id=chunk_id,
                        text=text,
                        source_id=metadata.get("source_id", ""),
                        filename=metadata.get("filename"),
                        url=metadata.get("url"),
                        type=metadata.get("type", ""),
                        page_number=metadata.get("page_number", 0),
                        tags=metadata.get("tags", []),
                        extras=metadata.get("extras", None),
                        uploaded_at=metadata.get("uploaded_at", ""),
                        similarity_score=1.0 - (i * 0.1),  # Simple scoring for now
                    )
                )

            logger.info("Found %d results for query", len(results))
            return QueryResponse(results=results, total=len(results))

        except Exception as e:
            logger.exception("Error during query: %s", str(e))
            raise

    def get_source_chunks(
        self, collection_name: str, source_id: str, page_number: Optional[int] = None
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
            page_number,
        )

        try:
            # Get memory store for this collection
            memory_store = self.indexer.get_memory_store(collection_name)

            if not memory_store:
                logger.error("Collection '%s' does not exist", collection_name)
                return CollectionNotFound(collection_name=collection_name)

            # Initialize metadata variables
            chunks = []
            filename = None
            url = None
            type = "unknown"  # Default type
            tags = []
            extras = None
            uploaded_at = ""
            pages = set()

            # Iterate through all documents in the memory store
            for node_id, text, metadata in memory_store.get_all_documents():
                # Check if this node belongs to the requested source_id
                if metadata.get("source_id") != source_id:
                    continue

                # Check page_number filter if specified
                if (
                    page_number is not None
                    and metadata.get("page_number") != page_number
                ):
                    continue

                # Extract text and chunk_id
                text, chunk_id = self._extract_text_from_node(text, node_id)

                # Get metadata from first matching node
                if not chunks:  # First node
                    filename = metadata.get("filename")
                    url = metadata.get("url")
                    type = metadata.get("type", "unknown")
                    tags = metadata.get("tags", [])
                    extras = metadata.get("extras")
                    uploaded_at = metadata.get("uploaded_at", "")

                # Get page number, defaulting to 1 for URLs
                page_num = metadata.get("page_number")
                if type == "url" and page_num is None:
                    page_num = 1
                elif page_num is None:
                    page_num = 0
                pages.add(page_num)

                chunks.append(
                    SourceChunk(chunk_id=chunk_id, text=text, page_number=page_num)
                )

            # Sort chunks by page number and then by chunk_id for stable ordering
            chunks.sort(key=lambda x: (x.page_number, x.chunk_id))

            # For URLs, ensure we have at least one page
            if type == "url" and not pages:
                pages = {1}

            logger.info(
                "Found %d chunks for source_id=%s across %d pages",
                len(chunks),
                source_id,
                len(pages),
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
                uploaded_at=uploaded_at,
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
            source_info = "Chunk from "
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
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Low temperature for more focused answers
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
        page_number: Optional[int] = None,
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
            page_number,
        )

        try:
            # Get memory store for this collection
            memory_store = self.indexer.get_memory_store(collection_name)

            if not memory_store:
                logger.error("Collection '%s' does not exist", collection_name)
                return CollectionNotFound(collection_name=collection_name)

            # Build metadata filter if needed
            metadata_filter = None
            if tags or source_id is not None or page_number is not None:
                metadata_filter = self._build_metadata_filter(
                    tags, source_id, page_number
                )

            # Get all documents and filter them
            all_documents = memory_store.get_all_documents()
            filtered_documents = []

            for node_id, text, metadata in all_documents:
                if metadata_filter is None or metadata_filter(node_id, text, metadata):
                    filtered_documents.append((node_id, text, metadata))

            # For now, we'll return the first top_k documents
            # In a real implementation, you would use embeddings for similarity search
            results = []
            for i, (node_id, text, metadata) in enumerate(filtered_documents[:top_k]):
                text, chunk_id = self._extract_text_from_node(text, node_id)

                results.append(
                    QueryResult(
                        chunk_id=chunk_id,
                        text=text,
                        source_id=metadata.get("source_id", ""),
                        filename=metadata.get("filename"),
                        url=metadata.get("url"),
                        type=metadata.get("type", ""),
                        page_number=metadata.get("page_number", 0),
                        tags=metadata.get("tags", []),
                        extras=metadata.get("extras", None),
                        uploaded_at=metadata.get("uploaded_at", ""),
                        similarity_score=1.0 - (i * 0.1),  # Simple scoring for now
                    )
                )

            # Generate answer using the retrieved chunks
            answer = self._generate_answer(query_text, results)

            return AnswerResponse(
                answer=answer,
                chunks=results,
                total_chunks=len(results),
            )

        except Exception as e:
            logger.exception("Error generating answer: %s", str(e))
            raise
