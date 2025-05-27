from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
from llama_index.core import Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
import fitz  # PyMuPDF
import uuid
import logging
from typing import Optional, Tuple, List, Dict, Any, Union
from datetime import datetime
from playwright.sync_api import sync_playwright
import requests
import trafilatura

from models import (
    CollectionCreated,
    CollectionExists,
    CollectionError,
    CollectionInfo,
    CollectionList,
    CollectionDeleted,
    CollectionNotFound,
    DocumentIndexed,
    DocumentError,
    DocumentEmptyError,
    SourceDeleted,
    SourceError,
    SourceInfo,
    SourceList,
    SourceListError,
)

# Configure logging
logger = logging.getLogger(__name__)


class Indexer:
    def __init__(
        self,
        client,
        chunk_size: int,
        chunk_overlap: int,
        embed_model: str,
        embed_dimensions: int,
    ):
        self.client = client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_model = embed_model
        self.embed_dimensions = embed_dimensions

        logger.info(
            "Initializing Indexer with chunk_size=%d, chunk_overlap=%d, embed_model=%s, embed_dimensions=%d",
            chunk_size,
            chunk_overlap,
            embed_model,
            embed_dimensions)

        # Configure LlamaIndex global settings
        Settings.embed_model = OpenAIEmbedding(model=embed_model)
        Settings.include_metadata = True
        Settings.include_embeddings = True
        Settings.disable_relationship_storage = True

    def create_collection(self, name: str):
        logger.info("Attempting to create collection=%s", name)
        try:
            existing = {
                c.name for c in self.client.get_collections().collections}
            if name in existing:
                logger.warning("Collection '%s' already exists", name)
                return CollectionExists(collection_name=name)

            logger.info(
                "Creating collection=%s with vector size=%d",
                name,
                self.embed_dimensions)
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self.embed_dimensions,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Successfully created collection=%s", name)
            return CollectionCreated(collection_name=name)

        except Exception as e:
            logger.exception("Error creating collection=%s: %s", name, str(e))
            return CollectionError(collection_name=name, error=str(e))

    def list_collections(self):
        logger.info("Listing all collections")
        try:
            collections = self.client.get_collections().collections
            logger.info("Found %d collections", len(collections))

            infos = []
            for col in collections:
                logger.debug("Getting details for collection=%s", col.name)
                info = self.client.get_collection(collection_name=col.name)
                count = self.client.count(
                    collection_name=col.name, exact=True).count
                logger.debug("Collection=%s has %d points", col.name, count)

                infos.append(
                    CollectionInfo(
                        name=col.name,
                        vector_size=info.config.params.vectors.size,
                        distance=info.config.params.vectors.distance,
                        points_count=count,
                    )
                )

            logger.info("Successfully listed %d collections", len(infos))
            return CollectionList(collections=infos, total=len(infos))

        except Exception as e:
            logger.exception("Error listing collections: %s", str(e))
            raise  # Re-raise as this is a core operation

    def delete_collection(
            self, name: str) -> Union[CollectionDeleted, CollectionError]:
        """Delete a collection and all its content.

        Returns:
            CollectionDeleted: If the collection was successfully deleted or did not exist
            CollectionError: For other processing errors
        """
        logger.info("Attempting to delete collection=%s", name)
        try:
            existing = {
                c.name for c in self.client.get_collections().collections}
            if name not in existing:
                logger.info(
                    "Collection '%s' does not exist, nothing to delete", name)
                return CollectionDeleted(collection_name=name)

            # Get count before deletion for logging
            count = self.client.count(collection_name=name, exact=True).count
            logger.info(
                "Collection=%s has %d points before deletion",
                name,
                count)

            self.client.delete_collection(collection_name=name)
            logger.info(
                "Successfully deleted collection=%s with %d points",
                name,
                count)
            return CollectionDeleted(collection_name=name)

        except Exception as e:
            logger.exception("Error deleting collection=%s: %s", name, str(e))
            return CollectionError(collection_name=name, error=str(e))

    def _extract_documents_from_pdf(self,
                                    pdf_path: str,
                                    filename: str,
                                    source_id: str,
                                    tags: List[str],
                                    uploaded_at: str,
                                    extras: Optional[Dict[str,
                                                          Any]] = None) -> Tuple[list[Document],
                                                                                 int]:
        """Extract text from PDF and create Document objects with metadata."""
        logger.info(
            "Starting PDF extraction for file=%s, source_id=%s, tags=%s, extras=%s",
            filename,
            source_id,
            tags,
            extras)
        documents = []

        logger.info("Using upload timestamp: %s", uploaded_at)

        with fitz.open(pdf_path) as pdf:
            total_pages = len(pdf)
            logger.info("PDF opened successfully, total pages=%d", total_pages)

            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text()
                if text.strip():  # Skip empty pages
                    documents.append(Document(
                        text=text,
                        metadata={
                            "source_id": source_id,
                            "filename": filename,
                            "url": None,  # PDFs don't have URLs
                            "type": "pdf",
                            "page_number": page_number,
                            "tags": tags,
                            "extras": extras,
                            "uploaded_at": uploaded_at
                        }
                    ))
                    logger.debug(
                        "Extracted text from page %d/%d",
                        page_number,
                        total_pages)
                else:
                    logger.debug(
                        "Skipping empty page %d/%d",
                        page_number,
                        total_pages)

        logger.info(
            "PDF extraction completed, extracted %d non-empty pages",
            len(documents))
        return documents, len(documents)

    def _extract_document_from_url(
        self,
        url: str,
        source_id: str,
        tags: List[str],
        uploaded_at: str,
        extras: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Extract text content from a URL using trafilatura.
        
        Args:
            url: The URL to extract content from
            source_id: Unique identifier for the source
            tags: List of tags to associate with the content
            uploaded_at: ISO format timestamp of when the content was uploaded
            extras: Optional dictionary of additional metadata
            
        Returns:
            Document: A Document object containing the extracted text and metadata
            
        Raises:
            ValueError: If no text content could be extracted from the URL
            requests.RequestException: If the URL could not be fetched
        """
        logger.info("Starting URL extraction: %s", url)

        try:
            # Download and extract content using trafilatura
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                raise ValueError(f"Could not download content from {url}")
                
            # Extract metadata first to get the title
            metadata = trafilatura.metadata.extract_metadata(downloaded)
            title = metadata.title if metadata else None
                
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                favor_precision=True,
                include_tables=True,
                include_images=False,
                include_links=False,
                include_formatting=True
            )
            
            if not text or len(text.strip()) < 100:  # Basic validation
                raise ValueError(f"No substantial text content found in {url}")
                
            # Add title if available
            if title:
                text = f"{title}\n\n{text}"
                
            logger.info("Successfully extracted content from %s (length=%d)", url, len(text))
            
            return Document(
                text=text,
                metadata={
                    "source_id": source_id,
                    "url": url,
                    "type": "url",
                    "tags": tags,
                    "extras": extras,
                    "uploaded_at": uploaded_at
                }
            )
            
        except requests.RequestException as e:
            logger.error("Failed to fetch URL=%s: %s", url, str(e))
            raise
        except Exception as e:
            logger.error("Error extracting content from URL=%s: %s", url, str(e))
            raise ValueError(f"Failed to extract content from {url}: {str(e)}")

    def _delete_source_chunks(
        self,
        collection_name: str,
        source_id: str
    ) -> Union[None, CollectionNotFound, DocumentError]:
        """Delete all chunks for a given source_id from the collection.

        Returns:
            None: If deletion was successful
            CollectionNotFound: If the collection does not exist
            DocumentError: For other processing errors
        """
        try:
            # Verify collection exists
            existing = {
                c.name for c in self.client.get_collections().collections}
            if collection_name not in existing:
                logger.error("Collection '%s' does not exist", collection_name)
                return CollectionNotFound(collection_name=collection_name)

            count_before = self.client.count(
                collection_name=collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_id",
                            match=MatchValue(value=source_id))]),
                exact=True).count

            if count_before > 0:
                logger.info(
                    "Found %d existing chunks with source_id=%s, deleting them first",
                    count_before,
                    source_id)
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="source_id",
                                match=MatchValue(value=source_id))]))
                logger.info(
                    "Successfully deleted existing chunks for source_id=%s",
                    source_id)
            return None

        except Exception as e:
            logger.exception("Error during source deletion: %s", str(e))
            return DocumentError(collection_name=collection_name, error=str(e))

    def _index_document(
        self,
        collection_name: str,
        source_id: str,
        document: Document,
        type: str,
        tags: List[str],
        extras: Optional[Dict[str, Any]] = None
    ) -> Union[DocumentIndexed, DocumentError, CollectionNotFound]:
        """Common indexing logic for both PDF and URL documents.

        Args:
            collection_name: Name of the collection to index into
            source_id: Unique identifier for the source
            document: The Document to index
            type: Type of document ("pdf" or "url")
            tags: List of tags to associate with the content
            extras: Optional dictionary of additional metadata

        Returns:
            DocumentIndexed: If the document was successfully indexed
            CollectionNotFound: If the collection does not exist
            DocumentError: For other processing errors
        """
        logger.info(
            "Starting document indexing process for collection=%s, type=%s, tags=%s, extras=%s",
            collection_name,
            type,
            tags,
            extras)

        try:
            # Get current timestamp in ISO format with UTC timezone indicator
            uploaded_at = datetime.utcnow().isoformat() + "Z"
            logger.info("Using upload timestamp: %s", uploaded_at)

            # Verify collection exists
            existing = {
                c.name for c in self.client.get_collections().collections}
            if collection_name not in existing:
                logger.error("Collection '%s' does not exist", collection_name)
                return CollectionNotFound(collection_name=collection_name)

            # Chunk with SentenceSplitter
            logger.info(
                "Starting document chunking with chunk_size=%d, chunk_overlap=%d",
                self.chunk_size,
                self.chunk_overlap)
            splitter = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            nodes = splitter.get_nodes_from_documents([document])
            logger.info("Created %d chunks from document", len(nodes))

            # Embed and store in Qdrant
            logger.info("Starting embedding generation and vector storage")
            vector_store = QdrantVectorStore(
                client=self.client, collection_name=collection_name)

            # Process nodes in batches for better logging
            batch_size = 10
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                logger.info("Processing embedding batch %d-%d/%d",
                            i + 1, min(i + batch_size, len(nodes)), len(nodes))
                texts = [node.text for node in batch]
                embeddings = Settings.embed_model.get_text_embedding_batch(
                    texts)
                for node, embedding in zip(batch, embeddings):
                    node.embedding = embedding
                vector_store.add(batch)

            # Get source identifier (filename or url) from document metadata
            source_identifier = document.metadata.get(
                "filename") or document.metadata.get("url")
            if not source_identifier:
                raise ValueError(
                    "Document metadata must contain either filename or url")

            logger.info(
                "Successfully completed document indexing for %s",
                source_identifier)
            return DocumentIndexed(
                collection_name=collection_name,
                source_id=source_id,
                filename=document.metadata.get("filename"),
                url=document.metadata.get("url"),
                type=type,
                pages_indexed=1,  # Both PDFs and URLs are treated as single documents for now
                chunks_created=len(nodes),
                tags=tags,
                extras=extras,
                uploaded_at=uploaded_at,
                message="Document indexed successfully"
            )

        except Exception as e:
            logger.exception("Error during document indexing: %s", str(e))
            return DocumentError(collection_name=collection_name, error=str(e))

    def index_pdf(
        self,
        collection_name: str,
        file_path: str,
        filename: str,
        source_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        extras: Optional[Dict[str, Any]] = None
    ) -> Union[DocumentIndexed, DocumentError, DocumentEmptyError, CollectionNotFound]:
        logger.info(
            "Starting PDF indexing process for collection=%s, file=%s, tags=%s, extras=%s",
            collection_name,
            filename,
            tags,
            extras)

        try:
            # Generate or use provided source_id
            source_id = source_id or str(uuid.uuid4())
            logger.info("Using source_id=%s", source_id)

            # Normalize tags
            tags = tags or []

            # Delete any existing chunks for this source_id
            delete_result = self._delete_source_chunks(
                collection_name, source_id)
            if delete_result is not None:
                return delete_result

            # Extract documents from PDF (one per page)
            documents, pages_count = self._extract_documents_from_pdf(
                file_path, filename, source_id, tags, datetime.utcnow().isoformat() + "Z", extras)
            if not documents:
                logger.error("No text content found in PDF file=%s", filename)
                return DocumentEmptyError(
                    collection_name=collection_name,
                    filename=filename,
                    url=None,
                    message="No text content found in PDF file"
                )

            # Process each page document separately to maintain page numbers
            total_chunks = 0
            for doc in documents:
                # Index each page document separately
                result = self._index_document(
                    collection_name=collection_name,
                    source_id=source_id,
                    document=doc,  # Use the individual page document
                    type="pdf",
                    tags=tags,
                    extras=extras
                )

                if isinstance(result, DocumentIndexed):
                    total_chunks += result.chunks_created
                else:
                    # If any page fails, return the error
                    return result

            # Create a success response with the total chunks
            logger.info("Successfully indexed %d pages from PDF file=%s with %d total chunks",
                        pages_count, filename, total_chunks)
            return DocumentIndexed(
                collection_name=collection_name,
                source_id=source_id,
                filename=filename,
                url=None,
                type="pdf",
                pages_indexed=pages_count,
                chunks_created=total_chunks,
                tags=tags,
                extras=extras,
                uploaded_at=documents[0].metadata["uploaded_at"],
                message="Document indexed successfully"
            )

        except Exception as e:
            logger.exception("Error during PDF indexing: %s", str(e))
            return DocumentError(collection_name=collection_name, error=str(e))

    def index_url(
        self,
        collection_name: str,
        url: str,
        source_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        extras: Optional[Dict[str, Any]] = None
    ) -> Union[DocumentIndexed, DocumentError, DocumentEmptyError, CollectionNotFound]:
        """Public method to index a url. This method adapts URL input into the common Document format.

        This is a facade method that:
        1. Handles URL-specific setup and validation
        2. Fetches and converts the url into a Document
        3. Delegates the actual indexing to _index_document

        Args:
            collection_name: Name of the collection to index into
            url: The URL to fetch and index
            source_id: Optional unique identifier for the source
            tags: Optional list of tags to associate with the content
            extras: Optional dictionary of additional metadata

        Returns:
            DocumentIndexed: If the document was successfully indexed
            CollectionNotFound: If the collection does not exist
            DocumentEmptyError: If the url has no text content
            DocumentError: For other processing errors
        """
        logger.info(
            "Starting URL indexing process for collection=%s, url=%s, tags=%s, extras=%s",
            collection_name,
            url,
            tags,
            extras)

        try:
            # Generate or use provided source_id
            source_id = source_id or str(uuid.uuid4())
            logger.info("Using source_id=%s", source_id)

            # Normalize tags
            tags = tags or []

            # Delete any existing chunks for this source_id
            delete_result = self._delete_source_chunks(
                collection_name, source_id)
            if delete_result is not None:
                return delete_result

            # Extract document from URL
            try:
                document = self._extract_document_from_url(
                    url, source_id, tags, datetime.utcnow().isoformat() + "Z", extras)
            except ValueError as e:
                logger.error("No text content found in URL=%s", url)
                return DocumentEmptyError(
                    collection_name=collection_name,
                    filename=None,
                    url=url,
                    message="No text content found in URL"
                )
            except requests.RequestException as e:
                logger.error("Failed to fetch URL=%s: %s", url, str(e))
                return DocumentError(
                    collection_name=collection_name,
                    error=f"Failed to fetch URL: {str(e)}"
                )

            return self._index_document(
                collection_name=collection_name,
                source_id=source_id,
                document=document,
                type="url",
                tags=tags,
                extras=extras
            )

        except Exception as e:
            logger.exception("Error during URL indexing: %s", str(e))
            return DocumentError(collection_name=collection_name, error=str(e))

    def delete_by_source_id(self,
                            collection_name: str,
                            source_id: str) -> Union[SourceDeleted,
                                                     SourceError,
                                                     CollectionNotFound]:
        """Delete all content associated with a given source_id from the collection.

        Returns:
            SourceDeleted: Information about the deleted source
            CollectionNotFound: If the collection does not exist
            SourceError: For other processing errors
        """
        logger.info(
            "Starting deletion of content for collection=%s, source_id=%s",
            collection_name,
            source_id)
        try:
            # Verify collection exists
            existing = {
                c.name for c in self.client.get_collections().collections}
            if collection_name not in existing:
                logger.error("Collection '%s' does not exist", collection_name)
                return CollectionNotFound(collection_name=collection_name)

            # Debug: Get total points in collection
            total_points = self.client.count(
                collection_name=collection_name, exact=True).count
            logger.info("Total points in collection: %d", total_points)

            # Debug: List all unique source_ids in collection
            search_result = self.client.scroll(
                collection_name=collection_name,
                limit=100,  # Adjust if needed
                with_payload=True,
                with_vectors=False
            )[0]  # scroll returns (points, next_page_offset)

            unique_source_ids = {point.payload.get(
                "source_id") for point in search_result if point.payload}
            logger.info(
                "Found source_ids in collection: %s",
                unique_source_ids)

            # Get count before deletion to know how many points were deleted
            count_before = self.client.count(
                collection_name=collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_id",
                            match=MatchValue(
                                value=source_id))]),
                exact=True).count
            logger.info(
                "Found %d points matching source_id=%s",
                count_before,
                source_id)

            if count_before == 0:
                logger.warning(
                    "No points found with source_id=%s in collection=%s",
                    source_id,
                    collection_name)
                return SourceDeleted(
                    collection_name=collection_name,
                    source_id=source_id,
                    points_deleted=0
                )

            # Delete points matching the source_id using proper filter models
            self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source_id",
                            match=MatchValue(
                                value=source_id))]))

            logger.info(
                "Successfully deleted %d points for source_id=%s",
                count_before,
                source_id)

            return SourceDeleted(
                collection_name=collection_name,
                source_id=source_id,
                points_deleted=count_before
            )

        except Exception as e:
            logger.exception("Error during source deletion: %s", str(e))
            return SourceError(
                collection_name=collection_name,
                source_id=source_id,
                error=str(e)
            )

    def list_sources(self,
                     collection_name: str) -> Union[SourceList,
                                                    SourceListError,
                                                    CollectionNotFound]:
        """List all sources in a collection with their details.

        Returns:
            SourceList: List of sources in the collection
            CollectionNotFound: If the collection does not exist
            SourceListError: For other processing errors
        """
        logger.info("Listing sources for collection=%s", collection_name)
        try:
            # Verify collection exists
            existing = {
                c.name for c in self.client.get_collections().collections}
            if collection_name not in existing:
                logger.error("Collection '%s' does not exist", collection_name)
                return CollectionNotFound(collection_name=collection_name)

            # Get all points with their payloads
            points, _ = self.client.scroll(
                collection_name=collection_name,
                limit=10000,  # Adjust if needed for larger collections
                with_payload=True,
                with_vectors=False
            )

            # Group points by source_id and collect metadata
            source_info = {}
            for point in points:
                if not point.payload:
                    continue

                source_id = point.payload.get("source_id")
                if not source_id:
                    continue

                if source_id not in source_info:
                    source_info[source_id] = {
                        "filename": point.payload.get("filename"),
                        "url": point.payload.get("url"),
                        "type": point.payload.get("type", "pdf"),
                        "chunks_count": 0,
                        "pages": set(),
                        "tags": point.payload.get("tags", []),
                        "extras": point.payload.get("extras", None),
                        "uploaded_at": point.payload.get("uploaded_at", "")
                    }

                source_info[source_id]["chunks_count"] += 1
                if "page_number" in point.payload:
                    source_info[source_id]["pages"].add(
                        point.payload["page_number"])

            # Convert to SourceInfo objects
            sources = []
            for source_id, info in source_info.items():
                pages = sorted(info["pages"])
                sources.append(SourceInfo(
                    source_id=source_id,
                    filename=info["filename"],
                    url=info["url"],
                    type=info["type"],
                    first_page=min(pages) if pages else 0,
                    last_page=max(pages) if pages else 0,
                    chunks_count=info["chunks_count"],
                    tags=info["tags"],
                    extras=info["extras"],
                    uploaded_at=info["uploaded_at"]
                ))

            # Sort sources by source_id
            sources.sort(key=lambda x: x.source_id)

            logger.info(
                "Found %d sources in collection=%s",
                len(sources),
                collection_name)
            return SourceList(
                collection_name=collection_name,
                sources=sources,
                total=len(sources)
            )

        except Exception as e:
            logger.exception("Error while listing sources: %s", str(e))
            return SourceListError(
                collection_name=collection_name,
                error=str(e)
            )
