import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF
import requests
import trafilatura
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from models import (
    CollectionCreated,
    CollectionDeleted,
    CollectionError,
    CollectionExists,
    CollectionInfo,
    CollectionList,
    CollectionNotFound,
    DocumentEmptyError,
    DocumentError,
    DocumentIndexed,
    SourceDeleted,
    SourceError,
    SourceInfo,
    SourceList,
    SourceListError,
)
from playwright.sync_api import sync_playwright

# Configure logging
logger = logging.getLogger(__name__)


class SimpleMemoryStore:
    """A simple in-memory storage for documents and their metadata."""

    def __init__(self):
        self.documents = {}  # node_id -> document_data
        self.metadata = {}  # node_id -> metadata

    def add_document(self, node_id: str, text: str, metadata: dict):
        """Add a document to the store."""
        self.documents[node_id] = text
        self.metadata[node_id] = metadata

    def get_document(self, node_id: str) -> Tuple[str, dict]:
        """Get a document from the store."""
        return self.documents.get(node_id, ""), self.metadata.get(node_id, {})

    def remove_document(self, node_id: str):
        """Remove a document from the store."""
        if node_id in self.documents:
            del self.documents[node_id]
        if node_id in self.metadata:
            del self.metadata[node_id]

    def get_all_documents(self) -> List[Tuple[str, str, dict]]:
        """Get all documents as (node_id, text, metadata) tuples."""
        return [
            (node_id, text, self.metadata.get(node_id, {}))
            for node_id, text in self.documents.items()
        ]

    def count(self) -> int:
        """Get the number of documents in the store."""
        return len(self.documents)

    def get_by_source_id(self, source_id: str) -> List[Tuple[str, str, dict]]:
        """Get all documents for a specific source_id."""
        result = []
        for node_id, text in self.documents.items():
            metadata = self.metadata.get(node_id, {})
            if metadata.get("source_id") == source_id:
                result.append((node_id, text, metadata))
        return result


class Indexer:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        embed_model: str,
        embed_dimensions: int,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_model = embed_model
        self.embed_dimensions = embed_dimensions

        # Store memory stores for each collection
        self.memory_stores = {}

        logger.info(
            "Initializing Indexer with chunk_size=%d, chunk_overlap=%d, embed_model=%s, embed_dimensions=%d",
            chunk_size,
            chunk_overlap,
            embed_model,
            embed_dimensions,
        )

        # Configure LlamaIndex global settings
        Settings.embed_model = OpenAIEmbedding(model=embed_model)
        Settings.include_metadata = True
        Settings.include_embeddings = True
        Settings.disable_relationship_storage = True

    def create_collection(
        self, collection_name: str
    ) -> Union[CollectionCreated, CollectionExists, CollectionError]:
        """Create a new collection with the specified name.

        Returns:
            CollectionCreated: If the collection was successfully created
            CollectionExists: If the collection already exists
            CollectionError: For other processing errors
        """
        try:
            if collection_name in self.memory_stores:
                logger.info("Collection '%s' already exists", collection_name)
                return CollectionExists(collection_name=collection_name)

            # Create a new memory store for this collection
            memory_store = SimpleMemoryStore()
            self.memory_stores[collection_name] = memory_store

            logger.info("Collection '%s' created successfully", collection_name)
            return CollectionCreated(collection_name=collection_name)

        except Exception as e:
            logger.exception(
                "Error creating collection=%s: %s", collection_name, str(e)
            )
            return CollectionError(collection_name=collection_name, error=str(e))

    def list_collections(self):
        logger.info("Listing all collections")
        try:
            infos = []
            for collection_name in self.memory_stores:
                memory_store = self.memory_stores[collection_name]
                points_count = memory_store.count()

                infos.append(
                    CollectionInfo(
                        name=collection_name,
                        vector_size=self.embed_dimensions,
                        distance="cosine",
                        points_count=points_count,
                    )
                )

            logger.info("Successfully listed %d collections", len(infos))
            return CollectionList(collections=infos, total=len(infos))

        except Exception as e:
            logger.exception("Error listing collections: %s", str(e))
            raise  # Re-raise as this is a core operation

    def delete_collection(self, name: str) -> Union[CollectionDeleted, CollectionError]:
        """Delete a collection and all its content.

        Returns:
            CollectionDeleted: If the collection was successfully deleted or did not exist
            CollectionError: For other processing errors
        """
        logger.info("Attempting to delete collection=%s", name)
        try:
            if name not in self.memory_stores:
                logger.info("Collection '%s' does not exist, nothing to delete", name)
                return CollectionDeleted(collection_name=name)

            # Get count before deletion for logging
            memory_store = self.memory_stores[name]
            count = memory_store.count()
            logger.info("Collection=%s has %d points before deletion", name, count)

            # Remove the collection from our storage
            del self.memory_stores[name]

            logger.info(
                "Successfully deleted collection=%s with %d points", name, count
            )
            return CollectionDeleted(collection_name=name)

        except Exception as e:
            logger.exception("Error deleting collection=%s: %s", name, str(e))
            return CollectionError(collection_name=name, error=str(e))

    def _extract_documents_from_pdf(
        self,
        pdf_path: str,
        filename: str,
        source_id: str,
        tags: List[str],
        uploaded_at: str,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Tuple[list[Document], int]:
        """Extract text from PDF and create Document objects with metadata."""
        logger.info(
            "Starting PDF extraction for file=%s, source_id=%s, tags=%s, extras=%s",
            filename,
            source_id,
            tags,
            extras,
        )
        documents = []

        logger.info("Using upload timestamp: %s", uploaded_at)

        with fitz.open(pdf_path) as pdf:
            total_pages = len(pdf)
            logger.info("PDF opened successfully, total pages=%d", total_pages)

            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text()
                if text.strip():  # Skip empty pages
                    documents.append(
                        Document(
                            text=text,
                            metadata={
                                "source_id": source_id,
                                "filename": filename,
                                "url": None,  # PDFs don't have URLs
                                "type": "pdf",
                                "page_number": page_number,
                                "tags": tags,
                                "extras": extras,
                                "uploaded_at": uploaded_at,
                            },
                        )
                    )
                    logger.debug(
                        "Extracted text from page %d/%d", page_number, total_pages
                    )
                else:
                    logger.debug("Skipping empty page %d/%d", page_number, total_pages)

        logger.info(
            "PDF extraction completed, extracted %d non-empty pages", len(documents)
        )
        return documents, len(documents)

    def _extract_document_from_url(
        self,
        url: str,
        source_id: str,
        tags: List[str],
        uploaded_at: str,
        extras: Optional[Dict[str, Any]] = None,
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
                include_formatting=True,
            )

            if not text or len(text.strip()) < 100:  # Basic validation
                raise ValueError(f"No substantial text content found in {url}")

            # Add title if available
            if title:
                text = f"{title}\n\n{text}"

            logger.info(
                "Successfully extracted content from %s (length=%d)", url, len(text)
            )

            return Document(
                text=text,
                metadata={
                    "source_id": source_id,
                    "url": url,
                    "type": "url",
                    "tags": tags,
                    "extras": extras,
                    "uploaded_at": uploaded_at,
                },
            )

        except requests.RequestException as e:
            logger.error("Failed to fetch URL=%s: %s", url, str(e))
            raise
        except Exception as e:
            logger.error("Error extracting content from URL=%s: %s", url, str(e))
            raise ValueError(f"Failed to extract content from {url}: {str(e)}")

    def _delete_source_chunks(
        self, collection_name: str, source_id: str
    ) -> Union[None, CollectionNotFound, DocumentError]:
        """Delete all chunks for a given source_id from the collection.

        Returns:
            None: If deletion was successful
            CollectionNotFound: If the collection does not exist
            DocumentError: For other processing errors
        """
        try:
            if collection_name not in self.memory_stores:
                logger.error("Collection '%s' does not exist", collection_name)
                return CollectionNotFound(collection_name=collection_name)

            memory_store = self.memory_stores[collection_name]

            # Find and remove nodes with matching source_id
            nodes_to_remove = []
            for node_id, text, metadata in memory_store.get_all_documents():
                if metadata.get("source_id") == source_id:
                    nodes_to_remove.append(node_id)

            for node_id in nodes_to_remove:
                memory_store.remove_document(node_id)

            logger.info(
                "Successfully deleted %d chunks for source_id=%s",
                len(nodes_to_remove),
                source_id,
            )
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
        extras: Optional[Dict[str, Any]] = None,
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
            extras,
        )

        try:
            # Get current timestamp in ISO format with UTC timezone indicator
            uploaded_at = datetime.utcnow().isoformat() + "Z"
            logger.info("Using upload timestamp: %s", uploaded_at)

            if collection_name not in self.memory_stores:
                logger.error("Collection '%s' does not exist", collection_name)
                return CollectionNotFound(collection_name=collection_name)

            # Chunk with SentenceSplitter
            logger.info(
                "Starting document chunking with chunk_size=%d, chunk_overlap=%d",
                self.chunk_size,
                self.chunk_overlap,
            )
            splitter = SentenceSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            nodes = splitter.get_nodes_from_documents([document])
            logger.info("Created %d chunks from document", len(nodes))

            # Get the memory store for this collection
            memory_store = self.memory_stores[collection_name]

            # Store each node in the memory store
            logger.info("Starting document storage")
            for i, node in enumerate(nodes):
                node_id = f"{source_id}_{i}"
                memory_store.add_document(node_id, node.text, node.metadata)

            # Get source identifier (filename or url) from document metadata
            source_identifier = document.metadata.get(
                "filename"
            ) or document.metadata.get("url")
            if not source_identifier:
                raise ValueError(
                    "Document metadata must contain either filename or url"
                )

            logger.info(
                "Successfully completed document indexing for %s", source_identifier
            )
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
                message="Document indexed successfully",
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
        extras: Optional[Dict[str, Any]] = None,
    ) -> Union[DocumentIndexed, DocumentError, DocumentEmptyError, CollectionNotFound]:
        logger.info(
            "Starting PDF indexing process for collection=%s, file=%s, tags=%s, extras=%s",
            collection_name,
            filename,
            tags,
            extras,
        )

        try:
            # Generate or use provided source_id
            source_id = source_id or str(uuid.uuid4())
            logger.info("Using source_id=%s", source_id)

            # Normalize tags
            tags = tags or []

            # Delete any existing chunks for this source_id
            delete_result = self._delete_source_chunks(collection_name, source_id)
            if delete_result is not None:
                return delete_result

            # Extract documents from PDF (one per page)
            documents, pages_count = self._extract_documents_from_pdf(
                file_path,
                filename,
                source_id,
                tags,
                datetime.utcnow().isoformat() + "Z",
                extras,
            )
            if not documents:
                logger.error("No text content found in PDF file=%s", filename)
                return DocumentEmptyError(
                    collection_name=collection_name,
                    filename=filename,
                    url=None,
                    message="No text content found in PDF file",
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
                    extras=extras,
                )

                if isinstance(result, DocumentIndexed):
                    total_chunks += result.chunks_created
                else:
                    # If any page fails, return the error
                    return result

            # Create a success response with the total chunks
            logger.info(
                "Successfully indexed %d pages from PDF file=%s with %d total chunks",
                pages_count,
                filename,
                total_chunks,
            )
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
                message="Document indexed successfully",
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
        extras: Optional[Dict[str, Any]] = None,
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
            extras,
        )

        try:
            # Generate or use provided source_id
            source_id = source_id or str(uuid.uuid4())
            logger.info("Using source_id=%s", source_id)

            # Normalize tags
            tags = tags or []

            # Delete any existing chunks for this source_id
            delete_result = self._delete_source_chunks(collection_name, source_id)
            if delete_result is not None:
                return delete_result

            # Extract document from URL
            try:
                document = self._extract_document_from_url(
                    url, source_id, tags, datetime.utcnow().isoformat() + "Z", extras
                )
            except ValueError as e:
                logger.error("No text content found in URL=%s", url)
                return DocumentEmptyError(
                    collection_name=collection_name,
                    filename=None,
                    url=url,
                    message="No text content found in URL",
                )
            except requests.RequestException as e:
                logger.error("Failed to fetch URL=%s: %s", url, str(e))
                return DocumentError(
                    collection_name=collection_name,
                    error=f"Failed to fetch URL: {str(e)}",
                )

            return self._index_document(
                collection_name=collection_name,
                source_id=source_id,
                document=document,
                type="url",
                tags=tags,
                extras=extras,
            )

        except Exception as e:
            logger.exception("Error during URL indexing: %s", str(e))
            return DocumentError(collection_name=collection_name, error=str(e))

    def delete_by_source_id(
        self, collection_name: str, source_id: str
    ) -> Union[SourceDeleted, SourceError, CollectionNotFound]:
        """Delete all content associated with a given source_id from the collection.

        Returns:
            SourceDeleted: Information about the deleted source
            CollectionNotFound: If the collection does not exist
            SourceError: For other processing errors
        """
        logger.info(
            "Starting deletion of content for collection=%s, source_id=%s",
            collection_name,
            source_id,
        )
        try:
            if collection_name not in self.memory_stores:
                logger.error("Collection '%s' does not exist", collection_name)
                return CollectionNotFound(collection_name=collection_name)

            memory_store = self.memory_stores[collection_name]

            # Find and remove nodes with matching source_id
            nodes_to_remove = []
            for node_id, text, metadata in memory_store.get_all_documents():
                if metadata.get("source_id") == source_id:
                    nodes_to_remove.append(node_id)

            for node_id in nodes_to_remove:
                memory_store.remove_document(node_id)

            logger.info(
                "Successfully deleted %d points for source_id=%s",
                len(nodes_to_remove),
                source_id,
            )

            return SourceDeleted(
                collection_name=collection_name,
                source_id=source_id,
                points_deleted=len(nodes_to_remove),
            )

        except Exception as e:
            logger.exception("Error during source deletion: %s", str(e))
            return SourceError(
                collection_name=collection_name, source_id=source_id, error=str(e)
            )

    def list_sources(
        self, collection_name: str
    ) -> Union[SourceList, SourceListError, CollectionNotFound]:
        """List all sources in a collection with their details.

        Returns:
            SourceList: List of sources in the collection
            CollectionNotFound: If the collection does not exist
            SourceListError: For other processing errors
        """
        logger.info("Listing sources for collection=%s", collection_name)
        try:
            if collection_name not in self.memory_stores:
                logger.error("Collection '%s' does not exist", collection_name)
                return CollectionNotFound(collection_name=collection_name)

            memory_store = self.memory_stores[collection_name]

            # Group nodes by source_id and collect metadata
            source_info = {}
            for node_id, text, metadata in memory_store.get_all_documents():
                source_id = metadata.get("source_id")
                if not source_id:
                    continue

                if source_id not in source_info:
                    source_info[source_id] = {
                        "filename": metadata.get("filename"),
                        "url": metadata.get("url"),
                        "type": metadata.get("type", "pdf"),
                        "chunks_count": 0,
                        "pages": set(),
                        "tags": metadata.get("tags", []),
                        "extras": metadata.get("extras", None),
                        "uploaded_at": metadata.get("uploaded_at", ""),
                    }

                source_info[source_id]["chunks_count"] += 1
                if "page_number" in metadata:
                    source_info[source_id]["pages"].add(metadata["page_number"])

            # Convert to SourceInfo objects
            sources = []
            for source_id, info in source_info.items():
                pages = sorted(info["pages"])
                sources.append(
                    SourceInfo(
                        source_id=source_id,
                        filename=info["filename"],
                        url=info["url"],
                        type=info["type"],
                        first_page=min(pages) if pages else 0,
                        last_page=max(pages) if pages else 0,
                        chunks_count=info["chunks_count"],
                        tags=info["tags"],
                        extras=info["extras"],
                        uploaded_at=info["uploaded_at"],
                    )
                )

            # Sort sources by source_id
            sources.sort(key=lambda x: x.source_id)

            logger.info(
                "Found %d sources in collection=%s", len(sources), collection_name
            )
            return SourceList(
                collection_name=collection_name, sources=sources, total=len(sources)
            )

        except Exception as e:
            logger.exception("Error while listing sources: %s", str(e))
            return SourceListError(collection_name=collection_name, error=str(e))

    def get_memory_store(self, collection_name: str):
        """Get the memory store for a collection."""
        return self.memory_stores.get(collection_name)
