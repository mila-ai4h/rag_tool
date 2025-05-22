from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
from llama_index.core import Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
import fitz  # PyMuPDF
import uuid
import logging
from typing import Optional, Tuple, List

from .models import (
    CollectionCreated,
    CollectionExists,
    CollectionError,
    CollectionInfo,
    CollectionList,
    CollectionDeleted,
    DocumentIndexed,
    DocumentError,
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
            chunk_size, chunk_overlap, embed_model, embed_dimensions
        )

        # Configure LlamaIndex global settings
        Settings.embed_model = OpenAIEmbedding(model=embed_model)
        Settings.include_metadata = True
        Settings.include_embeddings = True
        Settings.disable_relationship_storage = True

    def create_collection(self, name: str):
        logger.info("Attempting to create collection=%s", name)
        try:
            existing = {c.name for c in self.client.get_collections().collections}
            if name in existing:
                logger.warning("Collection '%s' already exists", name)
                return CollectionExists(collection_name=name)

            logger.info("Creating collection=%s with vector size=%d", name, self.embed_dimensions)
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
                count = self.client.count(collection_name=col.name, exact=True).count
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

    def delete_collection(self, name: str):
        logger.info("Attempting to delete collection=%s", name)
        try:
            existing = {c.name for c in self.client.get_collections().collections}
            if name not in existing:
                logger.warning("Collection '%s' does not exist", name)
                return CollectionError(collection_name=name, error=f"'{name}' does not exist")

            # Get count before deletion for logging
            count = self.client.count(collection_name=name, exact=True).count
            logger.info("Collection=%s has %d points before deletion", name, count)

            self.client.delete_collection(collection_name=name)
            logger.info("Successfully deleted collection=%s with %d points", name, count)
            return CollectionDeleted(collection_name=name)
            
        except Exception as e:
            logger.exception("Error deleting collection=%s: %s", name, str(e))
            return CollectionError(collection_name=name, error=str(e))

    def _extract_documents_from_pdf(self, pdf_path: str, filename: str, source_id: str, tags: List[str]) -> Tuple[list[Document], int]:
        """Extract text from PDF and create Document objects with metadata."""
        logger.info("Starting PDF extraction for file=%s, source_id=%s, tags=%s", filename, source_id, tags)
        documents = []
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
                            "page_number": page_number,
                            "filename": filename,
                            "tags": tags,
                            "type": "pdf"
                        }
                    ))
                    logger.debug("Extracted text from page %d/%d", page_number, total_pages)
                else:
                    logger.debug("Skipping empty page %d/%d", page_number, total_pages)
        
        logger.info("PDF extraction completed, extracted %d non-empty pages", len(documents))
        return documents, len(documents)

    def index_pdf(self, collection_name: str, pdf_path: str, filename: str, source_id: Optional[str] = None, tags: Optional[List[str]] = None) -> DocumentIndexed | DocumentError:
        """Index a PDF file into the specified collection."""
        logger.info("Starting PDF indexing process for collection=%s, file=%s, tags=%s", collection_name, filename, tags)
        try:
            # Verify collection exists
            existing = {c.name for c in self.client.get_collections().collections}
            if collection_name not in existing:
                logger.error("Collection '%s' does not exist", collection_name)
                return DocumentError(collection_name=collection_name, error=f"Collection '{collection_name}' does not exist")

            # Generate or use provided source_id
            source_id = source_id or str(uuid.uuid4())
            logger.info("Using source_id=%s", source_id)

            # Normalize tags
            tags = tags or []

            # Check if there are existing chunks with this source_id and delete them
            count_before = self.client.count(
                collection_name=collection_name,
                count_filter=Filter(
                    must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
                ),
                exact=True
            ).count

            if count_before > 0:
                logger.info("Found %d existing chunks with source_id=%s, deleting them first", count_before, source_id)
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=Filter(
                        must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
                    )
                )
                logger.info("Successfully deleted existing chunks for source_id=%s", source_id)

            # Extract documents from PDF
            documents, pages_count = self._extract_documents_from_pdf(pdf_path, filename, source_id, tags)
            if not documents:
                logger.error("No text content found in PDF file=%s", filename)
                return DocumentError(collection_name=collection_name, error="No text content found in PDF")

            # Chunk with SentenceSplitter
            logger.info("Starting document chunking with chunk_size=%d, chunk_overlap=%d", 
                       self.chunk_size, self.chunk_overlap)
            splitter = SentenceSplitter(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            nodes = splitter.get_nodes_from_documents(documents)
            logger.info("Created %d chunks from %d pages", len(nodes), pages_count)

            # Embed and store in Qdrant
            logger.info("Starting embedding generation and vector storage")
            vector_store = QdrantVectorStore(client=self.client, collection_name=collection_name)
            
            # Process nodes in batches for better logging
            batch_size = 10
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                logger.info("Processing embedding batch %d-%d/%d", 
                           i + 1, min(i + batch_size, len(nodes)), len(nodes))
                texts = [node.text for node in batch]
                embeddings = Settings.embed_model.get_text_embedding_batch(texts)
                for node, embedding in zip(batch, embeddings):
                    node.embedding = embedding
                vector_store.add(batch)
                            
            logger.info("Successfully completed PDF indexing for file=%s", filename)
            return DocumentIndexed(
                collection_name=collection_name,
                source_id=source_id,
                pages_indexed=pages_count,
                chunks_created=len(nodes),
                tags=tags
            )

        except Exception as e:
            logger.exception("Error during PDF indexing: %s", str(e))
            return DocumentError(collection_name=collection_name, error=str(e))

    def delete_by_source_id(self, collection_name: str, source_id: str) -> SourceDeleted | SourceError:
        """Delete all content associated with a given source_id from the collection."""
        logger.info("Starting deletion of content for collection=%s, source_id=%s", collection_name, source_id)
        try:
            # Verify collection exists
            existing = {c.name for c in self.client.get_collections().collections}
            if collection_name not in existing:
                logger.error("Collection '%s' does not exist", collection_name)
                return SourceError(
                    collection_name=collection_name,
                    source_id=source_id,
                    error=f"Collection '{collection_name}' does not exist"
                )

            # Debug: Get total points in collection
            total_points = self.client.count(collection_name=collection_name, exact=True).count
            logger.info("Total points in collection: %d", total_points)

            # Debug: List all unique source_ids in collection
            search_result = self.client.scroll(
                collection_name=collection_name,
                limit=100,  # Adjust if needed
                with_payload=True,
                with_vectors=False
            )[0]  # scroll returns (points, next_page_offset)
            
            unique_source_ids = {point.payload.get("source_id") for point in search_result if point.payload}
            logger.info("Found source_ids in collection: %s", unique_source_ids)

            # Get count before deletion to know how many points were deleted
            count_before = self.client.count(
                collection_name=collection_name,
                count_filter=Filter(
                    must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
                ),
                exact=True
            ).count
            logger.info("Found %d points matching source_id=%s", count_before, source_id)

            if count_before == 0:
                logger.warning("No points found with source_id=%s in collection=%s", source_id, collection_name)
                return SourceDeleted(
                    collection_name=collection_name,
                    source_id=source_id,
                    points_deleted=0
                )

            # Delete points matching the source_id using proper filter models
            self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
                )
            )

            logger.info("Successfully deleted %d points for source_id=%s", count_before, source_id)
            
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

    def list_sources(self, collection_name: str) -> SourceList | SourceListError:
        """List all sources in a collection with their details."""
        logger.info("Listing sources for collection=%s", collection_name)
        try:
            # Verify collection exists
            existing = {c.name for c in self.client.get_collections().collections}
            if collection_name not in existing:
                logger.error("Collection '%s' does not exist", collection_name)
                return SourceListError(
                    collection_name=collection_name,
                    error=f"Collection '{collection_name}' does not exist"
                )

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
                        "filename": point.payload.get("filename", "unknown"),
                        "chunks_count": 0,
                        "pages": set(),
                        "tags": point.payload.get("tags", []),
                        "type": point.payload.get("type", "pdf")
                    }
                
                source_info[source_id]["chunks_count"] += 1
                if "page_number" in point.payload:
                    source_info[source_id]["pages"].add(point.payload["page_number"])

            # Convert to SourceInfo objects
            sources = []
            for source_id, info in source_info.items():
                pages = sorted(info["pages"])
                sources.append(SourceInfo(
                    source_id=source_id,
                    filename=info["filename"],
                    chunks_count=info["chunks_count"],
                    first_page=min(pages) if pages else 0,
                    last_page=max(pages) if pages else 0,
                    tags=info["tags"],
                    type=info["type"]
                ))

            # Sort sources by filename
            sources.sort(key=lambda x: x.filename)

            logger.info("Found %d sources in collection=%s", len(sources), collection_name)
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
