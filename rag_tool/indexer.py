import os
import logging
import shutil
from typing import List, Dict, Any
import fitz  # PyMuPDF
from tiktoken import encoding_for_model

from llama_index.core import SimpleDirectoryReader, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.core.settings import Settings
from llama_index.core import Document

from .config import VECTOR_STORE_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, LLM_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Indexer:
    def __init__(self, storage_path: str = VECTOR_STORE_PATH):
        self.storage_path = storage_path
        # Initialize tokenizer based on embedding model
        self.tokenizer = encoding_for_model(EMBEDDING_MODEL)

    def inspect_collection(self, name: str) -> Dict[str, Any]:
        """
        Inspect a collection's contents and return statistics.
        
        Args:
            name: Name of the collection to inspect
            
        Returns:
            Dictionary containing collection statistics
        """
        persist_dir = os.path.join(self.storage_path, name)
        if not os.path.exists(persist_dir):
            raise ValueError(f"Collection '{name}' does not exist at {persist_dir}")

        # Load the index
        storage = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage)

        # Get basic stats
        stats = {
            "collection_name": name,
            "total_documents": len(index.docstore.docs),
            "total_nodes": len(index.docstore.docs),  # In newer versions, docs and nodes are the same
        }

        # Get vector store size if available
        if hasattr(index, '_vector_store'):
            try:
                # Try to get the number of vectors by accessing the internal data structure
                if hasattr(index._vector_store, '_data'):
                    data = index._vector_store._data
                    if hasattr(data, 'embeddings_dict'):
                        stats["vector_store_size"] = len(data.embeddings_dict)
                    elif hasattr(data, 'embeddings'):
                        stats["vector_store_size"] = len(data.embeddings)
                    else:
                        stats["vector_store_size"] = 0
                else:
                    stats["vector_store_size"] = 0
            except Exception as e:
                logger.warning(f"Could not get vector store size: {str(e)}")
                stats["vector_store_size"] = 0
        else:
            stats["vector_store_size"] = 0

        # Get document sources and calculate average chunk size
        sources = set()
        total_text_length = 0
        total_tokens = 0
        for doc_id, doc in index.docstore.docs.items():
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                sources.add(doc.metadata['source'])
            if hasattr(doc, 'text'):
                text = doc.text
                total_text_length += len(text)
                total_tokens += len(self.tokenizer.encode(text))

        stats["unique_sources"] = list(sources)
        stats["source_count"] = len(sources)
        
        # Calculate average sizes
        if stats["total_documents"] > 0:
            stats["average_chunk_size_chars"] = round(total_text_length / stats["total_documents"], 2)
            stats["average_chunk_size_tokens"] = round(total_tokens / stats["total_documents"], 2)
            stats["total_tokens"] = total_tokens
            stats["total_chars"] = total_text_length

        return stats

    def list_collections(self) -> List[str]:
        """
        List all available collections.
        
        Returns:
            List of collection names
        """
        if not os.path.exists(self.storage_path):
            return []
        
        return [d for d in os.listdir(self.storage_path) 
                if os.path.isdir(os.path.join(self.storage_path, d))]

    def create_collection(self, name: str, pdf_folder: str = None):
        """
        Create a new collection, optionally indexing PDFs from a folder.
        
        Args:
            name: Name of the collection
            pdf_folder: Optional path to folder containing PDFs to index
            
        Returns:
            Dictionary with collection info
        """
        # 1) Setup settings with embedding model, node parser, and LLM
        embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)
        llm = OpenAI(model_name=LLM_MODEL)
        node_parser = SimpleNodeParser(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        
        # Configure global settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.node_parser = node_parser

        # Create persist directory
        persist_dir = os.path.join(self.storage_path, name)
        os.makedirs(persist_dir, exist_ok=True)

        if pdf_folder:
            # 2) Load PDFs into Document objects with better error handling
            documents = []
            
            for root, _, files in os.walk(pdf_folder):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        file_path = os.path.join(root, file)
                        try:
                            logger.info(f"Processing file: {file_path}")
                            # Use PyMuPDF to extract text
                            doc = fitz.open(file_path)
                            text = ""
                            for page in doc:
                                text += page.get_text()
                            doc.close()
                            
                            # Create a Document object
                            documents.append(Document(text=text, metadata={"source": file_path}))
                        except Exception as e:
                            logger.error(f"Failed to load file {file_path} with error: {str(e)}. Skipping...")
                            continue

            if not documents:
                raise ValueError(f"No valid PDF documents found in {pdf_folder}")

            # 3) Build the vector index
            index = VectorStoreIndex.from_documents(documents)
        else:
            # Create an empty index
            index = VectorStoreIndex([])

        # 4) Persist into its own directory
        index.storage_context.persist(persist_dir=persist_dir)

        return {"collection": name, "count": len(documents) if pdf_folder else 0}

    def load_index(self, name: str):
        # 1) Rebuild storage context from persisted data
        persist_dir = os.path.join(self.storage_path, name)
        storage = StorageContext.from_defaults(persist_dir=persist_dir)

        # 2) Load and return the index
        return load_index_from_storage(storage)

    def delete_collection(self, name: str) -> Dict[str, Any]:
        """
        Delete a collection and all its associated files.
        
        Args:
            name: Name of the collection to delete
            
        Returns:
            Dictionary with deletion status
        """
        persist_dir = os.path.join(self.storage_path, name)
        if not os.path.exists(persist_dir):
            raise ValueError(f"Collection '{name}' does not exist at {persist_dir}")
        
        try:
            # Remove the entire collection directory
            shutil.rmtree(persist_dir)
            logger.info(f"Successfully deleted collection '{name}'")
            return {"status": "success", "message": f"Collection '{name}' deleted successfully"}
        except Exception as e:
            logger.error(f"Error deleting collection '{name}': {str(e)}")
            raise ValueError(f"Failed to delete collection: {str(e)}")

    def add_document(self, name: str, file_path: str) -> Dict[str, Any]:
        """
        Add a single document to an existing collection.
        
        Args:
            name: Name of the collection
            file_path: Path to the PDF file to add
            
        Returns:
            Dictionary with update status
        """
        persist_dir = os.path.join(self.storage_path, name)
        if not os.path.exists(persist_dir):
            raise ValueError(f"Collection '{name}' does not exist at {persist_dir}")

        try:
            # Load existing index
            storage = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage)

            # Extract text from PDF
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            # Create document object
            document = Document(text=text, metadata={"source": file_path})

            # Add document to index
            index.insert(document)

            # Persist updated index
            index.storage_context.persist(persist_dir=persist_dir)

            return {
                "status": "success",
                "message": f"Document added to collection '{name}'",
                "document": file_path
            }

        except Exception as e:
            logger.error(f"Error adding document to collection '{name}': {str(e)}")
            raise ValueError(f"Failed to add document: {str(e)}")
