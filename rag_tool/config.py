import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# Vector store configuration
# Base directory where collections will be persisted
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vectorstores")

# RAG API configuration
# Used by LangGraph tool or other clients
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

# Indexing parameters
# Maximum token/chunk size when splitting documents
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
# Overlap between chunks to preserve context
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Query parameters
# Number of top documents to retrieve for queries
TOP_K = int(os.getenv("TOP_K", "5"))
