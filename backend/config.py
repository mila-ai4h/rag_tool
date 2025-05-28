import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# API settings
API_KEY = os.getenv("X_API_KEY", "secret-key")
API_KEY_NAME = "x-api-key"

# Qdrant settings
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Node splitter settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Embedding model
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMENSIONS = 1536

# Query settings
DEFAULT_TOP_K = 5

# LLM settings
LLM_MODEL = "gpt-4o"  # OpenAI model to use for answer generation
