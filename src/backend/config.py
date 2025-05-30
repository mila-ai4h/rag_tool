import os

from dotenv import find_dotenv, load_dotenv

# Load environment variables from .env
load_dotenv(find_dotenv(".env"))

# API settings
API_KEY = os.getenv("X_API_KEY", "secret-key")
API_KEY_NAME = "x-api-key"

# Qdrant settings
QDRANT_URL = os.getenv("QDRANT_URL")  # Full URL for managed Qdrant service
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")  # Host for local Qdrant
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))  # Port for local Qdrant
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Optional API key for Qdrant

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
