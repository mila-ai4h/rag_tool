import os
import requests
import logging
from typing import Any, Dict, Optional, List
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class RAGTool:
    """A tool for interacting with the RAG API."""
    
    def __init__(self, api_url: Optional[str] = None):
        """Initialize the RAG tool.
        
        Args:
            api_url: Optional URL for the RAG API. If not provided, uses RAG_API_URL from .env
        """
        self.api_url = api_url or os.getenv('RAG_API_URL', 'http://localhost:8000')
        logger.debug(f"Initialized RAGTool with API URL: {self.api_url}")
        
    def rag_query(self, collection: str, query: str, answer: bool = False) -> Dict[str, Any]:
        """Query the RAG system.
        
        Args:
            collection: Name of the collection to query
            query: The query string
            answer: Whether to get an LLM-generated answer (True) or raw snippets (False)
            
        Returns:
            The raw response data from the RAG system
            
        Raises:
            Exception: If the query fails
        """
        try:
            url = f"{self.api_url}/collections/{collection}/query"
            logger.debug(f"Making request to: {url}")
            logger.debug(f"Query: {query}, answer: {answer}")
            
            response = requests.get(
                url,
                params={"q": query, "answer": answer}
            )
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response content: {response.text}")
            
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, dict):
                raise ValueError("Response is not a dictionary")
                
            # Validate response format
            if "snippets" not in data:
                raise ValueError("Response missing 'snippets' field")
            if answer and "answer" not in data:
                raise ValueError("Response missing 'answer' field when answer=True")
                
            return data
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to query RAG system: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except (KeyError, ValueError) as e:
            error_msg = f"Invalid response format from RAG system: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
