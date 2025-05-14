# rag_tool/query_engine.py
from typing import List, Optional
import os

from llama_index.core import load_index_from_storage
from llama_index.core.storage import StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import NodeWithScore

from .config import VECTOR_STORE_PATH, LLM_MODEL, EMBEDDING_MODEL, TOP_K

class QueryEngine:
    def __init__(self, collection: str):
        # 1) Setup global settings with the configured LLM model
        llm = OpenAI(model_name=LLM_MODEL)
        embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)
        
        # Configure global settings
        Settings.llm = llm
        Settings.embed_model = embed_model

        # 2) Load the persisted index
        persist_dir = os.path.join(VECTOR_STORE_PATH, collection)
        storage = StorageContext.from_defaults(persist_dir=persist_dir)
        self.index = load_index_from_storage(storage)

    def raw_search(self, question: str, top_k: int = TOP_K) -> List[NodeWithScore]:
        """
        Perform a similarity search, returning the top_k documents.
        
        Args:
            question: The search query
            top_k: Number of top documents to retrieve (default: TOP_K from config)
            
        Returns:
            List of NodeWithScore objects containing the retrieved documents and their scores
        """
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        return retriever.retrieve(question)

    def answer(self, question: str, top_k: int = TOP_K):
        """
        Perform a full RAG query: retrieve relevant docs and generate an answer via the LLM.
        
        Args:
            question: The search query
            top_k: Number of top documents to retrieve (default: TOP_K from config)
        """
        query_engine = self.index.as_query_engine(similarity_top_k=top_k)
        return query_engine.query(question)
