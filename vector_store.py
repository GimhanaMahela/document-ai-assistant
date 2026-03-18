"""
Vector database operations using ChromaDB and FAISS.
Handles embedding generation and similarity search.
"""

from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
import chromadb
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


@st.cache_resource(show_spinner="Loading embedding model...")
def _load_openai_embeddings(api_key: str):
    """Load and cache OpenAI embeddings. Runs once per session."""
    return OpenAIEmbeddings(
        openai_api_key=api_key,
        model="text-embedding-ada-002"
    )


@st.cache_resource(show_spinner="Loading embedding model...")
def _load_huggingface_embeddings():
    """Load and cache HuggingFace embeddings. Runs once per session."""
    from langchain.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

class VectorStoreManager:
    """
    Manages vector database operations for document embeddings.
    Supports multiple vector store backends.
    """
    
    def __init__(self, persist_directory: str = "data/chroma_db"):
        """
        Initialize vector store manager.
        
        Args:
            persist_directory: Directory to persist vector database
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = self._get_embeddings()
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory
        )
        
        self.vector_store = None
    
    def _get_embeddings(self):
        """Return cached embedding model (OpenAI or local HuggingFace)."""
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            return OpenAIEmbeddings(
                openai_api_key=api_key,
                model="text-embedding-ada-002"
            )
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
    
    def create_vector_store(self, documents: List[Document], 
                           collection_name: str = "documents"):
        """
        Create vector store from documents.
        
        Args:
            documents: List of chunked documents
            collection_name: Name of the collection
        """
        # Create using Chroma
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=collection_name
        )
        
        # Persist to disk
        self.vector_store.persist()
        
        return self.vector_store
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to existing vector store."""
        if self.vector_store:
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: User query
            k: Number of results to return
        
        Returns:
            List of relevant documents
        """
        if not self.vector_store:
            return []
        
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        Search with relevance scores.
        """
        if not self.vector_store:
            return []
        
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def load_existing_store(self, collection_name: str = "documents"):
        """Load existing vector store from disk."""
        try:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
            return True
        except Exception as e:
            print(f"Error loading existing store: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if not self.vector_store:
            return {}
        
        collection = self.chroma_client.get_collection("documents")
        return {
            'count': collection.count(),
            'name': collection.name,
            'metadata': collection.metadata
        }