"""
Document processing module for handling PDF and text files.
Implements chunking strategies and text extraction.
"""

import os
from typing import List, Dict, Any, Generator
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter
)
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
import tempfile

class DocumentProcessor:
    """
    Handles document loading, text extraction, and chunking.
    Implements various chunking strategies for optimal RAG performance.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Number of characters per chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize different chunking strategies
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_document(self, file_content: bytes, file_name: str) -> List[Document]:
        """
        Load document from uploaded file content.
        
        Args:
            file_content: Binary content of uploaded file
            file_name: Name of the file
        
        Returns:
            List of Document objects
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            # Load based on file extension
            if file_name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
            else:  # Text files
                loader = TextLoader(tmp_path, encoding='utf-8')
                documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'source_file': file_name,
                    'chunk_strategy': 'recursive'
                })
            
            return documents
            
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of loaded documents
        
        Returns:
            List of chunked documents
        """
        chunked_docs = self.text_splitter.split_documents(documents)
        
        # Add chunk indices to metadata
        for i, doc in enumerate(chunked_docs):
            doc.metadata['chunk_index'] = i
            doc.metadata['total_chunks'] = len(chunked_docs)
        
        return chunked_docs
    
    def semantic_chunking(self, documents: List[Document]) -> List[Document]:
        """
        Advanced chunking based on semantic boundaries.
        Useful for better context preservation.
        """
        # This is a placeholder for more advanced chunking
        # Could implement using NLP techniques
        return self.chunk_documents(documents)
    
    def process_document(self, file_content: bytes, file_name: str) -> Dict[str, Any]:
        """
        Complete document processing pipeline.
        
        Returns:
            Dictionary with processed chunks and metadata
        """
        # Load document
        documents = self.load_document(file_content, file_name)
        
        # Chunk documents
        chunked_docs = self.chunk_documents(documents)
        
        return {
            'chunks': chunked_docs,
            'total_chunks': len(chunked_docs),
            'original_docs': len(documents),
            'metadata': {
                'file_name': file_name,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            }
        }