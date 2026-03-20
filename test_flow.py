import os
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from chat_engine import RAGChatEngine

def test_flow():
    print("Testing flow...")
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    
    # Create dummy text file
    content = b"This is a test document. It contains some text for testing the RAG system."
    result = processor.process_document(content, "test.txt")
    print(f"Processed into {result['total_chunks']} chunks")
    
    vsm = VectorStoreManager(persist_directory=".test_chroma")
    vsm.create_vector_store(result['chunks'])
    print("Vector store created")
    
    engine = RAGChatEngine(vsm)
    res = engine.ask("What is this document about?")
    print(f"Answer: {res['answer']}")

if __name__ == '__main__':
    test_flow()
