"""
RAG implementation using LangChain.
Handles question answering with retrieved context.
"""

from typing import List, Dict, Any, Optional
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()

class RAGChatEngine:
    """
    Implements Retrieval-Augmented Generation for document-based Q&A.
    """
    
    def __init__(self, vector_store):
        """
        Initialize chat engine.
        
        Args:
            vector_store: Initialized vector store for retrieval
        """
        self.vector_store = vector_store
        self.llm = self._get_llm()
        self.qa_chain = None
        self._setup_chain()
    
    def _get_llm(self):
        """Initialize LLM (OpenAI or local)."""
        api_key = os.getenv('OPENAI_API_KEY')

        if api_key:
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()]
            )
        else:
            from langchain.llms import HuggingFacePipeline
            from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

            model_name = "google/flan-t5-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
            )
            return HuggingFacePipeline(pipeline=pipe)
    
    def _setup_chain(self):
        """Setup the QA chain with custom prompt."""
        
        # Custom prompt template
        template = """
        You are a helpful AI assistant answering questions based on the provided context.
        
        Context from documents:
        {context}
        
        Question: {question}
        
        Instructions:
        - Answer based ONLY on the provided context
        - If the answer isn't in the context, say "I cannot find this information in the provided documents"
        - Be concise but thorough
        - Include relevant details from the context
        
        Answer:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # Can be "map_reduce", "refine", "map_rerank"
            retriever=self.vector_store.vector_store.as_retriever(
                search_kwargs={"k": 4}
            ),
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": True
            },
            return_source_documents=True
        )
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get answer based on documents.
        
        Args:
            question: User's question
        
        Returns:
            Dictionary with answer and source documents
        """
        if not self.qa_chain:
            return {
                "answer": "LLM not configured. Please set up OpenAI API key or local LLM.",
                "sources": []
            }
        
        try:
            result = self.qa_chain({"query": question})
            
            return {
                "answer": result["result"],
                "sources": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in result["source_documents"]
                ]
            }
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": []
            }
    
    def ask_streaming(self, question: str):
        """
        Streaming version of ask for real-time responses.
        """
        # Implementation for streaming responses
        pass