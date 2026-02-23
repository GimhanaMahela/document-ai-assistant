"""
RAG implementation using LangChain.
Handles question answering with retrieved context.

LLM priority: Groq (llama-3.3-70b-versatile) → OpenAI → Local Flan-T5
"""

from typing import List, Dict, Any, Optional
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# ── Cached LLM loaders ────────────────────────────────────────────────────────
# @st.cache_resource ensures each model loads exactly once per session.
# Without this, the model would reload on every Streamlit rerun (every click).

@st.cache_resource(show_spinner="Connecting to Groq...")
def _load_groq_llm(api_key: str, model_name: str):
    """Load and cache Groq LLM. Fastest inference, free tier available."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.7,
        streaming=True,
    )


@st.cache_resource(show_spinner="Connecting to OpenAI...")
def _load_openai_llm(api_key: str):
    """Load and cache OpenAI LLM."""
    from langchain.chat_models import ChatOpenAI
    from langchain.callbacks import StreamingStdOutCallbackHandler
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )


@st.cache_resource(show_spinner="Loading local Flan-T5 model (first run may take a minute)...")
def _load_local_llm():
    """Load and cache local Flan-T5 model. No API key needed, runs offline."""
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


# ── Provider info ─────────────────────────────────────────────────────────────

def get_active_provider() -> Dict[str, str]:
    """
    Return active LLM provider info for display in the UI.
    Priority order: Groq → OpenAI → Local Flan-T5
    """
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key:
        model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
        return {"provider": "Groq", "model": model, "icon": "⚡"}

    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        return {"provider": "OpenAI", "model": "gpt-3.5-turbo", "icon": "🤖"}

    return {"provider": "Local", "model": "Flan-T5-Base", "icon": "💻"}


# ── RAG Chat Engine ───────────────────────────────────────────────────────────

class RAGChatEngine:
    """
    Retrieval-Augmented Generation engine for document-based Q&A.

    LLM selection priority:
        1. Groq  — if GROQ_API_KEY is set in .env  (fastest, free tier)
        2. OpenAI — if OPENAI_API_KEY is set in .env
        3. Flan-T5 — local fallback, no API key required
    """

    def __init__(self, vector_store):
        """
        Initialize the chat engine.

        Args:
            vector_store: Initialized VectorStoreManager instance.
        """
        self.vector_store = vector_store
        self.llm = self._get_llm()
        self.qa_chain = None
        self._setup_chain()

    def _get_llm(self):
        """Return the cached LLM based on available API keys."""
        groq_key = os.getenv('GROQ_API_KEY')
        if groq_key:
            model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
            return _load_groq_llm(groq_key, model)

        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            return _load_openai_llm(openai_key)

        return _load_local_llm()

    def _setup_chain(self):
        """Build the RetrievalQA chain with a custom prompt template."""
        template = """You are a helpful AI assistant answering questions \
based on the provided context.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided context
- If the answer isn't in the context, say \
"I cannot find this information in the provided documents"
- Be concise but thorough
- Include relevant details from the context

Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
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
        Ask a question and get a grounded answer from uploaded documents.

        Args:
            question: User's natural language question.

        Returns:
            Dict with 'answer' (str) and 'sources' (list of dicts).
        """
        if not self.qa_chain:
            provider = get_active_provider()
            return {
                "answer": (
                    f"LLM not configured. "
                    f"Please set GROQ_API_KEY in your .env file to use {provider['provider']}."
                ),
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
        """Streaming version of ask — to be implemented in Phase 2."""
        pass
