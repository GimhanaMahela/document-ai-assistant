"""
RAG chat engine using a retrieve-then-generate pipeline.
Retrieval is done via similarity search; the LLM receives context directly,
avoiding tool-calling entirely (Groq Llama models generate tool calls in a
format Groq's own API rejects, causing 400 errors).

LLM priority: Groq (llama-3.3-70b-versatile) → OpenAI → Local Flan-T5
"""

# stdlib
import os
from typing import Dict, Any, List

# third-party
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

_SUMMARY_KEYWORDS = ("summar", "overview", "outline", "full content", "all document", "what is in")

_SYSTEM_PROMPT = (
    "You are a helpful AI assistant that answers questions based on uploaded documents. "
    "Use the provided context to answer accurately and concisely. "
    "If the context does not contain enough information to answer, say so clearly. "
    "Always cite the source file when referencing specific content."
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Context from uploaded documents:\n{context}\n\nQuestion: {question}"),
])


# ── Cached LLM loaders ────────────────────────────────────────────────────────
# Imports are intentionally inside these functions (lazy loading).
# This avoids loading all three heavy libraries on startup — only the
# selected provider's library is imported at runtime.
# pylint: disable=import-outside-toplevel

@st.cache_resource(show_spinner="Connecting to Groq...")
def _load_groq_llm(api_key: str, model_name: str):
    """Load and cache Groq LLM. Fastest inference, free tier available."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.7,
    )


@st.cache_resource(show_spinner="Connecting to OpenAI...")
def _load_openai_llm(api_key: str):
    """Load and cache OpenAI LLM."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=api_key,
        temperature=0.7,
    )


@st.cache_resource(show_spinner="Loading local Flan-T5 model (first run may take a minute)...")
def _load_local_llm():
    """Load and cache local Flan-T5 model. No API key needed, runs offline."""
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
    )
    return HuggingFacePipeline(pipeline=pipe)

# pylint: enable=import-outside-toplevel


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
    Retrieve-then-generate RAG engine for document-based Q&A.

    On each turn:
      1. Detect whether the question asks for a summary or a specific lookup.
      2. Retrieve relevant chunks from the vector store.
      3. Pass chunks as context directly to the LLM — no tool calling.

    Conversation history is maintained across turns in a session.

    LLM selection priority:
        1. Groq   — if GROQ_API_KEY is set in .env  (fastest, free tier)
        2. OpenAI — if OPENAI_API_KEY is set in .env
        3. Flan-T5 — local fallback, no API key required
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.chat_history: List = []
        self.llm = self._get_llm()
        self.chain = _PROMPT | self.llm

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

    def _retrieve_for_summary(self) -> tuple[str, List]:
        """Fetch up to 40 chunks from the store for a full-document summary."""
        try:
            data = self.vector_store.vector_store.get()
            texts = data.get("documents", [])[:40]
            metadatas = data.get("metadatas", []) or []
            if not texts:
                return "No documents found in the vector store.", []
            parts = [
                f"[Source: {(meta or {}).get('source_file', 'unknown')}]\n{text}"
                for text, meta in zip(texts, metadatas)
            ]
            return "\n\n---\n\n".join(parts), []
        except Exception:  # pylint: disable=broad-except
            docs = self.vector_store.similarity_search("", k=8)
            return self._format_docs(docs), docs

    def _format_docs(self, docs: List) -> str:
        if not docs:
            return "No relevant content found in the uploaded documents."
        return "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source_file', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        )

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get a grounded answer from uploaded documents.

        Args:
            question: User's natural language question.

        Returns:
            Dict with 'answer' (str) and 'sources' (list of dicts).
        """
        is_summary = any(kw in question.lower() for kw in _SUMMARY_KEYWORDS)

        if is_summary:
            context, docs = self._retrieve_for_summary()
        else:
            docs = self.vector_store.similarity_search(question, k=4)
            context = self._format_docs(docs)

        try:
            response = self.chain.invoke({
                "question": question,
                "context": context,
                "chat_history": self.chat_history,
            })
            answer = response.content

            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=answer))

            sources = [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata,
                }
                for doc in docs
            ]
            return {"answer": answer, "sources": sources}

        except Exception as e:  # pylint: disable=broad-except
            return {"answer": f"Error processing question: {e}", "sources": []}

    def ask_streaming(self, question: str) -> None:
        """Streaming version of ask — to be implemented in Phase 2."""
        raise NotImplementedError(
            f"Streaming is not yet implemented. Question was: {question}"
        )
