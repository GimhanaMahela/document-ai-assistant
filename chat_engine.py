"""
RAG implementation using a LangGraph ReAct agent.
The agent decides when to search documents or summarise them,
and maintains conversation history across turns.

LLM priority: Groq (llama-3.3-70b-versatile) → OpenAI → Local Flan-T5
"""

# stdlib
import os
from typing import Dict, Any, List

# third-party
import streamlit as st
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()


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


# ── ReAct Agent Chat Engine ───────────────────────────────────────────────────

class RAGChatEngine:
    """
    ReAct agent for document-based Q&A.

    The agent reasons step-by-step and decides which tool to call:
      - document_search: similarity search over uploaded documents
      - document_summariser: retrieves all chunks for a full summary

    Conversation history is maintained across turns in a session.

    LLM selection priority:
        1. Groq   — if GROQ_API_KEY is set in .env  (fastest, free tier)
        2. OpenAI — if OPENAI_API_KEY is set in .env
        3. Flan-T5 — local fallback, no API key required
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self._last_retrieved_docs: List = []
        self.chat_history: List = []
        self.llm = self._get_llm()
        self.agent = self._build_agent()

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

    def _build_tools(self) -> List:
        """Build the tool list the agent can call."""
        vector_store = self.vector_store

        # Use a list so the closure can append to it from within the tool.
        retrieved_bucket: List = []

        @tool
        def document_search(query: str) -> str:
            """Search the uploaded documents for information relevant to the query.
            Use this whenever the user asks a specific question about the documents."""
            docs = vector_store.similarity_search(query, k=4)
            retrieved_bucket.extend(docs)
            if not docs:
                return "No relevant content found in the uploaded documents."
            parts = []
            for doc in docs:
                src = doc.metadata.get("source_file", "unknown")
                parts.append(f"[Source: {src}]\n{doc.page_content}")
            return "\n\n---\n\n".join(parts)

        @tool
        def document_summariser() -> str:
            """Retrieve the full content of all uploaded documents so you can write
            a comprehensive summary. Use this when the user asks for a summary or
            overview of the documents."""
            try:
                data = vector_store.vector_store.get()
                texts = data.get("documents", [])[:40]
                metadatas = data.get("metadatas", []) or []
                if not texts:
                    return "No documents found in the vector store."
                parts = []
                for text, meta in zip(texts, metadatas):
                    src = (meta or {}).get("source_file", "unknown")
                    parts.append(f"[Source: {src}]\n{text}")
                return "\n\n---\n\n".join(parts)
            except Exception as e:  # pylint: disable=broad-except
                return f"Could not retrieve documents for summarisation: {e}"

        # Store the bucket reference so ask() can read collected docs.
        self._retrieved_bucket = retrieved_bucket
        return [document_search, document_summariser]

    def _build_agent(self):
        """Build the LangGraph ReAct agent."""
        tools = self._build_tools()
        return create_react_agent(self.llm, tools)

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get a grounded answer from uploaded documents.

        Args:
            question: User's natural language question.

        Returns:
            Dict with 'answer' (str) and 'sources' (list of dicts).
        """
        if not self.agent:
            return {"answer": "Agent not initialised.", "sources": []}

        # Reset per-turn doc bucket.
        self._retrieved_bucket.clear()

        messages = self.chat_history + [HumanMessage(content=question)]

        try:
            response = self.agent.invoke({"messages": messages})
            final_message = response["messages"][-1]
            answer = final_message.content

            # Persist full message list for the next turn.
            self.chat_history = response["messages"]

            sources = [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata,
                }
                for doc in self._retrieved_bucket
            ]
            return {"answer": answer, "sources": sources}

        except (ValueError, RuntimeError, KeyError) as e:
            return {"answer": f"Error processing question: {e}", "sources": []}

    def ask_streaming(self, question: str) -> None:
        """Streaming version of ask — to be implemented in Phase 2."""
        raise NotImplementedError(
            f"Streaming is not yet implemented. Question was: {question}"
        )
