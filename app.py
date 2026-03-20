"""
Main Streamlit application for Document-Aware AI Assistant.
Provides UI for document upload, processing, and Q&A.
"""

import os

# Must be set before any chromadb import (includes LangChain's internal Chroma).
# Suppresses the PostHog API mismatch error on every startup.
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

import streamlit as st
import pandas as pd
from datetime import datetime
import uuid
from typing import Optional

from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from chat_engine import RAGChatEngine, get_active_provider
from database import DatabaseManager
from utils import (
    setup_directories, validate_file_type, 
    generate_file_hash, logger, format_timestamp
)

# Page configuration
st.set_page_config(
    page_title="Document AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #D4EDDA;
        color: #155724;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
    }
    .assistant-message {
        background-color: #F5F5F5;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStoreManager()
    if 'chat_engine' not in st.session_state:
        st.session_state.chat_engine = None
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'processed_document_ids' not in st.session_state:
        st.session_state.processed_document_ids = []

# Sidebar
def render_sidebar():
    """Render sidebar with configuration options."""
    with st.sidebar:
        st.markdown("""
            <div style="
                background: linear-gradient(135deg, #1E88E5, #1565C0);
                padding: 18px 12px;
                border-radius: 8px;
                text-align: center;
                margin-bottom: 8px;
            ">
                <div style="font-size: 2.2rem; line-height: 1;">🤖</div>
                <div style="color: white; font-size: 1.05rem; font-weight: 700;
                            margin-top: 6px; letter-spacing: 0.3px;">
                    Document AI Assistant
                </div>
                <div style="color: #BBDEFB; font-size: 0.72rem; margin-top: 3px;">
                    Powered by RAG + LangChain
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Active LLM provider badge
        provider = get_active_provider()
        badge_color = {
            "Groq": "#2E7D32",
            "OpenAI": "#6A1B9A",
            "Local": "#E65100"
        }.get(provider["provider"], "#37474F")

        st.markdown(f"""
            <div style="
                background-color: {badge_color};
                color: white;
                padding: 5px 10px;
                border-radius: 5px;
                font-size: 0.75rem;
                text-align: center;
                margin-bottom: 12px;
            ">
                {provider['icon']} {provider['provider']} &nbsp;·&nbsp;
                <code style="color: #ffffffcc;">{provider['model']}</code>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("## 📁 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF or Text files",
            type=['pdf', 'txt', 'md'],
            accept_multiple_files=True,
            help="Upload documents you want the AI to reference"
        )
        
        # Chunking configuration
        st.markdown("### ⚙️ Processing Settings")
        chunk_size = st.slider(
            "Chunk Size",
            min_value=200,
            max_value=2000,
            value=1000,
            step=100,
            help="Size of text chunks for processing"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=400,
            value=200,
            step=50,
            help="Overlap between chunks"
        )
        
        # Process button
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            if uploaded_files:
                process_documents(uploaded_files, chunk_size, chunk_overlap)
            else:
                st.error("Please upload at least one document")
        
        # Document statistics
        if st.session_state.documents_processed:
            st.markdown("### 📊 Statistics")
            stats = st.session_state.db_manager.get_document_stats()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats.get('total_docs', 0))
            with col2:
                st.metric("Total Chunks", stats.get('avg_chunks', 0))
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

def process_documents(uploaded_files, chunk_size, chunk_overlap):
    """
    Process uploaded documents and create vector store.
    """
    with st.spinner("Processing documents..."):
        all_chunks = []
        
        # Update processor settings
        st.session_state.processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        progress_bar = st.progress(0)
        
        max_file_size_mb = int(os.getenv('MAX_FILE_SIZE_MB', 50))
        max_file_size_bytes = max_file_size_mb * 1024 * 1024

        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # Read file content
                file_content = uploaded_file.getvalue()

                # 1.6 — File size limit enforcement
                if len(file_content) > max_file_size_bytes:
                    st.warning(
                        f"Skipped **{uploaded_file.name}**: "
                        f"File size ({len(file_content) // (1024*1024)}MB) "
                        f"exceeds the {max_file_size_mb}MB limit."
                    )
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    continue

                # Validate file type
                if not validate_file_type(uploaded_file.name):
                    st.warning(f"Skipped **{uploaded_file.name}**: Unsupported file type.")
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    continue

                # 1.5 — Explicit duplicate file detection
                file_hash = generate_file_hash(file_content)
                existing = st.session_state.db_manager.document_exists(file_hash)
                if existing:
                    st.warning(
                        f"Skipped **{uploaded_file.name}**: Already uploaded as "
                        f"**{existing['file_name']}** (duplicate detected)."
                    )
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    continue

                # Process document
                result = st.session_state.processor.process_document(
                    file_content,
                    uploaded_file.name
                )

                all_chunks.extend(result['chunks'])

                # 1.2 — Record in database and capture real document ID
                doc_id = st.session_state.db_manager.add_document_record(
                    file_name=uploaded_file.name,
                    file_hash=file_hash,
                    file_size=len(file_content),
                    chunk_count=result['total_chunks'],
                    metadata=result['metadata']
                )
                st.session_state.processed_document_ids.append(doc_id)

                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                logger.error(f"Processing error: {e}")
        
        if all_chunks:
            # Create vector store
            st.session_state.vector_store.create_vector_store(all_chunks)
            
            # Initialize chat engine
            st.session_state.chat_engine = RAGChatEngine(st.session_state.vector_store)
            
            st.session_state.documents_processed = True
            
            st.success(f"✅ Successfully processed {len(uploaded_files)} documents into {len(all_chunks)} chunks!")
        else:
            st.error("No documents were successfully processed")

# Main chat interface
def render_chat_interface():
    """Render the main chat interface."""
    st.markdown("<h1 class='main-header'>🤖 Document-Aware AI Assistant</h1>", unsafe_allow_html=True)
    
    # Check if documents are processed
    if not st.session_state.documents_processed:
        st.info("👈 Please upload and process documents from the sidebar to start chatting")
        
        # Show features
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📄 Document Upload")
            st.write("Upload PDFs and text files")
        with col2:
            st.markdown("### 🔍 Smart Retrieval")
            st.write("AI finds relevant information")
        with col3:
            st.markdown("### 💬 Contextual Answers")
            st.write("Get answers based on your documents")
        
        return
    
    # Chat messages container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources for assistant messages
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {i}:** {source['metadata'].get('source_file', 'Unknown')}")
                            st.markdown(f"*{source['content']}*")
                            st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.ask(prompt)
                
                if "error" in response["answer"].lower():
                    st.error(response["answer"])
                else:
                    st.markdown(response["answer"])
                    
                    # Show sources
                    if response.get("sources"):
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(response["sources"], 1):
                                st.markdown(f"**Source {i}**")
                                st.markdown(f"*From: {source['metadata'].get('source_file', 'Unknown')}*")
                                st.markdown(f"```\n{source['content']}\n```")
                                st.markdown("---")
        
        # Save to session
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response.get("sources", [])
        })
        
        # Save to database with real document IDs
        st.session_state.db_manager.add_chat_record(
            session_id=st.session_state.session_id,
            question=prompt,
            answer=response["answer"],
            document_ids=st.session_state.processed_document_ids,
            metadata={"source_count": len(response.get("sources", []))}
        )

# Analytics tab
def render_analytics():
    """Render analytics and history tab."""
    st.markdown("## 📊 Analytics & History")
    
    # Get chat history
    history = st.session_state.db_manager.get_chat_history(
        st.session_state.session_id
    )
    
    if history:
        # Convert to DataFrame for better display
        df = pd.DataFrame(history)
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Chat History",
            data=csv,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No chat history yet")

# Main app
def main():
    """Main application entry point."""
    # Initialize
    setup_directories()
    init_session_state()
    
    # Create tabs
    tab1, tab2 = st.tabs(["💬 Chat", "📊 Analytics"])
    
    with tab1:
        render_sidebar()
        render_chat_interface()
    
    with tab2:
        render_analytics()

if __name__ == "__main__":
    main()