# Document-Aware AI Assistant — Project Overview

> A hands-on learning project covering RAG, vector databases, LLMs, and full-stack AI development with Python.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technical Stack Deep Dive](#3-technical-stack-deep-dive)
4. [How RAG Works — Step by Step](#4-how-rag-works--step-by-step)
5. [Key Concepts Explained](#5-key-concepts-explained)
6. [Component Breakdown](#6-component-breakdown)
7. [Installation Guide](#7-installation-guide)
8. [Actual Versions Used (Fixed)](#8-actual-versions-used-fixed)

---

## 1. Project Overview

### 1.1 What This Project Does

**Document-Aware AI Assistant** is a **Retrieval-Augmented Generation (RAG)** application that lets users:

- Upload documents (PDF, TXT, Markdown)
- Ask natural language questions about those documents
- Get accurate, context-grounded answers with source citations

The AI only answers from the content of the uploaded documents — it does **not** hallucinate or use outside knowledge.

### 1.2 Key Features

| Feature | Description |
|---|---|
| 📄 Multi-format support | Upload PDF, TXT, and Markdown files |
| 🔍 Smart chunking | Documents are split into overlapping chunks for better retrieval |
| 💬 Context-aware Q&A | Answers are generated from relevant document chunks only |
| 📚 Source attribution | Every answer shows which part of which document it came from |
| 📊 Analytics dashboard | Chat history stored and exportable as CSV |
| 💾 Persistent storage | Vector store and chat history survive app restarts |

### 1.3 Why Build This?

| Use Case | Example |
|---|---|
| Knowledge Management | Turn a 200-page manual into a chatbot |
| Research Assistance | Query multiple research papers at once |
| Customer Support | Build a support bot grounded in product docs |
| Compliance | Ensure answers are based only on verified policies |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│ Document Processor│────▶│  Vector Store   │
│   (Frontend)    │◀────│   (Backend)       │◀────│   (ChromaDB)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Chat Engine    │────▶│      LLM          │     │   SQLite DB     │
│   (RAG Chain)   │◀────│  (Flan-T5/OpenAI) │     │   (Metadata)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

**What each block does:**
- **Streamlit UI** — the web interface the user sees and interacts with
- **Document Processor** — extracts text from files and splits it into chunks
- **Vector Store** — stores chunks as numerical vectors for semantic search
- **Chat Engine** — orchestrates retrieval + LLM to produce answers
- **LLM** — the language model that generates human-readable answers
- **SQLite DB** — stores document metadata and chat history

### 2.2 Data Flow

```
Document Upload → Text Extraction → Chunking → Embeddings → Vector Store
                                                                    │
User Question ──▶ Question Embedding ──▶ Similarity Search ──▶ Top-K Chunks
                                                                    │
                                                    ┌───────────────┘
                                                    ▼
                                          LLM (chunks + question) ──▶ Answer
```

**Plain English version:**
1. You upload a PDF → text is extracted and split into small chunks
2. Each chunk is converted into a vector (list of numbers) capturing its meaning
3. Vectors are saved in ChromaDB
4. When you ask a question, the question is also converted to a vector
5. The system finds the chunks whose vectors are most similar to your question
6. Those chunks are passed to the LLM with your question as context
7. The LLM generates an answer based only on those chunks

### 2.3 Component Interaction

```
┌─────────────┐
│    app.py   │  ← Entry point, Streamlit UI
│  (Streamlit)│
└──────┬──────┘
       │
       ├─────────────┬──────────────┬──────────────┐
       ▼             ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ document_   │ │ vector_     │ │ chat_       │ │ database.   │
│ processor   │ │ store.py    │ │ engine.py   │ │ py          │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
  PDF/TXT/MD    ChromaDB/FAISS   Flan-T5/OpenAI   SQLite
```

---

## 3. Technical Stack Deep Dive

### 3.1 Technology Table

| Component | Technology | Version Used | Purpose |
|---|---|---|---|
| Language | Python | 3.10+ | Primary programming language |
| Web UI | Streamlit | 1.31.0 | Browser-based UI without writing HTML/JS |
| RAG Framework | LangChain | 0.1.0 | Connects all AI components together |
| Community Integrations | LangChain Community | 0.0.9 | ChromaDB, HuggingFace integrations |
| Vector Database | ChromaDB | 0.5.23 | Stores and searches vector embeddings |
| Vector Index | FAISS | 1.8.0 | Fast similarity search (Facebook AI) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | — | Converts text to vectors (free, local) |
| LLM | Google Flan-T5-Base | — | Generates answers (free, local) |
| PDF Reader | PyPDF | 3.17.4 | Extracts text from PDF files |
| ORM / DB | SQLAlchemy + SQLite | 2.0.25 | Stores metadata and chat history |
| Data Analysis | Pandas | 2.1.4 | Handles chat history as DataFrames |
| Config | python-dotenv | 1.0.0 | Reads `.env` file for settings |

---

## 4. How RAG Works — Step by Step

RAG stands for **Retrieval-Augmented Generation**. It solves the biggest problem with LLMs: they can only answer from what they were trained on, and they hallucinate. RAG gives the LLM fresh, specific information at query time.

### Step 1 — Document Ingestion

```
PDF file
   │
   ▼
PyPDF (text extraction)
   │
   ▼
Raw text: "The quarterly revenue was $4.2M in Q3..."
   │
   ▼
LangChain RecursiveCharacterTextSplitter
   │   chunk_size=1000, chunk_overlap=200
   ▼
Chunk 1: "The quarterly revenue was $4.2M..."  (characters 0–1000)
Chunk 2: "...was $4.2M in Q3. Operating costs..." (characters 800–1800)
Chunk 3: ...
```

> **Why overlap?** If a key sentence falls at the boundary between two chunks, overlap ensures it appears in at least one complete chunk — preventing it from being split and losing context.

### Step 2 — Embedding Generation

```
Chunk: "The quarterly revenue was $4.2M in Q3"
   │
   ▼
HuggingFace all-MiniLM-L6-v2 (embedding model)
   │
   ▼
Vector: [0.021, -0.134, 0.087, 0.456, ..., -0.023]  ← 384 numbers
```

> **What is a vector?** A vector is a list of floating-point numbers that represents the *meaning* of a piece of text. Similar meanings produce vectors that are close to each other in 384-dimensional space. This is called a **semantic embedding**.

> **Why all-MiniLM-L6-v2?** It's a small, fast, free model from Sentence Transformers that produces high-quality 384-dimensional embeddings. "MiniLM" = Mini Language Model, "L6" = 6 transformer layers, "v2" = version 2.

### Step 3 — Vector Storage (ChromaDB)

```
Chunk text + Vector + Metadata (filename, chunk index)
   │
   ▼
ChromaDB (persistent storage on disk at data/chroma_db/)
   │
   ├── Collection: "documents"
   │     ├── id: "doc1_chunk0"  vector: [...] metadata: {source: "report.pdf"}
   │     ├── id: "doc1_chunk1"  vector: [...] metadata: {source: "report.pdf"}
   │     └── ...
```

> **What is ChromaDB?** ChromaDB is a vector database — a database designed specifically to store vectors and find the nearest neighbours efficiently. Unlike regular databases (which match exact values), ChromaDB finds vectors that are *semantically close* to a query vector.

### Step 4 — Query Processing

```
User question: "What was the revenue in Q3?"
   │
   ▼
Same embedding model (all-MiniLM-L6-v2)
   │
   ▼
Question vector: [0.019, -0.128, 0.092, 0.441, ..., -0.018]
   │
   ▼
ChromaDB similarity_search(question_vector, k=4)
   │   Finds 4 chunks whose vectors are closest to the question vector
   ▼
Retrieved chunks:
  Chunk A: "The quarterly revenue was $4.2M in Q3..."
  Chunk B: "Q3 performance exceeded targets by 12%..."
  Chunk C: "Revenue breakdown: Q1 $3.1M, Q2 $3.8M, Q3 $4.2M..."
  Chunk D: "The CFO reported strong Q3 results..."
```

> **Cosine Similarity** is the metric used. It measures the angle between two vectors — the smaller the angle, the more similar the meaning. A cosine similarity of 1.0 = identical meaning, 0.0 = completely unrelated.

### Step 5 — Answer Generation (LLM)

```
Prompt sent to Flan-T5:

  "You are a helpful AI assistant answering questions based on the provided context.

  Context from documents:
  [Chunk A text]
  [Chunk B text]
  [Chunk C text]
  [Chunk D text]

  Question: What was the revenue in Q3?

  Answer:"

   │
   ▼
Flan-T5 generates: "The quarterly revenue in Q3 was $4.2M, which exceeded targets by 12%."
```

> **Why constrain to context?** The prompt explicitly instructs the LLM: *"If the answer isn't in the context, say 'I cannot find this information'"*. This prevents hallucination — the model cannot make things up because it's told to only use what's provided.

---

## 5. Key Concepts Explained

### 5.1 LangChain

**LangChain** is a framework that acts as glue between all the AI components. Without it, you'd have to manually connect the embedding model → vector store → LLM → prompt template. LangChain provides pre-built "chains" that wire these together.

Key LangChain components used in this project:

| Component | What it does |
|---|---|
| `RecursiveCharacterTextSplitter` | Splits documents into chunks |
| `HuggingFaceEmbeddings` | Wraps the embedding model |
| `Chroma` | Wraps ChromaDB as a LangChain vector store |
| `RetrievalQA` | A chain that retrieves docs then asks the LLM |
| `PromptTemplate` | Defines the structure of the prompt sent to the LLM |
| `HuggingFacePipeline` | Wraps a local HuggingFace model as a LangChain LLM |

```python
# Example: RetrievalQA chain connects retriever + LLM + prompt automatically
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,                          # The language model
    retriever=vector_store.as_retriever(search_kwargs={"k": 4}),  # Fetch 4 chunks
    chain_type_kwargs={"prompt": prompt},   # Custom prompt template
    return_source_documents=True       # Include source chunks in the response
)
```

### 5.2 Vector Embeddings

An **embedding** is a numerical representation of text meaning. The key property: texts with similar meaning have similar vectors.

```
"The dog ran fast"     → [0.12, -0.34, 0.87, ...]
"The puppy sprinted"   → [0.11, -0.33, 0.85, ...]  ← very close!
"Stock market crashed" → [-0.45, 0.22, -0.61, ...]  ← very different
```

This is why you can search a document with a question phrased differently from the source text — the embedding captures *meaning*, not exact words.

### 5.3 FAISS vs ChromaDB

Both are used for vector similarity search, but they serve different roles:

| | FAISS | ChromaDB |
|---|---|---|
| Made by | Facebook AI Research | Chroma |
| Type | In-memory index | Full vector database |
| Persistence | Manual (save/load files) | Automatic (SQLite-backed) |
| Metadata | Limited | Rich metadata support |
| Best for | Speed, large scale | Ease of use, persistence |
| Used here for | Alternative backend option | Primary vector store |

### 5.4 Flan-T5 (Local LLM)

**Flan-T5** is a free, open-source language model from Google. "Flan" = Fine-tuned LAnguage Net, "T5" = Text-To-Text Transfer Transformer.

| Variant | Parameters | VRAM | Quality |
|---|---|---|---|
| flan-t5-small | 80M | ~320MB | Basic |
| **flan-t5-base** ← used | **250M** | **~1GB** | **Good for Q&A** |
| flan-t5-large | 780M | ~3GB | Better |
| flan-t5-xl | 3B | ~12GB | Very Good |

> **Why T5 architecture?** T5 is a seq2seq (sequence-to-sequence) model — it takes text in and produces text out. This makes it naturally suited for Q&A tasks, unlike GPT-style models which are decoder-only.

### 5.5 Streamlit

**Streamlit** turns Python scripts into interactive web apps without writing HTML, CSS, or JavaScript.

```python
# A complete interactive web app in 3 lines:
import streamlit as st
text = st.text_input("Ask a question")
st.write(f"You asked: {text}")
```

Key Streamlit concepts used:
- `st.session_state` — persists data between reruns (like React state)
- `st.chat_message` — built-in chat bubble UI
- `st.spinner` — loading indicator while the LLM processes
- `st.file_uploader` — handles file uploads
- `st.tabs` — creates tabbed navigation

### 5.6 SQLAlchemy & SQLite

**SQLite** is a file-based relational database — no server needed, the entire DB is a single `.db` file.

**SQLAlchemy** is a Python ORM (Object-Relational Mapper) that lets you interact with the database using Python objects instead of raw SQL.

```python
# Without SQLAlchemy (raw SQL):
cursor.execute("INSERT INTO documents VALUES (?, ?, ?)", (name, hash, size))

# With SQLAlchemy (Python objects):
doc = DocumentRecord(name=name, hash=hash, size=size)
session.add(doc)
session.commit()
```

---

## 6. Component Breakdown

### `app.py` — Main Application
- Entry point for Streamlit
- Manages UI layout, tabs, sidebar, and session state
- Coordinates all other components

### `document_processor.py` — Text Extraction & Chunking
- Uses **PyPDF** to extract text from PDF files
- Uses **LangChain's RecursiveCharacterTextSplitter** to chunk text
- Produces `Document` objects (text + metadata)

### `vector_store.py` — Embeddings & Vector Database
- Loads the **HuggingFace embedding model** (`all-MiniLM-L6-v2`)
- Uses **ChromaDB** to persist vectors to disk
- Provides `similarity_search()` to find relevant chunks

### `chat_engine.py` — RAG Question Answering
- Loads **Flan-T5** via HuggingFace Transformers
- Wraps it in a **LangChain RetrievalQA chain**
- Takes a question → retrieves chunks → generates answer

### `database.py` — Metadata Persistence
- Uses **SQLAlchemy** with SQLite
- Stores document records (filename, hash, chunk count)
- Stores chat records (question, answer, session, timestamp)

### `utils.py` — Shared Utilities
- Directory setup (`data/`, `logs/`)
- File type validation
- MD5 file hash generation (to detect duplicate uploads)
- Logger configuration

---

## 7. Installation Guide

### 7.1 Prerequisites

```bash
# Check Python version (need 3.10+)
python3 --version

# Check pip
pip --version
```

### 7.2 Step-by-Step Setup

**Step 1: Create project directory**
```bash
mkdir document-ai-assistant
cd document-ai-assistant
mkdir -p data/uploaded data/chroma_db logs
```

**Step 2: Set up virtual environment**
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# You should see (venv) in your terminal prompt
```

> **Why a virtual environment?** It isolates this project's packages from your system Python, preventing version conflicts. Each project gets its own sandbox.

**Step 3: Install build dependencies (Linux)**
```bash
# Required to compile some packages from source
sudo apt install build-essential g++ python3-dev
```

**Step 4: Install Python packages**
```bash
pip install -r requirements.txt
pip install sentence-transformers    # For free local embeddings
```

**Step 5: Configure environment**

Create a `.env` file in the project root:
```env
# Leave blank to use free local models (recommended)
# OPENAI_API_KEY=sk-your-key-here

# App settings (optional — defaults are fine)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
PERSIST_DIRECTORY=./data/chroma_db
DATABASE_PATH=./chat_history.db
LOG_LEVEL=INFO
```

**Step 6: Run the app**
```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

> **First run note:** The first time you process a document, it will download the HuggingFace models (~500MB total). After that they are cached locally.

---

## 8. Actual Versions Used (Fixed)

The original requirements had version conflicts. Here are the corrected versions that actually work together:

```txt
streamlit==1.31.0
langchain==0.1.0
langchain-community==0.0.9      # Fixed: 0.1.0 doesn't exist
chromadb==0.5.23                # Fixed: 0.4.22 requires C++ compilation
faiss-cpu==1.8.0.post1          # Fixed: 1.7.4 not available for Python 3.12
pypdf==3.17.4
python-dotenv==1.0.0
openai>=1.30.0                  # Fixed: 1.12.0 incompatible with newer httpx
tiktoken==0.5.2
sqlalchemy==2.0.25
pandas==2.1.4
sentence-transformers            # Added: needed for HuggingFace embeddings
```

### Why These Versions Conflicted

| Original | Problem | Fix |
|---|---|---|
| `langchain-community==0.1.0` | This version does not exist on PyPI | Changed to `0.0.9` |
| `chromadb==0.4.22` | Requires `chroma-hnswlib` compiled from C++ source; missing `Python.h` headers | Upgraded to `0.5.23` which has pre-built wheels |
| `faiss-cpu==1.7.4` | Not available for Python 3.12 — minimum is `1.8.0` | Upgraded to `1.8.0.post1` |
| `openai==1.12.0` | Passes `proxies` keyword to `httpx.Client` which newer httpx removed | Upgraded to `>=1.30.0` |

---

*This document covers the full architecture, all technologies used, and the RAG pipeline in depth. Use it as a reference while exploring the source code.*
